import torch
from torch import nn
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from collections import defaultdict
from util.misc import NestedTensor, nested_tensor_from_tensor_list,get_world_size,is_dist_avail_and_initialized
from .layers.roi_align import ROIAlign
from typing import List
from util.fed import load_class_freq, get_fed_loss_inds
import torch.nn.functional as F
import numpy as np

class SetCriterionATT(nn.Module):

    def __init__(self, num_att_classes, weight_dict, losses, loss_type, args=None):
        super().__init__()

        assert loss_type =='bce' or loss_type =='focal'

        self.num_att_classes = num_att_classes
        self.weight_dict = weight_dict
        self.losses= losses
        self.register_buffer('fed_loss_weight', load_class_freq(freq_weight=0.5))
        self.loss_type = loss_type #'bce' or 'focal'
    
    #attribute class loss
    def loss_att_labels(self, outputs, targets, log=True):
        
        #attribute predictions
        src_logits = outputs['att_preds'] 

        #attribute target
        target_classes = torch.zeros_like(src_logits)

        #only consider samples that have box annotations
        pos_labels, neg_labels = self.postprocess_att(targets)
        
        assert len(pos_labels) == len(src_logits)
        assert len(neg_labels) == len(src_logits)

        pos_batch_index, pos_gt_classes = np.where(pos_labels.detach().cpu()==1) 
        neg_batch_index, neg_gt_classes = np.where(neg_labels.detach().cpu()==1)
                
        #assing 1 to positive label
        target_classes[np.where(pos_labels.detach().cpu()==1)] = 1

        pos_gt_classes = torch.from_numpy(pos_gt_classes).unique()
        neg_gt_classes = torch.from_numpy(neg_gt_classes).unique()
        
        #loss calculation for 50 of 620 attribute classes
        inds = get_fed_loss_inds(
            gt_classes=torch.cat([pos_gt_classes,neg_gt_classes]),
            num_sample_cats=50,
            weight=self.fed_loss_weight,
            C=src_logits.shape[1])

        if self.loss_type == 'bce':
            loss_att_ce = F.binary_cross_entopry_with_logits(src_logits[...,inds], target_classes[...,inds])

        elif self.loss_type == 'focal':
            src_logits = src_logits.sigmoid()
            loss_att_ce = self._neg_loss(src_logits, target_classes)

        losses = {'loss_att_ce': loss_att_ce}
        return losses

    #modified focal loss
    def _neg_loss(self, pred, gt):
        pos_inds = gt.eq(1).float() 
        neg_inds = gt.lt(1).float() 

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1-pred,2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds
        
        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else: 
            loss = loss - (pos_loss + neg_loss) / num_pos      
        return loss

    def get_loss(self, loss, outputs, targets, **kwargs):
        loss_map = {
            'att_labels':self.loss_att_labels
        }

        return loss_map[loss](outputs,targets,**kwargs)

    def postprocess_att(self, targets):

        pos,neg = [], []
        for target in targets:
            #box label : attribute label = 1 : 1
            if (len(target['boxes']) > 0) and (len(target['pos_att_classes']) == (len(target['boxes']))):
                pos.append(target['pos_att_classes'])
                neg.append(target['neg_att_classes'])

            #box label : attribute label = 1 : N
            if (len(target['boxes']) == 1) and (len(target['pos_att_classes']) > (len(target['boxes']))): 
                pos.append(sum(target['pos_att_classes']).unsqueeze(0))
                neg.append(sum(target['pos_att_classes']).unsqueeze(0))

            #when len(box labels) > 1 and len(box_labels) != len(target['pos_att_classes']) can't assign
            if (len(target['boxes']) > 1) and len(target['boxes']) != len(target['pos_att_classes']): #loss 계산 제외 하도록 해야할듯? 
                tmp = torch.from_numpy(np.tile(np.array(-1),(target['boxes'].shape[0],620))).cuda()
                pos.append(tmp)
                neg.append(tmp)
                
        return torch.cat(pos), torch.cat(neg)

    def forward(self, outputs, targets):

        #compute all requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets))

        return losses
    
class PostProcess_ATT(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, outputs, targets_sizes):

        out_att_logits = outputs['pred_logits']

        assert len(out_att_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        attr_scores = out_att_logits.sigmoid()
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(attr_scores.device)
        
        results = []
        for att_label in attr_scores:
            res_dict = {
                'attr_scores' : att_label.to('cpu')
            }
            results[-1].update(res_dict)
    
        return results

class Attrclassifier(nn.Module):

    def __init__(self, args, backbone, transformer):
        super().__init__()
        self.transformer_encoder = transformer.encoder  
        self.backbone = backbone 
        self.conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, args.num_att_classes)
        self.distributed = args.distributed

    def forward(self, model, samples, targets): 

        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples) 
        
        object_boxes = [torch.Tensor([int(i)]+self.convert_bbox(box.tolist())) for i, target in enumerate(targets) for box in target['boxes']]
        box_tensors = torch.stack(object_boxes,0) #[K,5] , K: box annotation length in mini-batch
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        
        if self.distributed:
            src = model.module.input_proj(src)
        else:
            src = model.input_proj(src)

        B,C,H,W = src.shape
        src = src.flatten(2).permute(2, 0, 1) 
        pos_embed = pos[-1].flatten(2).permute(2, 0, 1) 
        mask = mask.flatten(1) 
        memory = self.transformer_encoder(src, src_key_padding_mask=mask, pos=pos_embed) 
        encoder_output = memory.permute(1, 2, 0) 
        encoder_output = encoder_output.view([B,C,H,W]) 
        box_roi_align = ROIAlign(output_size=(7,7), spatial_scale=1.0, sampling_ratio=-1, aligned=True)         
        feature_H, feature_W = encoder_output.shape[2], encoder_output.shape[3]

        #unnormalize box size
        box_tensors[...,1], box_tensors[...,3] = feature_W*box_tensors[...,1], feature_W*box_tensors[...,3] 
        box_tensors[...,2], box_tensors[...,4] = feature_H*box_tensors[...,2], feature_H*box_tensors[...,4] 
        
        #encoder_output : torch.Size([N, 256, H, W])
        pooled_feature = box_roi_align(input = encoder_output, rois = box_tensors.cuda()) #(B,C,W,H) , xyxy
        x = self.conv(pooled_feature) #torch.Size([N, 256, 7, 7])
        x = self.avgpool(x) #torch.Size([N, 256, 1, 1])
        x = torch.flatten(x, 1) #torch.Size([N, 256])
        attributes = self.fc1(x) ##torch.Size([N, 620])
        return attributes

    def convert_bbox(self,bbox:List): #annotation bbox (c_x,c_y,w,h)-> (x1,y1,x2,y2) for roi align
        c_x, c_y, w,h = bbox[0], bbox[1], bbox[2], bbox[3]
        x1,y1 = c_x-(w/2), c_y-(h/2)
        x2,y2 = c_x+(w/2), c_y+(h/2)  
        return [x1,y1,x2,y2]


def build_attrclassifier(args, backbone, transformer):
    return Attrclassifier(
        args = args,
        transformer = transformer,
        backbone = backbone
        )
