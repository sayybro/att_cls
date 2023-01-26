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

    def __init__(self, num_obj_classes, num_att_classes, weight_dict, eos_coef, losses, loss_type, args=None):
        super().__init__()

        assert loss_type =='bce' or loss_type =='focal'

        self.num_obj_classes = num_obj_classes
        self.num_att_classes = num_att_classes
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses= losses
        empty_weight = torch.ones(self.num_obj_classes)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.register_buffer('fed_loss_weight', load_class_freq(freq_weight=0.5))
        self.loss_type = loss_type #'bce' or 'focal'
    
    #attribute class loss
    def loss_att_labels(self, outputs, targets, log=True):

        assert 'att_preds' in outputs
        src_logits = outputs['att_preds']
        target_classes_o = torch.cat([target['pos_att_classes'] for target in targets])
        target_classes = torch.zeros_like(src_logits)
        pos_gt_classes = torch.nonzero(target_classes_o==1)[...,-1]        
        neg_classes_o = torch.cat([t['neg_att_classes'] for t in targets])
        neg_gt_classes = torch.nonzero(neg_classes_o==1)[...,-1]

        box_length = [len(target['boxes']) for target in targets]
        gt_pos = torch.cat([sum(target['pos_att_classes']).unsqueeze(0) if len(target['boxes']) == 1 else torch.tensor(np.tile(sum(target['pos_att_classes']).unsqueeze(0).detach().cpu(),(len(target['boxes']),1))).cuda() for target in targets])
        gt_neg = torch.cat([sum(target['neg_att_classes']).unsqueeze(0) if len(target['boxes']) == 1 else torch.tensor(np.tile(sum(target['neg_att_classes']).unsqueeze(0).detach().cpu(),(len(target['boxes']),1))).cuda() for target in targets])
        

        #how to assign attribute label to box index? to be updated
        if self.loss_type == 'bce':
            loss_att_ce = F.binary_cross_entopry_with_logits(src_logits, gt_pos)

        elif self.loss_type == 'focal':
            src_logits = src_logits.sigmoid()
            loss_att_ce = self._neg_loss(src_logits, gt_pos)

        losses = {'loss_att_ce': loss_att_ce}
        return losses

    def loss_obj_labels(self, outputs, targets, log=True):

        assert 'obj_preds' in outputs

        src_logits = outputs['obj_preds'] 
        target_classes_o = torch.cat([target['labels'] for target in targets])
        loss_obj_ce = F.cross_entropy(src_logits, target_classes_o, self.empty_weight)
        losses = {'loss_att_obj_ce': loss_obj_ce}

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
            'obj_labels':self.loss_obj_labels,
            'att_labels':self.loss_att_labels
        }

        return loss_map[loss](outputs,targets,**kwargs)

    def forward(self, outputs, targets):
        num_attributes = sum(len(t['labels']) for t in targets) 
        num_attributes = torch.as_tensor([num_attributes], dtype=torch.float, device=outputs['att_preds'].device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_attributes)
        num_attributes = torch.clamp(num_attributes / get_world_size(), min=1).item()

        #compute all requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets))

        #auxiliary output
        # if 'aux_outputs' in outputs:
        #     for i, aux_outputs in enumerate(outputs['aux_outputs']):
        return losses
    
class PostProcess_ATT(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, outputs, targets_sizes):
        out_obj_logits, out_att_logits = outputs['pred_obj_logits'], outputs['pred_logits']

        assert len(out_obj_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        # obj_prob = F.softmax(out_obj_logits, -1)
        obj_prob = F.softmax(torch.cat([out_obj_logits[...,:-2], out_obj_logits[...,-1:]],-1),-1)
        obj_scores, obj_labels = obj_prob[..., :-1].max(-1)

        attr_scores = out_att_logits.sigmoid()

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(attr_scores.device)
        
        results = []
        for obj_label, att_label in zip(obj_labels, attr_scores):
            results.append({'labels': obj_label.to('cpu')})
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
        self.fc2 = nn.Linear(256, args.num_obj_classes)
        self.distributed = args.distributed

    def forward(self, model, samples, targets): 

        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples) 
        
        object_boxes = [torch.Tensor([int(i)]+self.convert_bbox(box.tolist())) for i, target in enumerate(targets) for box in target['boxes']]
        box_tensors = torch.stack(object_boxes,0) #[K,5] , K: annotation length in mini-batch
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
        
        #encoder_output : torch.Size([N, 256, 29, 32])
        pooled_feature = box_roi_align(input = encoder_output, rois = box_tensors.cuda()) #(B,C,W,H) , xyxy
        x = self.conv(pooled_feature) #torch.Size([N, 256, 7, 7])
        x = self.avgpool(x) #torch.Size([N, 256, 1, 1])
        x = torch.flatten(x, 1) #torch.Size([N, 256])
        attributes = self.fc1(x) #torch.Size([9, 620])
        objects = self.fc2(x) #torch.Size([9, 81])
        #import pdb; pdb.set_trace()
        return attributes, objects

    def convert_bbox(self,bbox:List): #annotation bbox (c_x,c_y,w,h)-> (x1,y1,x2,y2)
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
