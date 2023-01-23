import torch
from torch import nn
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from collections import defaultdict
from util.misc import (NestedTensor, nested_tensor_from_tensor_list)
from .layers.roi_align import ROIAlign
from typing import List

class SetCriterionATTR(nn.Module):

    def __init__(self):
        super().__init__()
        self.weight_dict = defaultdict(str) #will be updated

    def forward(self, outputs, targets):
        losses = defaultdict(str) #will be updated
        return losses


class Attrclassifier(nn.Module):

    def __init__(self, backbone, transformer):
        super().__init__()
        self.transformer_encoder = transformer.encoder  
        self.backbone = backbone 

    def forward(self, model, samples, targets): #sample.tensors : B,C,H,W
        
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples) 
        bbox = targets[0]['boxes'][0].tolist()
        object_boxes = [torch.Tensor([int(i)]+self.convert_bbox(box.tolist())) for i, target in enumerate(targets) for box in target['boxes']]
        box_tensors = torch.stack(object_boxes,0) #[K,5] , K: annotation length in mini-batch
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
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
        box_tensors[...,1], box_tensors[...,3] = feature_W*box_tensors[...,1], feature_W*box_tensors[...,3] 
        box_tensors[...,2], box_tensors[...,4] = feature_H*box_tensors[...,2], feature_H*box_tensors[...,4] 
        pooled_feature = box_roi_align(input = encoder_output, rois = box_tensors.cuda()) #(B,C,W,H) , xyxy
        #import pdb; pdb.set_trace()

        #asdf = box_tensors

    def convert_bbox(self,bbox:List): #annotation bbox (c_x,c_y,w,h)-> (x1,y1,x2,y2)
        #import pdb; pdb.set_trace()
        c_x, c_y, w,h = bbox[0], bbox[1], bbox[2], bbox[3]
        x1,y1 = c_x-(w/2), c_y-(h/2)
        x2,y2 = c_x+(w/2), c_y+(h/2)  
        return [x1,y1,x2,y2]

    def conv2d_layer(self):

        return

    def attr_head(self):

        return

def build_attrclass(args, backbone, transformer):
    return Attrclassifier(
        transformer = transformer,
        backbone = backbone
        )
