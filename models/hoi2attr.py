import torch
from torch import nn
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from collections import defaultdict
from util.misc import (NestedTensor, nested_tensor_from_tensor_list)
from .layers.roi_align import ROIAlign


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

    def forward(self, model, samples, targets):
        
        object_boxes = [[i,box] for i, target in enumerate(targets) for box in target['boxes']]
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples) 
        
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
        
        
        #boxes (Tensor[K, 5] or List[Tensor[L, 4]]) – the box coordinates in (x1, y1, x2, y2) format where the regions will be taken from. The coordinate must satisfy 0 <= x1 < x2 and 0 <= y1 < y2. If a single Tensor is passed, then the first column should contain the batch index. If a list of Tensors is passed, then each Tensor will correspond to the boxes for an element i in a batch
        #import pdb; pdb.set_trace()
        input_sample = torch.ones(1, 3, 10, 10, dtype=torch.float32)
        rois = torch.zeros(1, 5, dtype=torch.float32)
        pooled_feature = box_roi_align(input = input_sample, rois = rois) #(B,C,W,H) , xyxy

def build_attrclass(args, backbone, transformer):
    return Attrclassifier(
        transformer = transformer,
        backbone = backbone
        )
