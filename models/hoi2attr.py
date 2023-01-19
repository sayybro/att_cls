import torch
from torch import nn
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from collections import defaultdict
from util.misc import (NestedTensor, nested_tensor_from_tensor_list)

class SetCriterionATTR(nn.Module):

    def __init__(self):
        super().__init__()
        self.weight_dict = defaultdict(str) #will be updated

    def forward(self, outputs, targets):
        losses = defaultdict(str)
        return losses


class Attrclassifier(nn.Module):

    def __init__(self, backbone, transformer):
        super().__init__()
        self.transformer_encoder = transformer.encoder 
        self.backbone = backbone

    def forward(self, model, samples, hoi_output):
        sub_boxes = hoi_output['pred_sub_boxes']
        obj_boxes = hoi_output['pred_obj_boxes']
        obj_logits = hoi_output['pred_obj_logits']
        
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        import pdb; pdb.set_trace()
        src = model.input_proj(src) #torch.Size([8, 256, 32, 32])
        src = src.flatten(2).permute(2, 0, 1) #torch.Size([1024, 8, 256])
        pos_embed = pos[-1].flatten(2).permute(2, 0, 1) #torch.Size([1024, 8, 256])
        mask = mask.flatten(1) #torch.Size([8, 1024])
        memory = self.transformer_encoder(src, src_key_padding_mask=mask, pos=pos_embed) 
        encoder_output = memory.permute(1, 2, 0) #torch.Size([8, 256, 1024])


def build_attrclass(args, backbone, transformer):
    return Attrclassifier(
        transformer = transformer,
        backbone = backbone
        )
