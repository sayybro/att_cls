import argparse
import torch
import cv2
from models import build_model
from datasets import build_dataset
from torch.utils.data import DataLoader, DistributedSampler
import util.misc as utils
import datasets.transforms as T
from util.misc import NestedTensor


def get_args_parser():
    parser = argparse.ArgumentParser('Set annotation visualizer', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--dataset', default='vaw',type=str)
    parser.add_argument('--dataset_file', default='vaw',type=str)
    parser.add_argument('--hoi_', default='vaw',type=str)
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--output_dir', default='output_dir/sample', type=str)
    
    return parser


class Visualize():

    def __init__(self,args):
        self.data_loader = self.make_train_dataset(args)
        self.color = (0,255,0) 
        self.output_path = args.output_dir + '.jpg'
        
    def make_train_dataset(self, args):
        dataset_train = build_dataset(image_set='train', args=args)
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=True)
        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                    collate_fn=utils.collate_fn, num_workers=args.num_workers)
        return data_loader_train

    def dataloader2sample(self, dataloader):
        iterator = iter(dataloader)
        sample, target = next(iterator)
        return sample, target

    def unnormalize(self, sample:NestedTensor):
        invTrans = T.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
            std=[1/0.229, 1/0.224, 1/0.255]
        )
        inv_img = invTrans(sample.tensors.squeeze(0))
        inv_sample = 255*inv_img[0].numpy().transpose(1, 2, 0)
        return inv_sample

    def convert_bbox(self, sample, target):
        boxes = target[0]['boxes'][0] #bounding box for visualize
        c_x, c_y, w,h = boxes[0], boxes[1], boxes[2], boxes[3]
        orig_h,orig_w = sample.shape[0], sample.shape[1]
        box_w, box_h = w*orig_w, h*orig_h
        box_cx, box_cy = c_x*orig_w, c_y*orig_h
        (x1,y1) = (int(box_cx-(box_w//2)), int(box_cy-(box_h//2)))
        (x2,y2) = (int(box_cx+(box_w//2)), int(box_cy+(box_h//2)))   
        return (x1,y1), (x2,y2)

    def visualize(self):
        img, target = self.dataloader2sample(self.data_loader)
        inv_sample = self.unnormalize(img)  
        inv_sample = inv_sample[...,::-1] #RGB to BGR
        (x1,y1), (x2,y2) = self.convert_bbox(inv_sample,target)
        inv_sample = cv2.rectangle(inv_sample, (x1,y1), (x2,y2), self.color)
        cv2.imwrite(self.output_path,inv_sample)
        import pdb; pdb.set_trace()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('annotation visualize', parents=[get_args_parser()])
    args = parser.parse_args()
    vis = Visualize(args)
    vis.visualize()