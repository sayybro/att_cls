import argparse
import torch
import cv2
#from main import get_args_parser
from models import build_model
from datasets import build_dataset
from torch.utils.data import DataLoader, DistributedSampler
import util.misc as utils


def get_args_parser():
    parser = argparse.ArgumentParser('Set annotation visualizer', add_help=False)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--dataset', default='vaw',type=str)
    parser.add_argument('--dataset_file', default='vaw',type=str)
    parser.add_argument('--hoi_', default='vaw',type=str)
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--num_workers', default=2, type=int)
    return parser

def make_train_dataset(args):
    dataset_train = build_dataset(image_set='train', args=args)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    return data_loader_train


class vis_anno():

    def __init__(self):
        
        return 
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('annotation visualize', parents=[get_args_parser()])
    args = parser.parse_args()
    data_loader_train = make_train_dataset(args)

    # model, _, postprocessors = build_model(args)
    # demo = Demo(args)
    # demo.save_video(args)
