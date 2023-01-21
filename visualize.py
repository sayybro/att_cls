import argparse
import torch
import cv2
from main import get_args_parser
from models import build_model
from datasets import build_dataset


class vis():

    def __init__(self):
    def 

if __name__ == '__main__':
    parser = argparse.ArgumentParser('annotation visualize', parents=[get_args_parser()])
    args = parser.parse_args()
    dataset_train = build_dataset(image_set='train', args=args)
    
    
    build_dataset
    model, _, postprocessors = build_model(args)
    demo = Demo(args)
    demo.save_video(args)
