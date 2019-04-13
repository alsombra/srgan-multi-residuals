import argparse
import torch
from data_loader import get_loader
import os
from solver import Solver
import time
import pickle

import argparse
import os
import numpy as np
import math
import itertools
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

from model import *

import torch.nn as nn
import torch.nn.functional as F
import torch

torch.cuda.set_device(1)

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(config, scope):
    # Create directories if not exist.
    
    #os.makedirs("images", exist_ok=True)
    #os.makedirs("saved_models", exist_ok=True)

    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)
    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    if not os.path.exists('./samples'):
        os.makedirs('./samples')
    if not os.path.exists(config.result_path + '/grids'):
        os.makedirs(config.result_path + '/grids')
    if not os.path.exists(config.result_path + '/HR_images'):
        os.makedirs(config.result_path + '/HR_images')
    if not os.path.exists(config.result_path + '/HR_bicub_images'):
        os.makedirs(config.result_path + '/HR_bicub_images')
    if not os.path.exists(config.result_path + '/HR_HAT_images'):
        os.makedirs(config.result_path + '/HR_HAT_images')
    if not os.path.exists(config.result_path + '/LR_images_snapshot'):
        os.makedirs(config.result_path + '/LR_images_snapshot')  
    
    
    # Data loader
    data_loader = get_loader(config.data_path, config)
    
    # Solver
    
    solver = Solver(data_loader, config)
    
    def load(filename, *args):
        solver.load(filename)

    def save(filename, *args):
        solver.save(filename)

    def evaluate(test_data, output):
        pass

    def decode(input):
        return input
    

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        if config.test_mode == 'single':
            solver.test()
        elif config.test_mode == 'many':
            solver.many_tests()
        elif config.test_mode == 'pick_from_set':
            solver.test_and_error()
        elif config.test_mode == 'evaluate':
            solver.evaluate()
            
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--image_size', type=int, default=64, help='LR image size')  #LR img size
    parser.add_argument('--num_channels', type=int, default=12)
    parser.add_argument('--scale_factor', type=int, default=2)

    # Training settings
    parser.add_argument('--total_step', type=int, default=200000)###
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--residual_loss_function', type=str, default=None, choices=['l1', 'l2'])
    parser.add_argument('--content_loss_function', type=str, default='l2', choices=['l1', 'l2'])#########
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")    
    parser.add_argument("--beta1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")   
    parser.add_argument("--beta2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument('--num_workers', type=int, default=4)  ########
    parser.add_argument('--trained_model', type=str, default=None) ########## mudei pra receber string
    parser.add_argument('--trained_discriminator', type=str, default=None) ########## mudei pra receber string

    # Test
    parser.add_argument('--test_mode', type=str, default='pick_from_set', choices=['single', 'many', 'pick_from_set', 'evaluate'])
    parser.add_argument('--test_image_path', type=str) #Use with a single file for 'single_test' and a folder for 'many_tests'
    parser.add_argument('--evaluation_step', type=int, default=10) #evaluation log print step
    parser.add_argument('--evaluation_size', type=int, default=10) #if evaluation size == -1 takes all test_set

    # Misc
    parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
    
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    
    parser.add_argument('--iteration', type=int, default=0)
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)

    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--model_save_path', type=str, default='./models')
    parser.add_argument('--result_path', type=str, default='./results')

    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=100, help="interval between saving image samples")
    parser.add_argument('--model_save_step', type=int, default=1000)

    # NSML setting
    parser.add_argument('--pause', type=int, default=0)

    config = parser.parse_args()

    # Device selection
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data path
    
    #config.image_path = os.path.join("../data/img_align_celeba/train/")
   
    DATA_PATH = "/data/antonio/%s" % config.dataset_name

    config.data_path = os.path.join(DATA_PATH, config.mode)##################
    print(config)
    print('-------------------------------------------------------------------------')
    print('Agora vou mandar o vars')
    print(vars(config))
    save_obj(vars(config), 'args_dict')
    print('-------------------------------------------------------------------------')
    start = time.time()
    main(config, scope=locals())
    end = time.time()
    print('Tempo total foi {} horas'.format((end - start)/3600))
