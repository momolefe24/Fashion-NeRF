### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
import yaml
from collections import OrderedDict
from VITON.Parser_Based.ACGPN.ACGPN_train.data.data_loader import CreateDataLoader
from VITON.Parser_Based.ACGPN.ACGPN_train.models.models import create_model
import VITON.Parser_Based.ACGPN.ACGPN_train.util.util as util
import os
import numpy as np
import argparse
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import cv2
import datetime
import ipdb


# writer = SummaryWriter('runs/uniform_all')
SIZE=320
NC=14

fix = lambda path: os.path.normpath(path)
opt,root_opt,wandb,sweep_id =None, None, None,None

def generate_label_plain(inputs):
    size = inputs.size()
    pred_batch = []
    for input in inputs:
        input = input.view(1, NC, 256,192)
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
        pred_batch.append(pred)

    pred_batch = np.array(pred_batch)
    pred_batch = torch.from_numpy(pred_batch)
    label_batch = pred_batch.view(size[0], 1, 256,192)

    return label_batch
def morpho(mask,iter):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    new=[]
    for i in range(len(mask)):
        tem=mask[i].squeeze().reshape(256,192,1)*255
        tem=tem.astype(np.uint8)
        tem=cv2.dilate(tem,kernel,iterations=iter)
        tem=tem.astype(float64)
        tem=tem.reshape(1,256,192)
        new.append(tem.astype(float64)/255.0)
    new=np.stack(new)
    return new

def save_checkpoint(model, save_path,opt):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    torch.save(model.cpu().state_dict(), save_path)
    if opt.cuda:
        model.cuda()
   
def get_wandb_image(image, wandb):
    if image.max() <= 1.0:
        image = image*255 
    image_numpy = image.permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
    return wandb.Image(image_numpy)     
        
def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print('no checkpoint')
        raise
    log = model.load_state_dict(torch.load(checkpoint_path), strict=False)

def generate_label_color(inputs, opt):
    label_batch = []
    for i in range(len(inputs)):
        label_batch.append(util.tensor2label(inputs[i], opt.label_nc))
    label_batch = np.array(label_batch)
    label_batch = label_batch * 2 - 1
    input_label = torch.from_numpy(label_batch)

    return input_label
def complete_compose(img,mask,label):
    label=label.cpu().numpy()
    M_f=label>0
    M_f=M_f.astype(int32)
    M_f=torch.FloatTensor(M_f).cuda()
    masked_img=img*(1-mask)
    M_c=(1-mask.cuda())*M_f
    M_c=M_c+torch.zeros(img.shape).cuda()##broadcasting
    return masked_img,M_c,M_f

def compose(label,mask,color_mask,edge,color,noise):
    # check=check>0
    # print(check)
    masked_label=label*(1-mask)
    masked_edge=mask*edge
    masked_color_strokes=mask*(1-color_mask)*color
    masked_noise=mask*noise
    return masked_label,masked_edge,masked_color_strokes,masked_noise

def changearm(data):
    label=data['label']
    arm1=torch.FloatTensor((data['label'].cpu().numpy()==11).astype(int32))
    arm2=torch.FloatTensor((data['label'].cpu().numpy()==13).astype(int32))
    noise=torch.FloatTensor((data['label'].cpu().numpy()==7).astype(int32))
    label=label*(1-arm1)+arm1*4
    label=label*(1-arm2)+arm2*4
    label=label*(1-noise)+noise*4
    return label


def get_root_experiment_runs(root_opt):
    root_opt.experiment_run = root_opt.experiment_run.format(root_opt.experiment_number, root_opt.run_number)
    root_opt.experiment_from_run = root_opt.experiment_from_run.format(root_opt.experiment_from_number, root_opt.run_from_number)
    root_opt.vgg_experiment_from_run = root_opt.vgg_experiment_from_run.format(root_opt.vgg_experiment_from_number, root_opt.vgg_run_from_number)
    root_opt.g1_experiment_from_run = root_opt.g1_experiment_from_run.format(root_opt.g1_experiment_from_number, root_opt.g1_run_from_number)
    root_opt.g2_experiment_from_run = root_opt.g2_experiment_from_run.format(root_opt.g2_experiment_from_number, root_opt.g2_run_from_number)
    root_opt.g_experiment_from_run = root_opt.g_experiment_from_run.format(root_opt.g_experiment_from_number, root_opt.g_run_from_number)
    root_opt.d_experiment_from_run = root_opt.d_experiment_from_run.format(root_opt.d_experiment_from_number, root_opt.d_run_from_number)
    root_opt.unet_experiment_from_run = root_opt.unet_experiment_from_run.format(root_opt.unet_experiment_from_number, root_opt.unet_run_from_number)
    
    return root_opt


def get_root_opt_results_dir(parser, root_opt):
    root_opt.transforms_dir = root_opt.transforms_dir.format(root_opt.dataset_name)
    parser.tensorboard_dir = parser.tensorboard_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    parser.results_dir = parser.results_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    return parser, root_opt

def copy_root_opt_to_opt(parser, root_opt):
    parser.display_count = root_opt.display_count
    parser.cuda = root_opt.cuda
    parser.dataset_name = root_opt.dataset_name
    parser.load_last_step = root_opt.load_last_step if type(root_opt.load_last_step) == bool else eval(root_opt.load_last_step)
    parser.run_wandb = root_opt.run_wandb
    parser.device = root_opt.device
    parser.datamode = root_opt.datamode
    parser.viton_batch_size = root_opt.viton_batch_size
    parser.save_period = root_opt.save_period
    parser.print_step = root_opt.print_step
    parser.val_count = root_opt.val_count
    parser.niter = root_opt.niter
    parser.niter_decay = root_opt.niter_decay
    return parser


def get_root_opt_experiment_dir(root_opt):
    root_opt.rail_dir = root_opt.rail_dir.format(root_opt.dataset_name, root_opt.res, root_opt.datamode)    
    root_opt.original_dir = root_opt.original_dir.format(root_opt.dataset_name, root_opt.res, root_opt.datamode)
    if root_opt.res == 'low_res':
        root_opt.original_dir = root_opt.original_dir.replace(root_opt.res, os.path.join(root_opt.res, root_opt.low_res_dataset_name))
    # Current model
    root_opt.this_viton_save_to_dir = os.path.join(root_opt.this_viton_save_to_dir, root_opt.VITON_Model)
    root_opt.this_viton_load_from_dir = root_opt.this_viton_load_from_dir.format(root_opt.VITON_Type, root_opt.VITON_Name, root_opt.this_viton_load_from_dir)
    root_opt.this_viton_load_from_dir = os.path.join(root_opt.this_viton_load_from_dir, root_opt.VITON_Model)
    
    # VGG
    root_opt.vgg_experiment_from_dir = root_opt.vgg_experiment_from_dir.format(root_opt.VITON_Type, root_opt.VITON_Name, root_opt.vgg_load_from_model)
    root_opt.vgg_experiment_from_dir = os.path.join(root_opt.vgg_experiment_from_dir, root_opt.VITON_Model)
    
    # G1
    root_opt.g1_experiment_from_dir = root_opt.g1_experiment_from_dir.format(root_opt.VITON_Type, root_opt.VITON_Name, root_opt.g1_load_from_model)
    root_opt.g1_experiment_from_dir = os.path.join(root_opt.g1_experiment_from_dir, root_opt.VITON_Model)
    
    # G2
    root_opt.g2_experiment_from_dir = root_opt.g2_experiment_from_dir.format(root_opt.VITON_Type, root_opt.VITON_Name, root_opt.g2_load_from_model)
    root_opt.g2_experiment_from_dir = os.path.join(root_opt.g2_experiment_from_dir, root_opt.VITON_Model)
    
    # G
    root_opt.g_experiment_from_dir = root_opt.g_experiment_from_dir.format(root_opt.VITON_Type, root_opt.VITON_Name, root_opt.g_load_from_model)
    root_opt.g_experiment_from_dir = os.path.join(root_opt.g_experiment_from_dir, root_opt.VITON_Model)
    
    #d 
    root_opt.d_experiment_from_dir = root_opt.d_experiment_from_dir.format(root_opt.VITON_Type, root_opt.VITON_Name, root_opt.d_load_from_model)
    root_opt.d_experiment_from_dir = os.path.join(root_opt.d_experiment_from_dir, root_opt.VITON_Model)
    root_opt.d_experiment_from_dir = fix(root_opt.d_experiment_from_dir)

    # Unet
    root_opt.unet_experiment_from_dir = root_opt.unet_experiment_from_dir.format(root_opt.VITON_Type, root_opt.VITON_Name, root_opt.unet_load_from_model)
    root_opt.unet_experiment_from_dir = os.path.join(root_opt.unet_experiment_from_dir, root_opt.VITON_Model)
    
    return root_opt

def process_opt(opt, root_opt):
    parser = opt
    parser = argparse.Namespace(**parser)
    root_opt = get_root_experiment_runs(root_opt)
    root_opt = get_root_opt_experiment_dir(root_opt)
    parser = get_root_opt_checkpoint_dir(parser, root_opt)
    parser, root_opt = get_root_opt_results_dir(parser, root_opt)    
    parser = copy_root_opt_to_opt(parser, root_opt)
    return parser, root_opt

def get_root_opt_checkpoint_dir(opt, root_opt):
    last_step = root_opt.load_last_step if type(root_opt.load_last_step) == bool else eval(root_opt.load_last_step)
    sort_digit = lambda name: int(name.split("_")[0])
    # ================================= VGG 19 =================================
    opt.vgg_save_step_checkpoint_dir = opt.vgg_save_step_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    opt.vgg_save_step_checkpoint_dir = fix(opt.vgg_save_step_checkpoint_dir)
    opt.vgg_save_step_checkpoint = os.path.join(opt.vgg_save_step_checkpoint_dir, opt.vgg_save_step_checkpoint)
    opt.vgg_save_step_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.vgg_save_step_checkpoint)
    opt.vgg_save_step_checkpoint_dir = os.path.join("/",*opt.vgg_save_step_checkpoint.split("/")[:-1])
    
    opt.vgg_save_final_checkpoint_dir = opt.vgg_save_final_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    opt.vgg_save_final_checkpoint_dir = fix(opt.vgg_save_final_checkpoint_dir)
    opt.vgg_save_final_checkpoint = os.path.join(opt.vgg_save_final_checkpoint_dir, opt.vgg_save_final_checkpoint)
    opt.vgg_save_final_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.vgg_save_final_checkpoint)
    opt.vgg_save_final_checkpoint_dir = os.path.join("/",*opt.vgg_save_final_checkpoint.split("/")[:-1])
    
    opt.vgg_load_final_checkpoint_dir = opt.vgg_load_final_checkpoint_dir.format(root_opt.vgg_experiment_from_run, root_opt.vgg_experiment_from_dir)
    opt.vgg_load_final_checkpoint_dir = fix(opt.vgg_load_final_checkpoint_dir)
    opt.vgg_load_final_checkpoint = os.path.join(opt.vgg_load_final_checkpoint_dir, opt.vgg_load_final_checkpoint)
    opt.vgg_load_final_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.vgg_load_final_checkpoint)
    opt.vgg_load_final_checkpoint_dir = os.path.join("/",*opt.vgg_load_final_checkpoint.split("/")[:-1])
   # ================================= G1 ================================= 
    opt.g1_save_step_checkpoint_dir = opt.g1_save_step_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    opt.g1_save_step_checkpoint = os.path.join(opt.g1_save_step_checkpoint_dir, opt.g1_save_step_checkpoint)
    opt.g1_save_step_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.g1_save_step_checkpoint)
    opt.g1_save_step_checkpoint_dir = os.path.join("/",*opt.g1_save_step_checkpoint.split("/")[:-1])
    
    opt.g1_save_final_checkpoint_dir = opt.g1_save_final_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    opt.g1_save_final_checkpoint_dir = fix(opt.g1_save_final_checkpoint_dir)
    opt.g1_save_final_checkpoint = os.path.join(opt.g1_save_final_checkpoint_dir, opt.g1_save_final_checkpoint)
    opt.g1_save_final_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.g1_save_final_checkpoint)
    opt.g1_save_final_checkpoint_dir = os.path.join("/",*opt.g1_save_final_checkpoint.split("/")[:-1])
    
    
    opt.g1_load_final_checkpoint_dir = opt.g1_load_final_checkpoint_dir.format(root_opt.g1_experiment_from_run, root_opt.g1_experiment_from_dir)
    opt.g1_load_final_checkpoint_dir = fix(opt.g1_load_final_checkpoint_dir)
    opt.g1_load_final_checkpoint = os.path.join(opt.g1_load_final_checkpoint_dir, opt.g1_load_final_checkpoint)
    opt.g1_load_final_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.g1_load_final_checkpoint)
    opt.g1_load_final_checkpoint_dir = os.path.join("/",*opt.g1_load_final_checkpoint.split("/")[:-1])
    if not last_step:
        opt.g1_load_step_checkpoint_dir = opt.g1_load_step_checkpoint_dir.format(root_opt.g1_experiment_from_run, root_opt.g1_experiment_from_dir)
    else:
        opt.g1_load_step_checkpoint_dir = opt.g1_load_step_checkpoint_dir.format(root_opt.g1_experiment_from_run, root_opt.this_viton_save_to_dir)
    opt.g1_load_step_checkpoint_dir = fix(opt.g1_load_step_checkpoint_dir)
    if not last_step:
        opt.g1_load_step_checkpoint = os.path.join(opt.g1_load_step_checkpoint_dir, opt.g1_load_step_checkpoint)
    else:
        os_list = os.listdir(opt.g1_load_step_checkpoint_dir.format(root_opt.g1_experiment_from_run, root_opt.this_viton_save_to_dir))
        os_list = [string for string in os_list if "G1" in string]
        last_step = sorted(os_list, key=sort_digit)[-1]
        opt.g1_load_step_checkpoint = os.path.join(opt.g1_load_step_checkpoint_dir, last_step)
    opt.g1_load_step_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.g1_load_step_checkpoint)
    opt.g1_load_step_checkpoint_dir = os.path.join("/",*opt.g1_load_step_checkpoint.split("/")[:-1])
   # ================================= G2 ================================= 
    opt.g2_save_step_checkpoint_dir = opt.g2_save_step_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    opt.g2_save_step_checkpoint_dir = fix(opt.g2_save_step_checkpoint_dir)
    opt.g2_save_step_checkpoint = os.path.join(opt.g2_save_step_checkpoint_dir, opt.g2_save_step_checkpoint)
    opt.g2_save_step_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.g2_save_step_checkpoint)
    opt.g2_save_step_checkpoint_dir = os.path.join("/",*opt.g2_save_step_checkpoint.split("/")[:-1])
    
    opt.g2_save_final_checkpoint_dir = opt.g2_save_final_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    opt.g2_save_final_checkpoint_dir = fix(opt.g2_save_final_checkpoint_dir)
    opt.g2_save_final_checkpoint = os.path.join(opt.g2_save_final_checkpoint_dir, opt.g2_save_final_checkpoint)
    opt.g2_save_final_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.g2_save_final_checkpoint)
    opt.g2_save_final_checkpoint_dir = os.path.join("/",*opt.g2_save_final_checkpoint.split("/")[:-1])
    
    opt.g2_load_final_checkpoint_dir = opt.g2_load_final_checkpoint_dir.format(root_opt.g2_experiment_from_run, root_opt.g2_experiment_from_dir)
    opt.g2_load_final_checkpoint_dir = fix(opt.g2_load_final_checkpoint_dir)
    opt.g2_load_final_checkpoint = os.path.join(opt.g2_load_final_checkpoint_dir, opt.g2_load_final_checkpoint)
    opt.g2_load_final_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.g2_load_final_checkpoint)
    opt.g2_load_final_checkpoint_dir = os.path.join("/",*opt.g2_load_final_checkpoint.split("/")[:-1])
    if not last_step:
        opt.g2_load_step_checkpoint_dir = opt.g2_load_step_checkpoint_dir.format(root_opt.g2_experiment_from_run, root_opt.g2_experiment_from_dir)
    else:
        opt.g2_load_step_checkpoint_dir = opt.g2_load_step_checkpoint_dir.format(root_opt.g2_experiment_from_run, root_opt.this_viton_save_to_dir)
    opt.g2_load_step_checkpoint_dir = fix(opt.g2_load_step_checkpoint_dir)
    if not last_step:
        opt.g2_load_step_checkpoint = os.path.join(opt.g2_load_step_checkpoint_dir, opt.g2_load_step_checkpoint)
    else:
        os_list = os.listdir(opt.g2_load_step_checkpoint_dir.format(root_opt.g2_experiment_from_run, root_opt.this_viton_save_to_dir))
        os_list = [string for string in os_list if "G2" in string]
        last_step = sorted(os_list, key=sort_digit)[-1]
        opt.g2_load_step_checkpoint = os.path.join(opt.g2_load_step_checkpoint_dir, last_step)
    opt.g2_load_step_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.g2_load_step_checkpoint)
    opt.g2_load_step_checkpoint_dir = os.path.join("/",*opt.g2_load_step_checkpoint.split("/")[:-1])
   # ================================= G ================================= 
    opt.g_save_step_checkpoint_dir = opt.g_save_step_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    opt.g_save_step_checkpoint_dir = fix(opt.g_save_step_checkpoint_dir)
    opt.g_save_step_checkpoint = os.path.join(opt.g_save_step_checkpoint_dir, opt.g_save_step_checkpoint)
    opt.g_save_step_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.g_save_step_checkpoint)
    opt.g_save_step_checkpoint_dir = os.path.join("/",*opt.g_save_step_checkpoint.split("/")[:-1])
    
    opt.g_save_final_checkpoint_dir = opt.g_save_final_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    opt.g_save_final_checkpoint_dir = fix(opt.g_save_final_checkpoint_dir)
    opt.g_save_final_checkpoint = os.path.join(opt.g_save_final_checkpoint_dir, opt.g_save_final_checkpoint)
    opt.g_save_final_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.g_save_final_checkpoint)
    opt.g_save_final_checkpoint_dir = os.path.join("/",*opt.g_save_final_checkpoint.split("/")[:-1])
    
    opt.g_load_final_checkpoint_dir = opt.g_load_final_checkpoint_dir.format(root_opt.g_experiment_from_run, root_opt.g_experiment_from_dir)
    opt.g_load_final_checkpoint_dir = fix(opt.g_load_final_checkpoint_dir)
    opt.g_load_final_checkpoint = os.path.join(opt.g_load_final_checkpoint_dir, opt.g_load_final_checkpoint)
    opt.g_load_final_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.g_load_final_checkpoint)
    opt.g_load_final_checkpoint_dir = os.path.join("/",*opt.g_load_final_checkpoint.split("/")[:-1])
    if not last_step:
        opt.g_load_step_checkpoint_dir = opt.g_load_step_checkpoint_dir.format(root_opt.g_experiment_from_run, root_opt.g_experiment_from_dir)
    else:
        opt.g_load_step_checkpoint_dir = opt.g_load_step_checkpoint_dir.format(root_opt.g_experiment_from_run, root_opt.this_viton_save_to_dir)
    opt.g_load_step_checkpoint_dir = fix(opt.g_load_step_checkpoint_dir)
    if not last_step:
        opt.g_load_step_checkpoint = os.path.join(opt.g_load_step_checkpoint_dir, opt.g_load_step_checkpoint)
    else:
        os_list = os.listdir(opt.g_load_step_checkpoint_dir.format(root_opt.g_experiment_from_run, root_opt.this_viton_save_to_dir))
        os_list = [string for string in os_list if "G.pth" in string]
        last_step = sorted(os_list, key=sort_digit)[-1]
        opt.g_load_step_checkpoint = os.path.join(opt.g_load_step_checkpoint_dir, last_step)
    opt.g_load_step_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.g_load_step_checkpoint)    
    opt.g_load_step_checkpoint_dir = os.path.join("/",*opt.g_load_step_checkpoint.split("/")[:-1])
   # ================================= d ================================= 
    opt.d_save_step_checkpoint_dir = opt.d_save_step_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    opt.d_save_step_checkpoint_dir = fix(opt.d_save_step_checkpoint_dir)
    opt.d_save_step_checkpoint = os.path.join(opt.d_save_step_checkpoint_dir, opt.d_save_step_checkpoint)
    opt.d_save_step_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.d_save_step_checkpoint)    
    opt.d_save_step_checkpoint_dir = os.path.join("/",*opt.d_save_step_checkpoint.split("/")[:-1])
    
    opt.d_save_final_checkpoint_dir = opt.d_save_final_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    opt.d_save_final_checkpoint_dir = fix(opt.d_save_final_checkpoint_dir)
    opt.d_save_final_checkpoint = os.path.join(opt.d_save_final_checkpoint_dir, opt.d_save_final_checkpoint)
    opt.d_save_final_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.d_save_final_checkpoint)    
    opt.d_save_final_checkpoint_dir = os.path.join("/",*opt.d_save_final_checkpoint.split("/")[:-1])
    
    opt.d_load_final_checkpoint_dir = opt.d_load_final_checkpoint_dir.format(root_opt.d_experiment_from_run, root_opt.d_experiment_from_dir)
    opt.d_load_final_checkpoint_dir = fix(opt.d_load_final_checkpoint_dir)
    opt.d_load_final_checkpoint = os.path.join(opt.d_load_final_checkpoint_dir, opt.d_load_final_checkpoint)
    opt.d_load_final_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.d_load_final_checkpoint)    
    opt.d_load_final_checkpoint_dir = os.path.join("/",*opt.d_load_final_checkpoint.split("/")[:-1])
    if not last_step:
        opt.d_load_step_checkpoint_dir = opt.d_load_step_checkpoint_dir.format(root_opt.d_experiment_from_run, root_opt.d_experiment_from_dir)
    else:
        opt.d_load_step_checkpoint_dir = opt.d_load_step_checkpoint_dir.format(root_opt.d_experiment_from_run, root_opt.this_viton_save_to_dir)
    opt.d_load_step_checkpoint_dir = fix(opt.d_load_step_checkpoint_dir)
    if not last_step:
        opt.d_load_step_checkpoint = os.path.join(opt.d_load_step_checkpoint_dir, opt.d_load_step_checkpoint)
    else:
        os_list = os.listdir(opt.d_load_step_checkpoint_dir.format(root_opt.d_experiment_from_run, root_opt.this_viton_save_to_dir))
        os_list = [string for string in os_list if "D.pth" in string]
        last_step = sorted(os_list, key=sort_digit)[-1]
        opt.d_load_step_checkpoint = os.path.join(opt.d_load_step_checkpoint_dir, last_step)
    opt.d_load_step_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.d_load_step_checkpoint) 
    opt.d_load_step_checkpoint_dir = os.path.join("/",*opt.d_load_step_checkpoint.split("/")[:-1])
   # ================================= Unet ================================= 
   
    opt.unet_save_step_checkpoint_dir = opt.unet_save_step_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    opt.unet_save_step_checkpoint_dir = fix(opt.unet_save_step_checkpoint_dir)
    opt.unet_save_step_checkpoint = os.path.join(opt.unet_save_step_checkpoint_dir, opt.unet_save_step_checkpoint)
    opt.unet_save_step_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.unet_save_step_checkpoint) 
    opt.unet_save_step_checkpoint_dir = os.path.join("/",*opt.unet_save_step_checkpoint.split("/")[:-1])
    
    opt.unet_save_final_checkpoint_dir = opt.unet_save_final_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    opt.unet_save_final_checkpoint_dir = fix(opt.unet_save_final_checkpoint_dir)
    opt.unet_save_final_checkpoint = os.path.join(opt.unet_save_final_checkpoint_dir, opt.unet_save_final_checkpoint)
    opt.unet_save_final_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.unet_save_final_checkpoint)
    opt.unet_save_final_checkpoint_dir = os.path.join("/",*opt.unet_save_final_checkpoint.split("/")[:-1])
    
    opt.unet_load_final_checkpoint_dir = opt.unet_load_final_checkpoint_dir.format(root_opt.unet_experiment_from_run, root_opt.unet_experiment_from_dir)
    opt.unet_load_final_checkpoint = os.path.join(opt.unet_load_final_checkpoint_dir, opt.unet_load_final_checkpoint)
    opt.unet_load_final_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.unet_load_final_checkpoint)
    opt.unet_load_final_checkpoint_dir = os.path.join("/",*opt.unet_load_final_checkpoint.split("/")[:-1])
    if not last_step:
        opt.unet_load_step_checkpoint_dir = opt.unet_load_step_checkpoint_dir.format(root_opt.unet_experiment_from_run, root_opt.unet_experiment_from_dir)
    else:
        opt.unet_load_step_checkpoint_dir = opt.unet_load_step_checkpoint_dir.format(root_opt.unet_experiment_from_run, root_opt.this_viton_save_to_dir)
    opt.unet_load_step_checkpoint_dir = fix(opt.unet_load_step_checkpoint_dir)
    if not last_step:
        opt.unet_load_step_checkpoint = os.path.join(opt.unet_load_step_checkpoint_dir, opt.unet_load_step_checkpoint)
    else:
        os_list = os.listdir(opt.unet_load_step_checkpoint_dir.format(root_opt.unet_experiment_from_run, root_opt.this_viton_save_to_dir))
        os_list = [string for string in os_list if "U.pth" in string]
        last_step = sorted(os_list, key=sort_digit)[-1]
        opt.unet_load_step_checkpoint = os.path.join(opt.unet_load_step_checkpoint_dir, last_step)
    opt.unet_load_step_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.unet_load_step_checkpoint)    
    opt.unet_load_step_checkpoint_dir = os.path.join("/",*opt.unet_load_step_checkpoint.split("/")[:-1])
    if opt.datamode == 'train':
        opt.isTrain = True
    else:
        opt.isTrain = False
    return opt



def get_opt(root_opt):
    
    # Load the YAML configuration file
    with open(os.path.join(root_opt.experiment_run_yaml,'tps_grid_gen.yml'), 'r') as config_file:
        config = yaml.safe_load(config_file)

    return config

def print_log(log_path, content, to_print=True):
    import os
    if os.path.exists(log_path):
        with open(log_path, 'a') as f:
            f.write(content)
            f.write('\n')

        if to_print:
            print(content)

def get_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cuda:0') if ckpt_path else None
    return ckpt

def load_ckpt(model, ckpt=None):
    if ckpt is not None:
        ckpt_new = model.state_dict()
        pretrained = ckpt.get('model') if ckpt.get('model') else ckpt
        for param in ckpt_new:
            ckpt_new[param] = pretrained[param]
        model.load_state_dict(ckpt_new)

def split_dataset(dataset,train_size=0.8):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    train_indices, validation_indices = train_test_split(indices, train_size=train_size)
    train_subset = Subset(dataset, train_indices)
    validation_subset = Subset(dataset, validation_indices)
    return train_subset, validation_subset

def train_acgpn_(opt_, root_opt_, run_wandb=False,sweep=None):
    global opt, root_opt, wandb,sweep_id
    opt,root_opt = process_opt(opt_, root_opt_)
    sweep_id = None
    if sweep is not None:
        import wandb
        wandb.login()
        sweep_id = wandb.sweep(sweep=sweep, project="Fashion-NeRF-Sweep")
        wandb.agent(sweep_id,__train_acgpn_sweep,count=5)
    elif run_wandb:
        import wandb
        wandb.login()
        wandb.init(project="Fashion-NeRF",entity='prime_lab',notes=f"question: {opt.question}, intent: {opt.intent}",tags=[f"{root_opt.experiment_run}"],config=vars(opt))
        temp_opt = vars(opt)
        temp_opt['wandb_name'] = wandb.run.name
        opt = argparse.Namespace(**temp_opt)
        _train_acgpn_()
    else:
        wandb = None
        _train_acgpn_()
        
def __train_acgpn_sweep():
    if wandb is not None:
        with wandb.init(project="Fashion-NeRF-Sweep",entity='prime_lab',notes=f"question: {opt.question}, intent: {opt.intent}",tags=[f"{root_opt.experiment_run}"],config=vars(opt)):
            _train_acgpn_()        
            
            
def _train_acgpn_():
    global opt, root_opt, wandb,sweep_id
    if sweep_id is not None:
        opt = wandb.config
    experiment_string = f"{root_opt.experiment_run.replace('/','_')}_{root_opt.opt_vton_yaml.replace('yaml/','')}"
    with open(os.path.join(root_opt.experiment_run_yaml, experiment_string), 'w') as outfile:
        yaml.dump(vars(opt), outfile, default_flow_style=False)
    # start_epoch, epoch_iter = 1, 0 # For in case of load shedding
    writer = SummaryWriter(opt.tensorboard_dir) 
    data_loader = CreateDataLoader(opt, root_opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    train_dataset = dataset.dataset
    train_dataset, validation_dataset = split_dataset(train_dataset)
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))
    validation_loader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    
    # Directories
    log_path = os.path.join(opt.results_dir, 'log.txt')
    if not os.path.exists(opt.results_dir):
        os.makedirs(opt.results_dir)
        with open(log_path, 'w') as file:
            file.write(f"Hello, this is experiment {root_opt.experiment_run} \n")
    
    last_step = root_opt.load_last_step if type(root_opt.load_last_step) == bool else eval(root_opt.load_last_step)
    if last_step:
        load_checkpoint(model, opt.g_load_step_checkpoint)
        print_log(log_path, f'Load pretrained model from {opt.g_load_step_checkpoint}')
    elif os.path.exists(opt.g_load_final_checkpoint):
        load_checkpoint(model, opt.g_load_final_checkpoint)
        print_log(log_path, f'Load pretrained model from {opt.g_load_final_checkpoint}')

    # total_steps = (start_epoch-1) * dataset_size + epoch_iter
    for epoch in tqdm(range(0, opt.niter + opt.niter_decay + 1)):
        train(epoch, train_loader, model, writer, wandb=wandb)
        if epoch % opt.val_count == 0:
            validate(epoch, validation_loader, model, writer, wandb=wandb)
        # break
    model.module.save()


def train(step, dataloader, model, writer, wandb=None):
    model.train()
    iter_path = os.path.join(opt.g1_save_final_checkpoint_dir, 'iter.txt')
    dataset_size = len(dataloader)
    total_loss_G = 0
    total_steps =0
    epoch_iter = 0
    for i, data in enumerate(dataloader):   
        epoch_iter += opt.batchSize
        t_mask = torch.FloatTensor((data['label'].cpu().numpy() == 7).astype(float))    
        data['label']=data['label']*(1-t_mask)+t_mask*4
        mask_clothes=torch.FloatTensor((data['label'].cpu().numpy()==4).astype(int))
        mask_fore=torch.FloatTensor((data['label'].cpu().numpy()>0).astype(int))
        img_fore=data['image']*mask_fore
        img_fore_wc=img_fore*mask_fore
        all_clothes_label=changearm(data)
        ############## Forward Pass ######################
        losses, fake_image, real_image,input_label,L1_loss,style_loss,clothes_mask,warped,refined,CE_loss,rx,ry,cx,cy,rg,cg= model(Variable(data['label'].cuda()),Variable(data['edge'].cuda()),Variable(img_fore.cuda()),Variable(mask_clothes.cuda()),Variable(data['color'].cuda()),Variable(all_clothes_label.cuda()),Variable(data['image'].cuda()),Variable(data['pose'].cuda()),Variable(data['mask'].cuda())  )

        # sum per device losses
        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
        loss_dict = dict(zip(model.module.loss_names, losses))

        # calculate final loss scalar
        # L2 (Cross-Entropy), L1 (CGAN)
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        loss_G = loss_dict['G_GAN']+loss_dict.get('G_GAN_Feat',0)+loss_dict.get('G_VGG',0)+torch.mean(L1_loss+CE_loss+rx+ry+cx+cy+rg+cg)
        total_loss_G += loss_G.item()
        ############### Backward Pass ####################
        # update generator weights
        model.module.optimizer_G.zero_grad()
        model.module.optimizer_G_remain.zero_grad()
        loss_G.backward()
        model.module.optimizer_G.step()
        model.module.optimizer_G_remain.step()

        model.module.optimizer_D_remain.zero_grad()
        model.module.optimizer_D.zero_grad()
        loss_D.backward()
        model.module.optimizer_D_remain.step()
        model.module.optimizer_D.step()

        ############## Display results and errors ##########
        
        ### display output images
        if (step + 1) % opt.display_count == 0:
            a = generate_label_color(generate_label_plain(input_label), opt).float().cuda()
            b = real_image.float().cuda()
            c = fake_image.float().cuda()
            d=torch.cat([clothes_mask,clothes_mask,clothes_mask],1)
            e=warped
            f=refined
            combine = torch.cat([a[0],b[0],c[0],d[0],e[0],f[0]], 2).squeeze()
            cv_img=(combine.permute(1,2,0).detach().cpu().numpy()+1)/2
            writer.add_scalar('loss_d', loss_D, step)
            writer.add_scalar('warping_loss', loss_G, step)
            writer.add_scalar('l1_cloth', torch.mean(L1_loss), step)
            writer.add_scalar('cross_entropy', torch.mean(CE_loss), step)
            writer.add_scalar('rx', torch.mean(rx), step)
            writer.add_scalar('ry', torch.mean(ry), step)
            writer.add_scalar('cx', torch.mean(cx), step)
            writer.add_scalar('cy', torch.mean(cy), step)

            writer.add_scalar('loss_g_gan', loss_dict['G_GAN'], step)
            writer.add_scalar('loss_g_gan_feat', loss_dict['G_GAN_Feat'], step)
            writer.add_scalar('loss_g_vgg', loss_dict['G_VGG'], step)
            
            writer.add_image('combine', (combine.data + 1) / 2.0, step)
            if wandb is not None:
                my_table = wandb.Table(columns=['Combined Image','Real Image','Fake Image','Clothes Mask','Refined Image','Label','Warped Clothing'])
                real_image_wandb = get_wandb_image(real_image[0], wandb=wandb)
                fake_image_wandb = get_wandb_image(fake_image[0], wandb=wandb)
                clothes_mask_wandb = get_wandb_image(d[0], wandb=wandb)
                refined_wandb = get_wandb_image(refined[0], wandb=wandb)
                labeled_wandb = get_wandb_image(a[0], wandb=wandb)
                warped_wandb = get_wandb_image(warped[0], wandb=wandb)
                my_table.add_data(wandb.Image((cv_img*255).astype(np.uint8)), real_image_wandb,fake_image_wandb,clothes_mask_wandb,refined_wandb, labeled_wandb, warped_wandb)
                wandb.log({'loss_d': loss_D,
                'warping_loss': loss_G,
                'l1_cloth': torch.mean(L1_loss),
                'cross_entropy_loss': torch.mean(CE_loss),
                'rx': torch.mean(rx),
                'ry': torch.mean(ry),
                'cx': torch.mean(cx),
                'cy': torch.mean(cy),
                'loss_g_gan': loss_dict['G_GAN'],
                'loss_g_gan_feat': loss_dict['G_GAN_Feat'],
                'loss_g_vgg': loss_dict['G_VGG'],
                'Table':my_table })
                
            rgb=(cv_img*255).astype(np.uint8)
            bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(opt.results_dir, 'train_' + str(step) +'.jpg'),bgr)
        total_steps += 1
        #print('{}:{}:[step-{}]--[loss_G-{:.6f}]--[loss_D-{:.6f}]--[ETA-{}]-[rx{}]-[ry{}]-[cx{}]-[cy{}]-[rg{}]-[cg{}]'.format(now,epoch_iter,step, loss_G, loss_D, eta,rx,ry,cx,cy,rg,cg))
        print('[total steps-{}]--[step-{}]--[loss_G-{:.6f}]--[loss_D-{:.6f}]]'.format(total_steps, step, loss_G,loss_D))
        if step >= dataset_size:
            break
        
    # end of epoch 
    ### save model for this epoch
    if step % opt.save_period == 0:
        print('saving the model at the end of epoch %d, iters %d' % (step, total_steps))        
        model.module.save_step(step)
        np.savetxt(iter_path, (step + 1, 0), delimiter=',', fmt='%d')

    ### instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (step == opt.niter_fix_global):
        model.module.update_fixed_params()

    ### linearly decay learning rate after certain iterations
    if step > opt.niter:
        model.module.update_learning_rate()
        
    avg_total_loss_G = total_loss_G / len(dataloader)
    if wandb is not None:
        wandb.log({"total_avg_warping_loss":avg_total_loss_G})
    
    
    
# validate(validation_loader, model, writer, wandb=wandb)
def validate(step, dataloader, model, writer, wandb=None):
    model.eval()
    total_loss_G = 0
    total_steps = 0
    epoch_iter = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            t_mask = torch.FloatTensor((data['label'].cpu().numpy() == 7).astype(float))    
            data['label']=data['label']*(1-t_mask)+t_mask*4
            mask_clothes=torch.FloatTensor((data['label'].cpu().numpy()==4).astype(int))
            mask_fore=torch.FloatTensor((data['label'].cpu().numpy()>0).astype(int))
            img_fore=data['image']*mask_fore
            all_clothes_label=changearm(data)
            ############## Forward Pass ######################
            losses, fake_image, real_image,input_label,L1_loss,style_loss,clothes_mask,warped,refined,CE_loss,rx,ry,cx,cy,rg,cg= model(Variable(data['label'].cuda()),Variable(data['edge'].cuda()),Variable(img_fore.cuda()),Variable(mask_clothes.cuda()),Variable(data['color'].cuda()),Variable(all_clothes_label.cuda()),Variable(data['image'].cuda()),Variable(data['pose'].cuda()),Variable(data['mask'].cuda())  )

            # sum per device losses
            losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
            loss_dict = dict(zip(model.module.loss_names, losses))

            # calculate final loss scalar
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            loss_G = loss_dict['G_GAN']+loss_dict.get('G_GAN_Feat',0)+loss_dict.get('G_VGG',0)+torch.mean(L1_loss+CE_loss+rx+ry+cx+cy+rg+cg)
            total_loss_G += loss_G.item()
        
            a = generate_label_color(generate_label_plain(input_label), opt).float().cuda()
            b = real_image.float().cuda()
            c = fake_image.float().cuda()
            d=torch.cat([clothes_mask,clothes_mask,clothes_mask],1)
            e=warped
            f=refined
            combine = torch.cat([a[0],b[0],c[0],d[0],e[0],f[0]], 2).squeeze()
            cv_img=(combine.permute(1,2,0).detach().cpu().numpy()+1)/2
            writer.add_scalar('val_loss_d', loss_D, step)
            writer.add_scalar('val_warping_loss', loss_G, step)
            writer.add_scalar('val_l1_cloth', torch.mean(L1_loss), step)
            writer.add_scalar('val_cross_entropy_loss', torch.mean(CE_loss), step)
            writer.add_scalar('val_rx', torch.mean(rx), step)
            writer.add_scalar('val_ry', torch.mean(ry), step)
            writer.add_scalar('val_cx', torch.mean(cx), step)
            writer.add_scalar('val_cy', torch.mean(cy), step)

            writer.add_scalar('val_loss_g_gan', loss_dict['G_GAN'], step)
            writer.add_scalar('val_loss_g_gan_feat', loss_dict['G_GAN_Feat'], step)
            writer.add_scalar('val_loss_g_vgg', loss_dict['G_VGG'], step)
            
            writer.add_image('combine', (combine.data + 1) / 2.0, step)
            if wandb is not None:
                my_table = wandb.Table(columns=['Combined Image','Real Image','Fake Image','Clothes Mask','Refined Image','Label','Warped Clothing'])
                real_image_wandb = get_wandb_image(real_image[0], wandb=wandb)
                fake_image_wandb = get_wandb_image(fake_image[0], wandb=wandb)
                clothes_mask_wandb = get_wandb_image(d[0], wandb=wandb)
                refined_wandb = get_wandb_image(refined[0], wandb=wandb)
                labeled_wandb = get_wandb_image(a[0], wandb=wandb)
                warped_wandb = get_wandb_image(warped[0], wandb=wandb)
                my_table.add_data(wandb.Image((cv_img*255).astype(np.uint8)), real_image_wandb,fake_image_wandb,clothes_mask_wandb,refined_wandb, labeled_wandb, warped_wandb)
                wandb.log({'val_loss_d': loss_D,
                'val_warping_loss': loss_G,
                'val_l1_cloth': torch.mean(L1_loss),
                'val_cross_entropy_loss': torch.mean(CE_loss),
                'val_rx': torch.mean(rx),
                'val_ry': torch.mean(ry),
                'val_cx': torch.mean(cx),
                'val_cy': torch.mean(cy),
                'val_loss_g_gan': loss_dict['G_GAN'],
                'val_loss_g_gan_feat': loss_dict['G_GAN_Feat'],
                'val_loss_g_vgg': loss_dict['G_VGG'],
                'Val_Table':my_table })
            rgb=(cv_img*255).astype(np.uint8)
            bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(opt.results_dir, 'valid_' + str(step) +'.jpg'),bgr)
            step += 1
    avg_total_loss_G = total_loss_G / len(dataloader)
    if wandb is not None:
        wandb.log({"val_total_avg_warping_loss":avg_total_loss_G})
    
