import torch
import torch.nn as nn
from VITON.Parser_Based.HR_VITON.sync_batchnorm import DataParallelWithCallback
from torchvision.utils import make_grid as make_image_grid
from torchvision.utils import save_image
import argparse
import os
import time
from dataset import FashionDataLoader, FashionNeRFDataset
from VITON.Parser_Based.HR_VITON.networks import ConditionGenerator, load_checkpoint, make_grid
from VITON.Parser_Based.HR_VITON.network_generator import SPADEGenerator
from VITON.Parser_Based.HR_VITON.utils import *

import torchgeometry as tgm
from collections import OrderedDict

fix = lambda path: os.path.normpath(path)

def get_root_experiment_runs(root_opt):
    root_opt.experiment_run = root_opt.experiment_run.format(root_opt.experiment_number, root_opt.run_number)
    root_opt.experiment_from_run = root_opt.experiment_from_run.format(root_opt.experiment_from_number, root_opt.run_from_number)
    
    root_opt.tocg_experiment_from_run = root_opt.tocg_experiment_from_run.format(root_opt.tocg_experiment_from_number, root_opt.tocg_run_from_number)
    root_opt.tocg_discriminator_experiment_from_run = root_opt.tocg_discriminator_experiment_from_run.format(root_opt.tocg_discriminator_experiment_from_number, root_opt.tocg_discriminator_run_from_number)
    
    root_opt.gen_experiment_from_run = root_opt.gen_experiment_from_run.format(root_opt.gen_experiment_from_number, root_opt.gen_run_from_number)
    root_opt.gen_discriminator_experiment_from_run = root_opt.gen_discriminator_experiment_from_run.format(root_opt.gen_discriminator_experiment_from_number, root_opt.gen_discriminator_run_from_number)
    return root_opt

def get_root_opt_experiment_dir(root_opt):
    root_opt.rail_dir = root_opt.rail_dir.format(root_opt.dataset_name, root_opt.res, root_opt.datamode)    
    root_opt.original_dir = root_opt.original_dir.format(root_opt.dataset_name, root_opt.res, root_opt.datamode)
    if root_opt.res == 'low_res':
        root_opt.original_dir = root_opt.original_dir.replace(root_opt.res, os.path.join(root_opt.res, root_opt.low_res_dataset_name))
    # Current model
    root_opt.this_viton_save_to_dir = os.path.join(root_opt.this_viton_save_to_dir, root_opt.VITON_Model)
    root_opt.this_viton_load_from_dir = root_opt.this_viton_load_from_dir.format(root_opt.VITON_Type, root_opt.VITON_Name, root_opt.this_viton_load_from_dir)
    root_opt.this_viton_load_from_dir = os.path.join(root_opt.this_viton_load_from_dir, root_opt.VITON_Model)
    
    # tocg
    root_opt.tocg_experiment_from_dir = root_opt.tocg_experiment_from_dir.format(root_opt.VITON_Type, root_opt.VITON_Name, root_opt.tocg_load_from_model)
    root_opt.tocg_experiment_from_dir = os.path.join(root_opt.tocg_experiment_from_dir, 'TOCG')
    
    # tocg discriminator
    root_opt.tocg_discriminator_experiment_from_dir = root_opt.tocg_discriminator_experiment_from_dir.format(root_opt.VITON_Type, root_opt.VITON_Name, root_opt.tocg_discriminator_load_from_model)
    root_opt.tocg_discriminator_experiment_from_dir = os.path.join(root_opt.tocg_discriminator_experiment_from_dir, 'TOCG')    
    
    
    # gen
    root_opt.gen_experiment_from_dir = root_opt.gen_experiment_from_dir.format(root_opt.VITON_Type, root_opt.VITON_Name, root_opt.gen_load_from_model)
    root_opt.gen_experiment_from_dir = os.path.join(root_opt.gen_experiment_from_dir, root_opt.VITON_Model)
    
    # gen discriminator
    root_opt.gen_discriminator_experiment_from_dir = root_opt.gen_discriminator_experiment_from_dir.format(root_opt.VITON_Type, root_opt.VITON_Name, root_opt.gen_discriminator_load_from_model)
    root_opt.gen_discriminator_experiment_from_dir = os.path.join(root_opt.gen_discriminator_experiment_from_dir, root_opt.VITON_Model)    
    
    return root_opt


def get_root_opt_results_dir(parser, root_opt):
    root_opt.transforms_dir = root_opt.transforms_dir.format(root_opt.dataset_name)
    parser.tensorboard_dir = parser.tensorboard_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    parser.results_dir = parser.results_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    return parser, root_opt

def copy_root_opt_to_opt(parser, root_opt):
    parser.display_count = root_opt.display_count
    parser.cuda = root_opt.cuda
    parser.device = int(root_opt.device)
    parser.dataset_name = root_opt.dataset_name
    parser.warp_load_from_model = root_opt.warp_load_from_model
    parser.load_last_step = root_opt.load_last_step if type(root_opt.load_last_step) == bool else eval(root_opt.load_last_step)
    parser.run_wandb = root_opt.run_wandb
    parser.viton_batch_size = root_opt.viton_batch_size
    parser.save_period = root_opt.save_period
    parser.print_step = root_opt.print_step
    parser.niter = root_opt.niter
    parser.niter_decay = root_opt.niter_decay
    parser.VITON_Type = root_opt.VITON_Type
    parser.VITON_selection_dir = parser.VITON_selection_dir.format(parser.VITON_Type, parser.VITON_Name)
    return parser

def get_root_opt_checkpoint_dir(opt, root_opt):
    last_step = root_opt.load_last_step if type(root_opt.load_last_step) == bool else eval(root_opt.load_last_step)
    sort_digit = lambda name: int(name.split('_')[-1].split('.')[0])
    # ================================= tocg =================================
    opt.tocg_save_step_checkpoint_dir = opt.tocg_save_step_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    opt.tocg_save_step_checkpoint_dir = fix(opt.tocg_save_step_checkpoint_dir)
    opt.tocg_save_step_checkpoint = os.path.join(opt.tocg_save_step_checkpoint_dir, opt.tocg_save_step_checkpoint)
    opt.tocg_save_step_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.tocg_save_step_checkpoint)
    opt.tocg_save_step_checkpoint_dir = os.path.join("/",*opt.tocg_save_step_checkpoint.split("/")[:-1])
    
    opt.tocg_save_final_checkpoint_dir = opt.tocg_save_final_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    opt.tocg_save_final_checkpoint_dir = fix(opt.tocg_save_final_checkpoint_dir)
    opt.tocg_save_final_checkpoint = os.path.join(opt.tocg_save_final_checkpoint_dir, opt.tocg_save_final_checkpoint)
    opt.tocg_save_final_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.tocg_save_final_checkpoint)
    opt.tocg_save_final_checkpoint_dir = os.path.join("/",*opt.tocg_save_final_checkpoint.split("/")[:-1])
    
    opt.tocg_load_final_checkpoint_dir = opt.tocg_load_final_checkpoint_dir.format(root_opt.tocg_experiment_from_run, root_opt.tocg_experiment_from_dir)
    opt.tocg_load_final_checkpoint_dir = fix(opt.tocg_load_final_checkpoint_dir)
    opt.tocg_load_final_checkpoint = os.path.join(opt.tocg_load_final_checkpoint_dir, opt.tocg_load_final_checkpoint)
    opt.tocg_load_final_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.tocg_load_final_checkpoint)
    opt.tocg_load_final_checkpoint_dir = os.path.join("/",*opt.tocg_load_final_checkpoint.split("/")[:-1])
    
    if not last_step:
        opt.tocg_load_step_checkpoint_dir = opt.tocg_load_step_checkpoint_dir.format(root_opt.tocg_experiment_from_run, root_opt.tocg_experiment_from_dir)
    else:
        opt.tocg_load_step_checkpoint_dir = opt.tocg_load_step_checkpoint_dir.format(root_opt.tocg_experiment_from_run, root_opt.this_viton_save_to_dir)
    opt.tocg_load_step_checkpoint_dir = fix(opt.tocg_load_step_checkpoint_dir)
    if not last_step:
        opt.tocg_load_step_checkpoint = os.path.join(opt.tocg_load_step_checkpoint_dir, opt.tocg_load_step_checkpoint)
    else:
        if os.path.isdir(opt.tocg_load_step_checkpoint_dir.format(root_opt.tocg_experiment_from_run, root_opt.this_viton_save_to_dir)):
            os_list = os.listdir(opt.tocg_load_step_checkpoint_dir.format(root_opt.tocg_experiment_from_run, root_opt.this_viton_save_to_dir))
            os_list = [string for string in os_list if "tocg" in string]
            last_step = sorted(os_list, key=sort_digit)[-1]
            opt.tocg_load_step_checkpoint = os.path.join(opt.tocg_load_step_checkpoint_dir, last_step)
    opt.tocg_load_step_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.tocg_load_step_checkpoint)
    opt.tocg_load_step_checkpoint_dir = os.path.join("/",*opt.tocg_load_step_checkpoint.split("/")[:-1])
    # ================================= tocg DISCRIMINATOR =================================
    opt.tocg_discriminator_save_step_checkpoint_dir = opt.tocg_discriminator_save_step_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    opt.tocg_discriminator_save_step_checkpoint_dir = fix(opt.tocg_discriminator_save_step_checkpoint_dir)
    opt.tocg_discriminator_save_step_checkpoint = os.path.join(opt.tocg_discriminator_save_step_checkpoint_dir, opt.tocg_discriminator_save_step_checkpoint)
    opt.tocg_discriminator_save_step_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.tocg_discriminator_save_step_checkpoint)
    opt.tocg_discriminator_save_step_checkpoint_dir = os.path.join("/",*opt.tocg_discriminator_save_step_checkpoint.split("/")[:-1])
    
    opt.tocg_discriminator_save_final_checkpoint_dir = opt.tocg_discriminator_save_final_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    opt.tocg_discriminator_save_final_checkpoint_dir = fix(opt.tocg_discriminator_save_final_checkpoint_dir)
    opt.tocg_discriminator_save_final_checkpoint = os.path.join(opt.tocg_discriminator_save_final_checkpoint_dir, opt.tocg_discriminator_save_final_checkpoint)
    opt.tocg_discriminator_save_final_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.tocg_discriminator_save_final_checkpoint)
    opt.tocg_discriminator_save_final_checkpoint_dir = os.path.join("/",*opt.tocg_discriminator_save_final_checkpoint.split("/")[:-1])
    
    
    opt.tocg_discriminator_load_final_checkpoint_dir = opt.tocg_discriminator_load_final_checkpoint_dir.format(root_opt.tocg_experiment_from_run, root_opt.tocg_discriminator_experiment_from_dir)
    opt.tocg_discriminator_load_final_checkpoint_dir = fix(opt.tocg_discriminator_load_final_checkpoint_dir)
    opt.tocg_discriminator_load_final_checkpoint = os.path.join(opt.tocg_discriminator_load_final_checkpoint_dir, opt.tocg_discriminator_load_final_checkpoint)
    opt.tocg_discriminator_load_final_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.tocg_discriminator_load_final_checkpoint)
    opt.tocg_discriminator_load_final_checkpoint_dir = os.path.join("/",*opt.tocg_discriminator_load_final_checkpoint.split("/")[:-1])

    if not last_step:
        opt.tocg_discriminator_load_step_checkpoint_dir = opt.tocg_discriminator_load_step_checkpoint_dir.format(root_opt.tocg_discriminator_experiment_from_run, root_opt.tocg_discriminator_experiment_from_dir)
    else:
        opt.tocg_discriminator_load_step_checkpoint_dir = opt.tocg_discriminator_load_step_checkpoint_dir.format(root_opt.tocg_discriminator_experiment_from_run, root_opt.this_viton_save_to_dir)
    opt.tocg_discriminator_load_step_checkpoint_dir = fix(opt.tocg_discriminator_load_step_checkpoint_dir)
    if not last_step:
        opt.tocg_discriminator_load_step_checkpoint = os.path.join(opt.tocg_discriminator_load_step_checkpoint_dir, opt.tocg_discriminator_load_step_checkpoint)
    else:
        if os.path.isdir(opt.tocg_discriminator_load_step_checkpoint_dir.format(root_opt.tocg_discriminator_experiment_from_run, root_opt.this_viton_save_to_dir)):
            os_list = os.listdir(opt.tocg_discriminator_load_step_checkpoint_dir.format(root_opt.tocg_discriminator_experiment_from_run, root_opt.this_viton_save_to_dir))
            os_list = [string for string in os_list if "tocg_discriminator" in string]
            last_step = sorted(os_list, key=sort_digit)[-1]
            opt.tocg_discriminator_load_step_checkpoint = os.path.join(opt.tocg_discriminator_load_step_checkpoint_dir, last_step)
    opt.tocg_discriminator_load_step_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.tocg_discriminator_load_step_checkpoint)
    opt.tocg_discriminator_load_step_checkpoint_dir = os.path.join("/",*opt.tocg_discriminator_load_step_checkpoint.split("/")[:-1])
    # ================================= gen =================================
    opt.gen_save_step_checkpoint_dir = opt.gen_save_step_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    opt.gen_save_step_checkpoint_dir = fix(opt.gen_save_step_checkpoint_dir)
    opt.gen_save_step_checkpoint = os.path.join(opt.gen_save_step_checkpoint_dir, opt.gen_save_step_checkpoint)
    opt.gen_save_step_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.gen_save_step_checkpoint)
    opt.gen_save_step_checkpoint_dir = os.path.join("/",*opt.gen_save_step_checkpoint.split("/")[:-1])
    
    opt.gen_save_final_checkpoint_dir = opt.gen_save_final_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    opt.gen_save_final_checkpoint_dir = fix(opt.gen_save_final_checkpoint_dir)
    opt.gen_save_final_checkpoint = os.path.join(opt.gen_save_final_checkpoint_dir, opt.gen_save_final_checkpoint)
    opt.gen_save_final_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.gen_save_final_checkpoint)
    opt.gen_save_final_checkpoint_dir = os.path.join("/",*opt.gen_save_final_checkpoint.split("/")[:-1])
    
    opt.gen_load_final_checkpoint_dir = opt.gen_load_final_checkpoint_dir.format(root_opt.gen_experiment_from_run, root_opt.gen_experiment_from_dir)
    opt.gen_load_final_checkpoint_dir = fix(opt.gen_load_final_checkpoint_dir)
    opt.gen_load_final_checkpoint = os.path.join(opt.gen_load_final_checkpoint_dir, opt.gen_load_final_checkpoint)
    opt.gen_load_final_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.gen_load_final_checkpoint)
    opt.gen_load_final_checkpoint_dir = os.path.join("/",*opt.gen_load_final_checkpoint.split("/")[:-1])
    
    if not last_step:
        opt.gen_load_step_checkpoint_dir = opt.gen_load_step_checkpoint_dir.format(root_opt.gen_experiment_from_run, root_opt.gen_experiment_from_dir)
    else:
        opt.gen_load_step_checkpoint_dir = opt.gen_load_step_checkpoint_dir.format(root_opt.gen_experiment_from_run, root_opt.this_viton_save_to_dir)
    opt.gen_load_step_checkpoint_dir = fix(opt.gen_load_step_checkpoint_dir)
    if not last_step:
        opt.gen_load_step_checkpoint = os.path.join(opt.gen_load_step_checkpoint_dir, opt.gen_load_step_checkpoint)
    else:
        if os.path.isdir(opt.gen_load_step_checkpoint_dir.format(root_opt.gen_experiment_from_run, root_opt.this_viton_save_to_dir)):
            os_list = os.listdir(opt.gen_load_step_checkpoint_dir.format(root_opt.gen_experiment_from_run, root_opt.this_viton_save_to_dir))
            os_list = [string for string in os_list if "gen" in string]
            last_step = sorted(os_list, key=sort_digit)[-1]
            opt.gen_load_step_checkpoint = os.path.join(opt.gen_load_step_checkpoint_dir, last_step)
    opt.gen_load_step_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.gen_load_step_checkpoint)
    opt.gen_load_step_checkpoint_dir = os.path.join("/",*opt.gen_load_step_checkpoint.split("/")[:-1])
    # ================================= gen DISCRIMINATOR =================================
    opt.gen_discriminator_save_step_checkpoint_dir = opt.gen_discriminator_save_step_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    opt.gen_discriminator_save_step_checkpoint_dir = fix(opt.gen_discriminator_save_step_checkpoint_dir)
    opt.gen_discriminator_save_step_checkpoint = os.path.join(opt.gen_discriminator_save_step_checkpoint_dir, opt.gen_discriminator_save_step_checkpoint)
    opt.gen_discriminator_save_step_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.gen_discriminator_save_step_checkpoint)
    opt.gen_discriminator_save_step_checkpoint_dir = os.path.join("/",*opt.gen_discriminator_save_step_checkpoint.split("/")[:-1])
    
    opt.gen_discriminator_save_final_checkpoint_dir = opt.gen_discriminator_save_final_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    opt.gen_discriminator_save_final_checkpoint_dir = fix(opt.gen_discriminator_save_final_checkpoint_dir)
    opt.gen_discriminator_save_final_checkpoint = os.path.join(opt.gen_discriminator_save_final_checkpoint_dir, opt.gen_discriminator_save_final_checkpoint)
    opt.gen_discriminator_save_final_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.gen_discriminator_save_final_checkpoint)
    opt.gen_discriminator_save_final_checkpoint_dir = os.path.join("/",*opt.gen_discriminator_save_final_checkpoint.split("/")[:-1])
    
    opt.gen_discriminator_load_final_checkpoint_dir = opt.gen_discriminator_load_final_checkpoint_dir.format(root_opt.gen_experiment_from_run, root_opt.gen_discriminator_experiment_from_dir)
    opt.gen_discriminator_load_final_checkpoint_dir = fix(opt.gen_discriminator_load_final_checkpoint_dir)
    opt.gen_discriminator_load_final_checkpoint = os.path.join(opt.gen_discriminator_load_final_checkpoint_dir, opt.gen_discriminator_load_final_checkpoint)
    opt.gen_discriminator_load_final_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.gen_discriminator_load_final_checkpoint)
    opt.gen_discriminator_load_final_checkpoint_dir = os.path.join("/",*opt.gen_discriminator_load_final_checkpoint.split("/")[:-1])
    
    if not last_step:
        opt.gen_discriminator_load_step_checkpoint_dir = opt.gen_discriminator_load_step_checkpoint_dir.format(root_opt.gen_discriminator_experiment_from_run, root_opt.gen_discriminator_experiment_from_dir)
    else:
        opt.gen_discriminator_load_step_checkpoint_dir = opt.gen_discriminator_load_step_checkpoint_dir.format(root_opt.gen_discriminator_experiment_from_run, root_opt.this_viton_save_to_dir)
    opt.gen_discriminator_load_step_checkpoint_dir = fix(opt.gen_discriminator_load_step_checkpoint_dir)
    if not last_step:
        opt.gen_discriminator_load_step_checkpoint = os.path.join(opt.gen_discriminator_load_step_checkpoint_dir, opt.gen_discriminator_load_step_checkpoint)
    else:
        if os.path.isdir(opt.gen_discriminator_load_step_checkpoint_dir.format(root_opt.gen_discriminator_experiment_from_run, root_opt.this_viton_save_to_dir)):
            os_list = os.listdir(opt.gen_discriminator_load_step_checkpoint_dir.format(root_opt.gen_discriminator_experiment_from_run, root_opt.this_viton_save_to_dir))
            os_list = [string for string in os_list if "gen_discriminator" in string]
            last_step = sorted(os_list, key=sort_digit)[-1]
            opt.gen_discriminator_load_step_checkpoint = os.path.join(opt.gen_discriminator_load_step_checkpoint_dir, last_step)
    opt.gen_discriminator_load_step_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.gen_discriminator_load_step_checkpoint)
    opt.gen_discriminator_load_step_checkpoint_dir = os.path.join("/",*opt.gen_discriminator_load_step_checkpoint.split("/")[:-1])
    return opt

def get_root_opt_results_dir(parser, root_opt):
    root_opt.transforms_dir = root_opt.transforms_dir.format(root_opt.dataset_name)
    parser.tensorboard_dir = parser.tensorboard_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    parser.results_dir = parser.results_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    return parser, root_opt

def copy_root_opt_to_opt(parser, root_opt):
    parser.display_count = root_opt.display_count
    parser.cuda = root_opt.cuda
    parser.device = int(root_opt.device)
    parser.dataset_name = root_opt.dataset_name
    parser.warp_load_from_model = root_opt.warp_load_from_model
    parser.load_last_step = root_opt.load_last_step if type(root_opt.load_last_step) == bool else eval(root_opt.load_last_step)
    parser.run_wandb = root_opt.run_wandb
    parser.viton_batch_size = root_opt.viton_batch_size
    parser.save_period = root_opt.save_period
    parser.print_step = root_opt.print_step
    parser.niter = root_opt.niter
    parser.niter_decay = root_opt.niter_decay
    parser.VITON_Type = root_opt.VITON_Type
    parser.VITON_selection_dir = parser.VITON_selection_dir.format(parser.VITON_Type, parser.VITON_Name)
    return parser

def process_opt(opt, root_opt):
    parser = opt
    parser = argparse.Namespace(**parser)
    root_opt = get_root_experiment_runs(root_opt)
    root_opt = get_root_opt_experiment_dir(root_opt)
    parser = get_root_opt_checkpoint_dir(parser, root_opt)
    parser, root_opt = get_root_opt_results_dir(parser, root_opt)    
    parser = copy_root_opt_to_opt(parser, root_opt)
    return parser, root_opt



def remove_overlap(seg_out, warped_cm):
    
    assert len(warped_cm.shape) == 4
    
    warped_cm = warped_cm - (torch.cat([seg_out[:, 1:3, :, :], seg_out[:, 5:, :, :]], dim=1)).sum(dim=1, keepdim=True) * warped_cm
    return warped_cm

def load_checkpoint_G(model, checkpoint_path,opt):
    if not os.path.exists(checkpoint_path):
        print("Invalid path!")
        return
    state_dict = torch.load(checkpoint_path)
    new_state_dict = OrderedDict([(k.replace('ace', 'alias').replace('.Spade', ''), v) for (k, v) in state_dict.items()])
    new_state_dict._metadata = OrderedDict([(k.replace('ace', 'alias').replace('.Spade', ''), v) for (k, v) in state_dict._metadata.items()])
    model.load_state_dict(new_state_dict, strict=True)
    if opt.cuda :
        model.cuda()

def test(opt, test_loader, tocg, generator):
    gauss = tgm.image.GaussianBlur((15, 15), (3, 3))
    if opt.cuda:
        gauss = gauss.cuda()

    if opt.cuda :
        tocg.cuda()
    tocg.eval()
    generator.eval()
    
    output_dir = os.path.join(opt.results_dir, opt.datamode,'output')
    grid_dir = os.path.join(opt.results_dir, opt.datamode,'grid')

    
    os.makedirs(grid_dir, exist_ok=True)
    
    os.makedirs(output_dir, exist_ok=True)
    
    tocg = DataParallelWithCallback(tocg, device_ids=[0])
    generator = DataParallelWithCallback(generator, device_ids=[0])
     
    num = 0
    iter_start_time = time.time()
    with torch.no_grad():
        for inputs in test_loader.data_loader:

            if opt.cuda :
                pose_map = inputs['pose'].cuda()
                pre_clothes_mask = inputs['cloth_mask']['paired'].cuda()
                label = inputs['parse']
                parse_agnostic = inputs['parse_agnostic']
                agnostic = inputs['agnostic'].cuda()
                parse_cloth = inputs['parse_cloth'].cuda()
                clothes = inputs['cloth']['paired'].cuda() # target cloth
                densepose = inputs['densepose'].cuda()
                im = inputs['image']
                input_label, input_parse_agnostic = label.cuda(), parse_agnostic.cuda()
                pre_clothes_mask = torch.FloatTensor((pre_clothes_mask.detach().cpu().numpy() > 0.5).astype(np.float)).cuda()
            else :
                pose_map = inputs['pose']
                pre_clothes_mask = inputs['cloth_mask']['paired']
                label = inputs['parse']
                parse_agnostic = inputs['parse_agnostic']
                agnostic = inputs['agnostic']
                parse_cloth = inputs['parse_cloth']
                clothes = inputs['cloth']['paired'] # target cloth
                densepose = inputs['densepose']
                im = inputs['image']
                input_label, input_parse_agnostic = label, parse_agnostic
                pre_clothes_mask = torch.FloatTensor((pre_clothes_mask.detach().cpu().numpy() > 0.5).astype(np.float))



            # down
            pose_map_down = F.interpolate(pose_map, size=(256, 192), mode='bilinear')
            pre_clothes_mask_down = F.interpolate(pre_clothes_mask, size=(256, 192), mode='nearest')
            input_label_down = F.interpolate(input_label, size=(256, 192), mode='bilinear')
            input_parse_agnostic_down = F.interpolate(input_parse_agnostic, size=(256, 192), mode='nearest')
            agnostic_down = F.interpolate(agnostic, size=(256, 192), mode='nearest')
            clothes_down = F.interpolate(clothes, size=(256, 192), mode='bilinear')
            densepose_down = F.interpolate(densepose, size=(256, 192), mode='bilinear')

            shape = pre_clothes_mask.shape
            
            # multi-task inputs
            input1 = torch.cat([clothes_down, pre_clothes_mask_down], 1)
            input2 = torch.cat([input_parse_agnostic_down, densepose_down], 1)

            # forward
            # flow_list, fake_segmap, warped_cloth_paired, warped_clothmask_paired = tocg(opt,input1, input2)
            if opt.segment_anything:
                flow_list, fake_segmap, warped_cloth_paired, warped_clothmask_paired = tocg(opt, input1, input2, im_c=parse_cloth)
            else:
                flow_list, fake_segmap, warped_cloth_paired, warped_clothmask_paired = tocg(opt, input1, input2)
            
            # warped cloth mask one hot
            if opt.cuda :
                warped_cm_onehot = torch.FloatTensor((warped_clothmask_paired.detach().cpu().numpy() > 0.5).astype(np.float)).cuda()
            else :
                warped_cm_onehot = torch.FloatTensor((warped_clothmask_paired.detach().cpu().numpy() > 0.5).astype(np.float))

            if opt.clothmask_composition != 'no_composition':
                if opt.clothmask_composition == 'detach':
                    cloth_mask = torch.ones_like(fake_segmap)
                    cloth_mask[:,3:4, :, :] = warped_cm_onehot
                    fake_segmap = fake_segmap * cloth_mask
                    
                if opt.clothmask_composition == 'warp_grad':
                    cloth_mask = torch.ones_like(fake_segmap)
                    cloth_mask[:,3:4, :, :] = warped_clothmask_paired
                    fake_segmap = fake_segmap * cloth_mask
                    
            # make generator input parse map
            fake_parse_gauss = gauss(F.interpolate(fake_segmap, size=(opt.fine_height, opt.fine_width), mode='bilinear'))
            fake_parse = fake_parse_gauss.argmax(dim=1)[:, None]

            if opt.cuda :
                old_parse = torch.FloatTensor(fake_parse.size(0), 13, opt.fine_height, opt.fine_width).zero_().cuda()
            else:
                old_parse = torch.FloatTensor(fake_parse.size(0), 13, opt.fine_height, opt.fine_width).zero_()
            old_parse.scatter_(1, fake_parse, 1.0)

            labels = {
                0:  ['background',  [0]],
                1:  ['paste',       [2, 4, 7, 8, 9, 10, 11]],
                2:  ['upper',       [3]],
                3:  ['hair',        [1]],
                4:  ['left_arm',    [5]],
                5:  ['right_arm',   [6]],
                6:  ['noise',       [12]]
            }
            if opt.cuda :
                parse = torch.FloatTensor(fake_parse.size(0), 7, opt.fine_height, opt.fine_width).zero_().cuda()
            else:
                parse = torch.FloatTensor(fake_parse.size(0), 7, opt.fine_height, opt.fine_width).zero_()
            for i in range(len(labels)):
                for label in labels[i][1]:
                    parse[:, i] += old_parse[:, label]
                    
            # warped cloth
            N, _, iH, iW = clothes.shape
            flow = F.interpolate(flow_list[-1].permute(0, 3, 1, 2), size=(iH, iW), mode='bilinear').permute(0, 2, 3, 1)
            flow_norm = torch.cat([flow[:, :, :, 0:1] / ((96 - 1.0) / 2.0), flow[:, :, :, 1:2] / ((128 - 1.0) / 2.0)], 3)
            
            grid = make_grid(N, iH, iW,opt)
            warped_grid = grid + flow_norm
            warped_cloth = F.grid_sample(clothes, warped_grid, padding_mode='border')
            warped_clothmask = F.grid_sample(pre_clothes_mask, warped_grid, padding_mode='border')
            if opt.occlusion:
                warped_clothmask = remove_overlap(F.softmax(fake_parse_gauss, dim=1), warped_clothmask)
                warped_cloth = warped_cloth * warped_clothmask + torch.ones_like(warped_cloth) * (1-warped_clothmask)
            

            output = generator(torch.cat((agnostic, densepose, warped_cloth), dim=1), parse)
            # visualize
            unpaired_names = []
            for i in range(shape[0]):
                grid = make_image_grid([(clothes[i].cpu() / 2 + 0.5), (pre_clothes_mask[i].cpu()).expand(3, -1, -1), visualize_segmap(parse_agnostic.cpu(), batch=i), ((densepose.cpu()[i]+1)/2),
                                        (warped_cloth[i].cpu().detach() / 2 + 0.5), (warped_clothmask[i].cpu().detach()).expand(3, -1, -1), visualize_segmap(fake_parse_gauss.cpu(), batch=i),
                                        (pose_map[i].cpu()/2 +0.5), (warped_cloth[i].cpu()/2 + 0.5), (agnostic[i].cpu()/2 + 0.5),
                                        (im[i]/2 +0.5), (output[i].cpu()/2 +0.5)],
                                        nrow=4)
                unpaired_name = (inputs['c_name']['paired'][i].split('.')[0] + '_' + inputs['c_name']['paired'][i].split('.')[0] + '.png')
                save_image(grid, os.path.join(grid_dir, unpaired_name))
                unpaired_names.append(unpaired_name)
                
            # save output
            save_images(output, unpaired_names, output_dir)
                
            num += shape[0]
            print(num)

    print(f"Test time {time.time() - iter_start_time}")


def test_hrviton_gen_(opt, root_opt):
    opt,root_opt = process_opt(opt, root_opt)
    print("Start to test %s!")
    _test_hrviton_gen_(opt, root_opt)
    

def _test_hrviton_gen_(opt, root_opt):
    print("Start to test %s!")
    
    # create test dataset & loader
    test_dataset = FashionNeRFDataset(root_opt, opt, viton=True, mode='test', model='viton')
    test_loader = FashionDataLoader(test_dataset, opt.num_test_visualize, 1, False)
    
    input1_nc = 4  # cloth + cloth-mask
    input2_nc = opt.semantic_nc + 3  # parse_agnostic + densepose
    tocg = ConditionGenerator(opt, input1_nc=input1_nc, input2_nc=input2_nc, output_nc=opt.output_nc, ngf=96, norm_layer=nn.BatchNorm2d)
       
    # generator
    opt.semantic_nc = 7
    generator = SPADEGenerator(opt, 3+3+3)
    generator.print_network()
       
    # Load Checkpoint
    load_checkpoint(tocg, opt.tocg_load_final_checkpoint,opt)
    load_checkpoint_G(generator, opt.gen_load_final_checkpoint,opt)

    # Train
    test(opt, test_loader, tocg, generator)

    print("Finished testing!")


