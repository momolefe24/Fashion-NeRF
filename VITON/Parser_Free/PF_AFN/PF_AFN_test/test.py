import time
import argparse
from VITON.Parser_Free.PF_AFN.PF_AFN_test.data.base_data_loader import CreateDataLoader, CreateDataTestLoader
from torchvision.utils import make_grid, save_image
from VITON.Parser_Free.PF_AFN.PF_AFN_test.models.networks import ResUnetGenerator, load_checkpoint
from VITON.Parser_Free.PF_AFN.PF_AFN_test.models.afwm import AFWM
import torch.nn as nn
import os
import numpy as np
import torch
import cv2
import torch.nn.functional as F

fix = lambda path: os.path.normpath(path)

def get_root_experiment_runs(root_opt):
    root_opt.experiment_run = root_opt.experiment_run.format(root_opt.experiment_number, root_opt.run_number)
    root_opt.experiment_from_run = root_opt.experiment_from_run.format(root_opt.experiment_from_number, root_opt.run_from_number)
    
    # Parser Based
    root_opt.parser_based_warp_experiment_from_run = root_opt.parser_based_warp_experiment_from_run.format(root_opt.parser_based_warp_experiment_from_number, root_opt.parser_based_warp_run_from_number)
    root_opt.parser_based_gen_experiment_from_run = root_opt.parser_based_gen_experiment_from_run.format(root_opt.parser_based_gen_experiment_from_number, root_opt.parser_based_gen_run_from_number)
    
    # Parser Free
    root_opt.parser_free_warp_experiment_from_run = root_opt.parser_free_warp_experiment_from_run.format(root_opt.parser_free_warp_experiment_from_number, root_opt.parser_free_warp_run_from_number)
    root_opt.parser_free_gen_experiment_from_run = root_opt.parser_free_gen_experiment_from_run.format(root_opt.parser_free_gen_experiment_from_number, root_opt.parser_free_gen_run_from_number)
    return root_opt


def get_root_opt_experiment_dir(root_opt):
    root_opt.rail_dir = root_opt.rail_dir.format(root_opt.dataset_name, root_opt.res, root_opt.datamode)
    root_opt.original_dir = root_opt.original_dir.format(root_opt.dataset_name, root_opt.res, root_opt.datamode)
    if root_opt.res == 'low_res':
        root_opt.original_dir = root_opt.original_dir.replace(root_opt.res, os.path.join(root_opt.res, root_opt.low_res_dataset_name))
        
    # This experiment
    root_opt.this_viton_save_to_dir = os.path.join(root_opt.this_viton_save_to_dir, root_opt.VITON_Model)
    root_opt.this_viton_load_from_dir = root_opt.this_viton_load_from_dir.format(root_opt.VITON_Type, root_opt.VITON_Name, root_opt.warp_load_from_model)
    root_opt.this_viton_load_from_dir = os.path.join(root_opt.this_viton_load_from_dir, root_opt.VITON_Model)
    
    # Parser Based e2e
    root_opt.parser_based_warp_experiment_from_dir = root_opt.parser_based_warp_experiment_from_dir.format(root_opt.VITON_Type, root_opt.VITON_Name, root_opt.warp_load_from_model)
    root_opt.parser_based_warp_experiment_from_dir = os.path.join(root_opt.parser_based_warp_experiment_from_dir, "PB_Warp")
    
    root_opt.parser_based_gen_experiment_from_dir = root_opt.parser_based_gen_experiment_from_dir.format(root_opt.VITON_Type, root_opt.VITON_Name, root_opt.gen_load_from_model)
    root_opt.parser_based_gen_experiment_from_dir = os.path.join(root_opt.parser_based_gen_experiment_from_dir, "PB_Gen")
    
    
    # Parser Free Warp
    root_opt.parser_free_warp_experiment_from_dir = root_opt.parser_free_warp_experiment_from_dir.format(root_opt.VITON_Type, root_opt.VITON_Name, root_opt.parser_free_warp_load_from_model)
    root_opt.parser_free_warp_experiment_from_dir = os.path.join(root_opt.parser_free_warp_experiment_from_dir, "PF_Warp")
    
    root_opt.parser_free_gen_experiment_from_dir = root_opt.parser_free_gen_experiment_from_dir.format(root_opt.VITON_Type, root_opt.VITON_Name, root_opt.parser_free_gen_load_from_model)
    root_opt.parser_free_gen_experiment_from_dir = os.path.join(root_opt.parser_free_gen_experiment_from_dir, "PF_Gen")
    
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
    parser.datamode = root_opt.datamode
    parser.warp_load_from_model = root_opt.warp_load_from_model
    parser.load_last_step = root_opt.load_last_step if type(root_opt.load_last_step) == bool else eval(root_opt.load_last_step)
    parser.run_wandb = root_opt.run_wandb
    parser.viton_batch_size = root_opt.viton_batch_size
    parser.save_period = root_opt.save_period
    parser.print_step = root_opt.print_step
    parser.niter = root_opt.niter
    parser.niter_decay = root_opt.niter_decay
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


def get_root_opt_checkpoint_dir(parser, root_opt):
    last_step = root_opt.load_last_step if type(root_opt.load_last_step) == bool else eval(root_opt.load_last_step)
    sort_digit = lambda name: int(name.split('_')[-1].split('.')[0])
    # Parser Based Warping
    parser.pb_warp_save_step_checkpoint_dir = parser.pb_warp_save_step_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    parser.pb_warp_save_step_checkpoint = os.path.join(parser.pb_warp_save_step_checkpoint_dir, parser.pb_warp_save_step_checkpoint)
    parser.pb_warp_save_step_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pb_warp_save_step_checkpoint)
    parser.pb_warp_save_step_checkpoint_dir = os.path.join("/",*parser.pb_warp_save_step_checkpoint.split("/")[:-1])

    parser.pb_warp_load_step_checkpoint_dir = parser.pb_warp_load_step_checkpoint_dir.format(root_opt.parser_based_warp_experiment_from_run, root_opt.parser_based_warp_experiment_from_dir)
    parser.pb_warp_load_step_checkpoint = os.path.join(parser.pb_warp_load_step_checkpoint_dir, parser.pb_warp_load_step_checkpoint)
    parser.pb_warp_load_step_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pb_warp_load_step_checkpoint)
    parser.pb_warp_load_step_checkpoint_dir = os.path.join("/",*parser.pb_warp_load_step_checkpoint.split("/")[:-1])

    parser.pb_warp_save_final_checkpoint_dir = parser.pb_warp_save_final_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    parser.pb_warp_save_final_checkpoint = os.path.join(parser.pb_warp_save_final_checkpoint_dir, parser.pb_warp_save_final_checkpoint)
    parser.pb_warp_save_final_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pb_warp_save_final_checkpoint)
    parser.pb_warp_save_final_checkpoint_dir = os.path.join("/",*parser.pb_warp_save_final_checkpoint.split("/")[:-1])

    parser.pb_warp_load_final_checkpoint_dir = parser.pb_warp_load_final_checkpoint_dir.format(root_opt.parser_based_warp_experiment_from_run, root_opt.parser_based_warp_experiment_from_dir)
    parser.pb_warp_load_final_checkpoint = os.path.join(parser.pb_warp_load_final_checkpoint_dir, parser.pb_warp_load_final_checkpoint)
    parser.pb_warp_load_final_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pb_warp_load_final_checkpoint)
    parser.pb_warp_load_final_checkpoint_dir = os.path.join("/",*parser.pb_warp_load_final_checkpoint.split("/")[:-1])
    
    # Parser Based Gen
    parser.pb_gen_save_step_checkpoint_dir = parser.pb_gen_save_step_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    parser.pb_gen_save_step_checkpoint = os.path.join(parser.pb_gen_save_step_checkpoint_dir, parser.pb_gen_save_step_checkpoint)
    parser.pb_gen_save_step_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pb_gen_save_step_checkpoint)
    parser.pb_gen_save_step_checkpoint_dir = os.path.join("/",*parser.pb_gen_save_step_checkpoint.split("/")[:-1])
    
    parser.pb_gen_load_step_checkpoint_dir = parser.pb_gen_load_step_checkpoint_dir.format(root_opt.parser_based_gen_experiment_from_run, root_opt.parser_based_gen_experiment_from_dir)
    parser.pb_gen_load_step_checkpoint = os.path.join(parser.pb_gen_load_step_checkpoint_dir, parser.pb_gen_load_step_checkpoint)
    parser.pb_gen_load_step_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pb_gen_load_step_checkpoint)
    parser.pb_gen_load_step_checkpoint_dir = os.path.join("/",*parser.pb_gen_load_step_checkpoint.split("/")[:-1])

    parser.pb_gen_save_final_checkpoint_dir = parser.pb_gen_save_final_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    parser.pb_gen_save_final_checkpoint = os.path.join(parser.pb_gen_save_final_checkpoint_dir, parser.pb_gen_save_final_checkpoint)
    parser.pb_gen_save_final_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pb_gen_save_final_checkpoint)
    parser.pb_gen_save_final_checkpoint_dir = os.path.join("/",*parser.pb_gen_save_final_checkpoint.split("/")[:-1])
    
    parser.pb_gen_load_final_checkpoint_dir = parser.pb_gen_load_final_checkpoint_dir.format(root_opt.parser_based_gen_experiment_from_run, root_opt.parser_based_gen_experiment_from_dir)
    parser.pb_gen_load_final_checkpoint = os.path.join(parser.pb_gen_load_final_checkpoint_dir, parser.pb_gen_load_final_checkpoint)
    parser.pb_gen_load_final_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pb_gen_load_final_checkpoint)
    parser.pb_gen_load_final_checkpoint_dir = os.path.join("/",*parser.pb_gen_load_final_checkpoint.split("/")[:-1])
    # Parser Free Warping
    parser.pf_warp_save_step_checkpoint_dir = parser.pf_warp_save_step_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    parser.pf_warp_save_step_checkpoint = os.path.join(parser.pf_warp_save_step_checkpoint_dir, parser.pf_warp_save_step_checkpoint)
    parser.pf_warp_save_step_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pf_warp_save_step_checkpoint)
    parser.pf_warp_save_step_checkpoint_dir = os.path.join("/",*parser.pf_warp_save_step_checkpoint.split("/")[:-1])
    
    parser.pf_warp_load_step_checkpoint_dir = parser.pf_warp_load_step_checkpoint_dir.format(root_opt.parser_free_warp_experiment_from_run, root_opt.parser_free_warp_experiment_from_dir)
    parser.pf_warp_load_step_checkpoint = os.path.join(parser.pf_warp_load_step_checkpoint_dir, parser.pf_warp_load_step_checkpoint)
    parser.pf_warp_load_step_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pf_warp_load_step_checkpoint)
    parser.pf_warp_load_step_checkpoint_dir = os.path.join("/",*parser.pf_warp_load_step_checkpoint.split("/")[:-1])
    
    parser.pf_warp_save_final_checkpoint_dir = parser.pf_warp_save_final_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    parser.pf_warp_save_final_checkpoint = os.path.join(parser.pf_warp_save_final_checkpoint_dir, parser.pf_warp_save_final_checkpoint)
    parser.pf_warp_save_final_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pf_warp_save_final_checkpoint)

    parser.pf_warp_load_final_checkpoint_dir = parser.pf_warp_load_final_checkpoint_dir.format(root_opt.parser_free_warp_experiment_from_run, root_opt.parser_free_warp_experiment_from_dir)
    parser.pf_warp_load_final_checkpoint = os.path.join(parser.pf_warp_load_final_checkpoint_dir, parser.pf_warp_load_final_checkpoint)
    parser.pf_warp_load_final_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pf_warp_load_final_checkpoint)
    parser.pf_warp_load_final_checkpoint_dir = os.path.join("/",*parser.pf_warp_load_final_checkpoint.split("/")[:-1])
    # Parser Free Gen
    parser.pf_gen_save_step_checkpoint_dir = parser.pf_gen_save_step_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    parser.pf_gen_save_step_checkpoint = os.path.join(parser.pf_gen_save_step_checkpoint_dir, parser.pf_gen_save_step_checkpoint)
    parser.pf_gen_save_step_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pf_gen_save_step_checkpoint)
    parser.pf_gen_save_step_checkpoint_dir = os.path.join("/",*parser.pf_gen_save_step_checkpoint.split("/")[:-1])
    # parser.pf_gen_load_step_checkpoint_dir = parser.pf_gen_load_step_checkpoint_dir.format(root_opt.parser_free_gen_experiment_from_run, root_opt.parser_free_gen_experiment_from_dir)
    # parser.pf_gen_load_step_checkpoint = os.path.join(parser.pf_gen_load_step_checkpoint_dir, parser.pf_gen_load_step_checkpoint)
    
    parser.pf_gen_save_final_checkpoint_dir = parser.pf_gen_save_final_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    parser.pf_gen_save_final_checkpoint = os.path.join(parser.pf_gen_save_final_checkpoint_dir, parser.pf_gen_save_final_checkpoint)
    parser.pf_gen_save_final_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pf_gen_save_final_checkpoint)
    parser.pf_gen_save_final_checkpoint_dir = os.path.join("/",*parser.pf_gen_save_final_checkpoint.split("/")[:-1])
    
    parser.pf_gen_load_final_checkpoint_dir = parser.pf_gen_load_final_checkpoint_dir.format(root_opt.parser_free_gen_experiment_from_run, root_opt.parser_free_gen_experiment_from_dir)
    parser.pf_gen_load_final_checkpoint = os.path.join(parser.pf_gen_load_final_checkpoint_dir, parser.pf_gen_load_final_checkpoint)
    parser.pf_gen_load_final_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pf_gen_load_final_checkpoint)
    parser.pf_gen_load_final_checkpoint_dir = os.path.join("/",*parser.pf_gen_load_final_checkpoint.split("/")[:-1])
    if not last_step:
        parser.pf_gen_load_step_checkpoint_dir = parser.pf_gen_load_step_checkpoint_dir.format(root_opt.parser_free_gen_experiment_from_run, root_opt.parser_free_gen_experiment_from_dir)
    else:
        parser.pf_gen_load_step_checkpoint_dir = parser.pf_gen_load_step_checkpoint_dir.format(root_opt.parser_free_gen_experiment_from_run, root_opt.this_viton_save_to_dir)
    parser.pf_gen_load_step_checkpoint_dir = fix(parser.pf_gen_load_step_checkpoint_dir)
    if not last_step:
        parser.pf_gen_load_step_checkpoint_dir = os.path.join(parser.pf_gen_load_step_checkpoint_dir, parser.pf_gen_load_step_checkpoint)
    else:
        if os.path.isdir(parser.pf_gen_load_step_checkpoint_dir.format(root_opt.pf_gen_experiment_from_run, root_opt.this_viton_save_to_dir)):
            os_list = os.listdir(parser.pf_gen_load_step_checkpoint_dir.format(root_opt.pf_gen_experiment_from_run, root_opt.this_viton_save_to_dir))
            os_list = [string for string in os_list if "pf_gen" in string]
            last_step = sorted(os_list, key=sort_digit)[-1]
            parser.pf_gen_load_step_checkpoint = os.path.join(parser.pf_gen_load_step_checkpoint_dir, last_step)
    parser.pf_gen_load_step_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pf_gen_load_step_checkpoint)
    parser.pf_gen_load_step_checkpoint_dir = os.path.join("/",*parser.pf_gen_load_step_checkpoint.split("/")[:-1])
    return parser


def test_pfafn_(opt, root_opt):
    opt,root_opt = process_opt(opt, root_opt)
    _test_pfafn_(opt, root_opt)


def _test_pfafn_(opt, root_opt):
    start_epoch, epoch_iter = 1, 0
    data_loader = CreateDataTestLoader(opt, root_opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print(dataset_size)

    warp_model = AFWM(opt, 3)
    print(warp_model)
    warp_model.eval()
    warp_model.cuda()
    load_checkpoint(warp_model, opt.pf_warp_load_final_checkpoint)

    gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
    print(gen_model)
    gen_model.eval()
    gen_model.cuda()
    load_checkpoint(gen_model, opt.pf_gen_load_final_checkpoint)

    total_steps = (start_epoch-1) * dataset_size + epoch_iter
    step = 0
    step_per_batch = dataset_size / opt.viton_batch_size
    prediction_dir = os.path.join(opt.results_dir, 'prediction')
    ground_truth_dir = os.path.join(opt.results_dir, 'ground_truth')
    ground_truth_mask_dir = os.path.join(opt.results_dir, 'ground_truth_mask')
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)
    if not os.path.exists(ground_truth_dir):
        os.makedirs(ground_truth_dir)
    if not os.path.exists(ground_truth_mask_dir):
        os.makedirs(ground_truth_mask_dir)
    for i, data in enumerate(dataset, start=epoch_iter):
        iter_start_time = time.time()
        total_steps += opt.viton_batch_size
        epoch_iter += opt.viton_batch_size
        
        real_image = data['image']
        clothes = data['clothes']
        t_mask = torch.FloatTensor((data['label'].cpu().numpy()==7).astype(np.float))
        data['label'] = data['label']*(1-t_mask)+t_mask*4
        edge = data['edge']
        person_clothes_edge = torch.FloatTensor((data['label'].cpu().numpy()==4).astype(np.int))
        pre_clothes_edge = torch.FloatTensor((edge.detach().cpu().numpy() > 0.5).astype(np.int64))
        clothes = clothes * pre_clothes_edge
        # edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int))
        # clothes = clothes * edge        
        person_clothes = real_image * person_clothes_edge
        flow_out = warp_model(real_image.cuda(), clothes.cuda())
        warped_cloth, last_flow, = flow_out
        warped_edge = F.grid_sample(edge.cuda(), last_flow.permute(0, 2, 3, 1),
                        mode='bilinear', padding_mode='zeros')

        gen_inputs = torch.cat([real_image.cuda(), warped_cloth, warped_edge], 1)
        gen_outputs = gen_model(gen_inputs)
        p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)
        m_composite = m_composite * warped_edge
        p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)
        
        
        image_name = os.path.join(prediction_dir, data['im_name'][0])
        ground_truth_image_name = os.path.join(ground_truth_dir, data['im_name'][0])
        ground_truth_mask_name = os.path.join(ground_truth_mask_dir, data['im_name'][0])
        if opt.VITON_Model == 'PF_Warp' or opt.VITON_Model == 'PB_Warp':
            save_image(warped_cloth, image_name, normalize=True, value_range=(-1,1))
            save_image(person_clothes, ground_truth_image_name, normalize=True, value_range=(-1,1))
            save_image(person_clothes_edge, ground_truth_mask_name)
        elif opt.VITON_Model == 'PF_Gen' or opt.VITON_Model == 'PB_Gen':
            save_image(p_tryon, image_name, normalize=True, value_range=(-1,1))
            save_image(real_image, ground_truth_image_name, normalize=True, value_range=(-1,1))
            save_image(person_clothes_edge, ground_truth_mask_name)
    



