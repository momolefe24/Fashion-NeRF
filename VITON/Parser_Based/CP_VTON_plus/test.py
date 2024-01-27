# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import time
from VITON.Parser_Based.CP_VTON_plus.cp_dataset import CPDataset, CPDataLoader
from VITON.Parser_Based.CP_VTON_plus.networks import GMM, UnetGenerator, load_checkpoint

from tensorboardX import SummaryWriter
from VITON.Parser_Based.CP_VTON_plus.visualization import board_add_image, board_add_images, save_images
fix = lambda path: os.path.normpath(path)

def test_gmm(opt, test_loader, model, board):
    model.cuda()
    model.eval()

    save_dir = os.path.join(opt.results_dir, opt.datamode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    warp_cloth_dir = os.path.join(save_dir, 'warp-cloth')
    if not os.path.exists(warp_cloth_dir):
        os.makedirs(warp_cloth_dir)
    warp_mask_dir = os.path.join(save_dir, 'warp-mask')
    if not os.path.exists(warp_mask_dir):
        os.makedirs(warp_mask_dir)
    result_dir1 = os.path.join(save_dir, 'result_dir')
    if not os.path.exists(result_dir1):
        os.makedirs(result_dir1)
    overlayed_TPS_dir = os.path.join(save_dir, 'overlayed_TPS')
    if not os.path.exists(overlayed_TPS_dir):
        os.makedirs(overlayed_TPS_dir)
    warped_grid_dir = os.path.join(save_dir, 'warped_grid')
    if not os.path.exists(warped_grid_dir):
        os.makedirs(warped_grid_dir)
    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()

        c_names = inputs['c_name']
        im_names = inputs['im_name']
        im = inputs['image'].cuda()
        im_pose = inputs['pose_image'].cuda()
        im_h = inputs['head'].cuda()
        shape = inputs['shape'].cuda()
        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        im_c = inputs['parse_cloth'].cuda()
        im_g = inputs['grid_image'].cuda()
        shape_ori = inputs['shape_ori']  # original body shape without blurring

        grid, theta = model(agnostic, cm)
        warped_cloth = F.grid_sample(c, grid, padding_mode='border')
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
        warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')
        overlay = 0.7 * warped_cloth + 0.3 * im

        visuals = [[im_h, shape, im_pose],
                   [c, warped_cloth, im_c],
                   [warped_grid, (warped_cloth+im)*0.5, im]]

        # save_images(warped_cloth, c_names, warp_cloth_dir)
        # save_images(warped_mask*2-1, c_names, warp_mask_dir)
        save_images(warped_cloth, im_names, warp_cloth_dir)
        save_images(warped_mask * 2 - 1, im_names, warp_mask_dir)
        save_images(shape_ori.cuda() * 0.2 + warped_cloth *
                    0.8, im_names, result_dir1)
        save_images(warped_grid, im_names, warped_grid_dir)
        save_images(overlay, im_names, overlayed_TPS_dir)

        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f' % (step+1, t), flush=True)


def test_tom(opt, test_loader, gmm_model, model, board):
    model.cuda()
    model.eval()

    gmm_model.cuda()
    gmm_model.eval()
    save_dir = os.path.join(opt.results_dir, opt.datamode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    try_on_dir = os.path.join(save_dir, 'try-on')
    if not os.path.exists(try_on_dir):
        os.makedirs(try_on_dir)
    p_rendered_dir = os.path.join(save_dir, 'p_rendered')
    if not os.path.exists(p_rendered_dir):
        os.makedirs(p_rendered_dir)
    m_composite_dir = os.path.join(save_dir, 'm_composite')
    if not os.path.exists(m_composite_dir):
        os.makedirs(m_composite_dir)
    im_pose_dir = os.path.join(save_dir, 'im_pose')
    if not os.path.exists(im_pose_dir):
        os.makedirs(im_pose_dir)
    shape_dir = os.path.join(save_dir, 'shape')
    if not os.path.exists(shape_dir):
        os.makedirs(shape_dir)
    im_h_dir = os.path.join(save_dir, 'im_h')
    if not os.path.exists(im_h_dir):
        os.makedirs(im_h_dir)  # for test data

    print('Dataset size: %05d!' % (len(test_loader.dataset)), flush=True)
    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()

        im_names = inputs['im_name']
        im = inputs['image'].cuda()
        im_pose = inputs['pose_image']
        im_h = inputs['head']
        shape = inputs['shape']

        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()

        # outputs = model(torch.cat([agnostic, c], 1))  # CP-VTON
        outputs = model(torch.cat([agnostic, c, cm], 1))  # CP-VTON+
        p_rendered, m_composite = torch.split(outputs, 3, 1)
        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)
        p_tryon = c * m_composite + p_rendered * (1 - m_composite)


        grid, theta = gmm_model(agnostic, cm)
        warped_cloth = F.grid_sample(c, grid, padding_mode='border')
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
        
        visuals = [[im_h, shape, im_pose],
                   [warped_cloth, 2*warped_mask-1, m_composite],
                   [p_rendered, p_tryon, im]]

        save_images(p_tryon, im_names, try_on_dir)
        save_images(im_h, im_names, im_h_dir)
        save_images(shape, im_names, shape_dir)
        save_images(im_pose, im_names, im_pose_dir)
        save_images(m_composite, im_names, m_composite_dir)
        save_images(p_rendered, im_names, p_rendered_dir)  # For test data

        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f' % (step+1, t), flush=True)

def get_root_experiment_runs(root_opt):
    root_opt.experiment_run = root_opt.experiment_run.format(root_opt.experiment_number, root_opt.run_number)
    root_opt.experiment_from_run = root_opt.experiment_from_run.format(root_opt.experiment_from_number, root_opt.run_from_number)
    root_opt.gmm_experiment_from_run = root_opt.gmm_experiment_from_run.format(root_opt.gmm_experiment_from_number, root_opt.gmm_run_from_number)
    root_opt.tom_experiment_from_run = root_opt.tom_experiment_from_run.format(root_opt.tom_experiment_from_number, root_opt.tom_run_from_number)
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
    
    # GMM
    root_opt.gmm_experiment_from_dir = root_opt.gmm_experiment_from_dir.format(root_opt.VITON_Type, root_opt.VITON_Name, root_opt.gmm_load_from_model)
    root_opt.gmm_experiment_from_dir = os.path.join(root_opt.gmm_experiment_from_dir, "GMM")
    
    # TOM
    root_opt.tom_experiment_from_dir = root_opt.tom_experiment_from_dir.format(root_opt.VITON_Type, root_opt.VITON_Name, root_opt.tom_load_from_model)
    root_opt.tom_experiment_from_dir = os.path.join(root_opt.tom_experiment_from_dir, 'TOM')    
    return root_opt


def get_root_opt_checkpoint_dir(opt, root_opt):
    last_step = root_opt.load_last_step if type(root_opt.load_last_step) == bool else eval(root_opt.load_last_step)
    sort_digit = lambda name: int(name.split('_')[-1].split('.')[0])
    # ================================= GMM =================================
    opt.gmm_save_step_checkpoint_dir = opt.gmm_save_step_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    opt.gmm_save_step_checkpoint_dir = fix(opt.gmm_save_step_checkpoint_dir)
    opt.gmm_save_step_checkpoint = os.path.join(opt.gmm_save_step_checkpoint_dir, opt.gmm_save_step_checkpoint)
    opt.gmm_save_step_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.gmm_save_step_checkpoint)
    
    opt.gmm_save_final_checkpoint_dir = opt.gmm_save_final_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    opt.gmm_save_final_checkpoint_dir = fix(opt.gmm_save_final_checkpoint_dir)
    opt.gmm_save_final_checkpoint = os.path.join(opt.gmm_save_final_checkpoint_dir, opt.gmm_save_final_checkpoint)
    opt.gmm_save_final_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.gmm_save_final_checkpoint)
    
    opt.gmm_load_final_checkpoint_dir = opt.gmm_load_final_checkpoint_dir.format(root_opt.gmm_experiment_from_run, root_opt.gmm_experiment_from_dir)
    opt.gmm_load_final_checkpoint_dir = fix(opt.gmm_load_final_checkpoint_dir)
    opt.gmm_load_final_checkpoint = os.path.join(opt.gmm_load_final_checkpoint_dir, opt.gmm_load_final_checkpoint)
    opt.gmm_load_final_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.gmm_load_final_checkpoint)
    
    if not last_step:
        opt.gmm_load_step_checkpoint_dir = opt.gmm_load_step_checkpoint_dir.format(root_opt.gmm_experiment_from_run, root_opt.gmm_experiment_from_dir)
    else:
        opt.gmm_load_step_checkpoint_dir = opt.gmm_load_step_checkpoint_dir.format(root_opt.gmm_experiment_from_run, root_opt.this_viton_save_to_dir)
    opt.gmm_load_step_checkpoint_dir = fix(opt.gmm_load_step_checkpoint_dir)
    if not last_step:
        opt.gmm_load_step_checkpoint = os.path.join(opt.gmm_load_step_checkpoint_dir, opt.gmm_load_step_checkpoint)
    else:
        os_list = os.listdir(opt.gmm_load_step_checkpoint_dir.format(root_opt.gmm_experiment_from_run, root_opt.this_viton_save_to_dir))
        os_list = [string for string in os_list if "gmm" in string]
        last_step = sorted(os_list, key=sort_digit)[-1]
        opt.gmm_load_step_checkpoint = os.path.join(opt.gmm_load_step_checkpoint_dir, last_step)
    opt.gmm_load_step_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.gmm_load_step_checkpoint)
    # ================================= TOM =================================
    opt.tom_save_step_checkpoint_dir = opt.tom_save_step_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    opt.tom_save_step_checkpoint_dir = fix(opt.tom_save_step_checkpoint_dir)
    opt.tom_save_step_checkpoint = os.path.join(opt.tom_save_step_checkpoint_dir, opt.tom_save_step_checkpoint)
    opt.tom_save_step_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.tom_save_step_checkpoint)
    
    opt.tom_save_final_checkpoint_dir = opt.tom_save_final_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    opt.tom_save_final_checkpoint_dir = fix(opt.tom_save_final_checkpoint_dir)
    opt.tom_save_final_checkpoint = os.path.join(opt.tom_save_final_checkpoint_dir, opt.tom_save_final_checkpoint)
    opt.tom_save_final_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.tom_save_final_checkpoint)
    
    opt.tom_load_final_checkpoint_dir = opt.tom_load_final_checkpoint_dir.format(root_opt.gmm_experiment_from_run, root_opt.tom_experiment_from_dir)
    opt.tom_load_final_checkpoint_dir = fix(opt.tom_load_final_checkpoint_dir)
    opt.tom_load_final_checkpoint = os.path.join(opt.tom_load_final_checkpoint_dir, opt.tom_load_final_checkpoint)
    opt.tom_load_final_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.tom_load_final_checkpoint)
    
    if not last_step:
        opt.tom_load_step_checkpoint_dir = opt.tom_load_step_checkpoint_dir.format(root_opt.tom_experiment_from_run, root_opt.tom_experiment_from_dir)
    else:
        opt.tom_load_step_checkpoint_dir = opt.tom_load_step_checkpoint_dir.format(root_opt.tom_experiment_from_run, root_opt.this_viton_save_to_dir)
    opt.tom_load_step_checkpoint_dir = fix(opt.tom_load_step_checkpoint_dir)
    if not last_step:
        opt.tom_load_step_checkpoint = os.path.join(opt.tom_load_step_checkpoint_dir, opt.tom_load_step_checkpoint)
    else:
        if os.path.isdir(opt.tom_load_step_checkpoint_dir.format(root_opt.tom_experiment_from_run, root_opt.this_viton_save_to_dir)):
            os_list = os.listdir(opt.tom_load_step_checkpoint_dir.format(root_opt.tom_experiment_from_run, root_opt.this_viton_save_to_dir))
            os_list = [string for string in os_list if "tom" in string]
            last_step = sorted(os_list, key=sort_digit)[-1]
            opt.tom_load_step_checkpoint = os.path.join(opt.tom_load_step_checkpoint_dir, last_step)
    opt.tom_load_step_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.tom_load_step_checkpoint)
    return opt

def get_root_opt_results_dir(parser, root_opt):
    root_opt.transforms_dir = root_opt.transforms_dir.format(root_opt.dataset_name)
    parser.tensorboard_dir = parser.tensorboard_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    parser.results_dir = parser.results_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    return parser, root_opt

def copy_root_opt_to_opt(parser, root_opt):
    parser.display_count = root_opt.display_count
    parser.cuda = root_opt.cuda
    parser.dataset_name = root_opt.dataset_name
    parser.datamode = root_opt.datamode
    parser.load_last_step = root_opt.load_last_step if type(root_opt.load_last_step) == bool else eval(root_opt.load_last_step)
    parser.run_wandb = root_opt.run_wandb
    parser.stage = root_opt.stage
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



def test_cpvton_plus_(opt, root_opt):
    opt,root_opt = process_opt(opt, root_opt)
    print("Start to test %s!")
    _test_cpvton_plus_(opt, root_opt)

def _test_cpvton_plus_(opt, root_opt):
    print("Start to test stage: %s" % opt.stage)
   
    # create dataset 
    test_dataset = CPDataset(root_opt, opt)
    test_dataset.__getitem__(0)
    # create dataloader
    test_loader = CPDataLoader(opt, test_dataset)

    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(log_dir = os.path.join(opt.tensorboard_dir, opt.stage))

    # create model & test
    if opt.stage == 'GMM':
        model = GMM(opt)
        load_checkpoint(model, opt.gmm_load_final_checkpoint)
        with torch.no_grad():
            test_gmm(opt, test_loader, model, board)
    elif opt.stage == 'TOM':
        # model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)  # CP-VTON
        model = UnetGenerator(26, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)  # CP-VTON+
        gmm_model = GMM(opt)
        load_checkpoint(model, opt.tom_load_final_checkpoint)
        load_checkpoint(gmm_model, opt.gmm_load_final_checkpoint)
        with torch.no_grad():
            test_tom(opt, test_loader,gmm_model, model, board)
    else:
        raise NotImplementedError('Model [%s] is not implemented' % opt.stage)

    print('Finished test %s!' % opt.stage)


