from pathlib import Path
import argparse
import os
import cupy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torch.utils.data import DataLoader

from VITON.Parser_Free.DM_VTON.dataloader.viton_dataset import LoadVITONDataset
from VITON.Parser_Free.DM_VTON.models.generators.mobile_unet import MobileNetV2_unet
from VITON.Parser_Free.DM_VTON.models.generators.res_unet import ResUnetGenerator
from VITON.Parser_Free.DM_VTON.models.warp_modules.mobile_afwm import MobileAFWM as AFWM
from VITON.Parser_Free.DM_VTON.models.warp_modules.style_afwm import StyleAFWM as PBAFWM
from VITON.Parser_Free.DM_VTON.utils.general import print_log
from VITON.Parser_Free.DM_VTON.utils.torch_utils import get_ckpt, load_ckpt, select_device
fix = lambda path: os.path.normpath(path)

def get_palette(num_cls: int) -> list[int]:
    """
    Returns the color map for visualizing the segmentation mask.

    Args:
      num_cls: Number of classes or categories for which a color palette is needed.

    Returns:
      RGB color map.

    To use this palette: PIL.Image.putpalette(get_palette(num_cls))
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            palette[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            palette[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            i += 1
            lab >>= 3
    return palette


def visualize_train_process(opt, data, models, device):
    pb_warp_model, pb_gen_model, pf_warp_model, pf_gen_model = (
        models['pb_warp'],
        models['pb_gen'],
        models['pf_warp'],
        models['pf_gen'],
    )

    t_mask = torch.FloatTensor((data['label'].cpu().numpy() == 7).astype(np.float64))
    data['label'] = data['label'] * (1 - t_mask) + t_mask * 4
    edge = data['edge']
    pre_clothes_edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int64))
    clothes = data['color']
    clothes = clothes * pre_clothes_edge
    edge_un = data['edge_un']
    pre_clothes_edge_un = torch.FloatTensor((edge_un.detach().numpy() > 0.5).astype(np.int64))
    clothes_un = data['color_un']
    clothes_un = clothes_un * pre_clothes_edge_un
    real_image = data['image']
    pose = data['pose']
    size = data['label'].size()
    oneHot_size1 = (size[0], 25, size[2], size[3])
    densepose = torch.cuda.FloatTensor(torch.Size(oneHot_size1), device=device).zero_()
    densepose = densepose.scatter_(1, data['densepose'].data.long().to(device), 1.0)
    face_mask = torch.FloatTensor(
        (data['label'].cpu().numpy() == 1).astype(np.int64)
    ) + torch.FloatTensor((data['label'].cpu().numpy() == 12).astype(np.int64))
    other_clothes_mask = (
        torch.FloatTensor((data['label'].cpu().numpy() == 5).astype(np.int64))
        + torch.FloatTensor((data['label'].cpu().numpy() == 6).astype(np.int64))
        + torch.FloatTensor((data['label'].cpu().numpy() == 8).astype(np.int64))
        + torch.FloatTensor((data['label'].cpu().numpy() == 9).astype(np.int64))
        + torch.FloatTensor((data['label'].cpu().numpy() == 10).astype(np.int64))
    )
    face_img = face_mask * real_image
    other_clothes_img = other_clothes_mask * real_image
    preserve_mask = torch.cat([face_mask, other_clothes_mask], 1)

    concat_un = torch.cat([preserve_mask.to(device), densepose, pose.to(device)], 1)
    with cupy.cuda.Device(int(device.split(':')[-1])):
        flow_out_un = pb_warp_model(
            concat_un.to(device), clothes_un.to(device), pre_clothes_edge_un.to(device)
        )
    (
        warped_cloth_un,
        last_flow_un,
        cond_un_all,
        flow_un_all,
        delta_list_un,
        x_all_un,
        x_edge_all_un,
        delta_x_all_un,
        delta_y_all_un,
    ) = flow_out_un
    warped_prod_edge_un = F.grid_sample(
        pre_clothes_edge_un.to(device),
        last_flow_un.permute(0, 2, 3, 1),
        mode='bilinear',
        padding_mode='zeros',
        align_corners=opt.align_corners,
    )

    arm_mask = torch.FloatTensor(
        (data['label'].cpu().numpy() == 11).astype(np.float64)
    ) + torch.FloatTensor((data['label'].cpu().numpy() == 13).astype(np.float64))
    hand_mask = torch.FloatTensor(
        (data['densepose'].cpu().numpy() == 3).astype(np.int64)
    ) + torch.FloatTensor((data['densepose'].cpu().numpy() == 4).astype(np.int64))
    dense_preserve_mask = (
        torch.FloatTensor((data['densepose'].cpu().numpy() == 15).astype(np.int64))
        + torch.FloatTensor((data['densepose'].cpu().numpy() == 16).astype(np.int64))
        + torch.FloatTensor((data['densepose'].cpu().numpy() == 17).astype(np.int64))
        + torch.FloatTensor((data['densepose'].cpu().numpy() == 18).astype(np.int64))
        + torch.FloatTensor((data['densepose'].cpu().numpy() == 19).astype(np.int64))
        + torch.FloatTensor((data['densepose'].cpu().numpy() == 20).astype(np.int64))
        + torch.FloatTensor((data['densepose'].cpu().numpy() == 21).astype(np.int64))
        + torch.FloatTensor((data['densepose'].cpu().numpy() == 22).astype(np.int64))
    )
    hand_img = (arm_mask * hand_mask) * real_image
    dense_preserve_mask = dense_preserve_mask.to(device) * (1 - warped_prod_edge_un)
    preserve_region = face_img + other_clothes_img + hand_img

    gen_inputs_un = torch.cat(
        [preserve_region.to(device), warped_cloth_un, warped_prod_edge_un, dense_preserve_mask], 1
    )
    gen_outputs_un = pb_gen_model(gen_inputs_un)
    p_rendered_un, m_composite_un = torch.split(gen_outputs_un, [3, 1], 1)
    p_rendered_un = torch.tanh(p_rendered_un)
    m_composite_un = torch.sigmoid(m_composite_un)
    m_composite_un = m_composite_un * warped_prod_edge_un
    p_tryon_un = warped_cloth_un * m_composite_un + p_rendered_un * (1 - m_composite_un)

    with cupy.cuda.Device(int(device.split(':')[-1])):
        flow_out = pf_warp_model(
            p_tryon_un.detach(), clothes.to(device), pre_clothes_edge.to(device)
        )
    (
        warped_cloth,
        last_flow,
        cond_all,
        flow_all,
        delta_list,
        x_all,
        x_edge_all,
        delta_x_all,
        delta_y_all,
    ) = flow_out
    warped_prod_edge = x_edge_all[4]

    gen_inputs = torch.cat([p_tryon_un.detach(), warped_cloth, warped_prod_edge], 1)
    gen_outputs = pf_gen_model(gen_inputs)

    p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
    p_rendered = torch.tanh(p_rendered)
    m_composite = torch.sigmoid(m_composite)
    m_composite = m_composite * warped_prod_edge
    p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

    premap, _ = torch.max(preserve_mask.squeeze(), 0)
    premap = premap * 2 - 1

    denseposemap = torch.argmax(densepose.squeeze(), dim=0) / densepose.shape[1] * 2 - 1

    posemap, _ = torch.max(pose.squeeze(), 0)
    human_map = torch.cat([premap.to(device), denseposemap.to(device), posemap.to(device)], 1)

    dense_preserve = (
        torch.cat([dense_preserve_mask, dense_preserve_mask, dense_preserve_mask], 1) * 2 - 1
    )
    preserve_map = torch.cat([preserve_region[0].to(device), dense_preserve[0].to(device)], 2)

    return (
        human_map,
        preserve_map,
        real_image[0],
        clothes_un[0],
        warped_cloth_un[0],
        p_tryon_un[0],
        clothes[0],
        warped_cloth[0],
        p_tryon[0],
    )


def _visualize_(opt, root_opt):
    # Device
    device = select_device(opt.device, batch_size=root_opt.viton_batch_size)

    # Directories
    log_path = os.path.join(opt.results_dir, 'log.txt')
    samples_dir = os.path.join(opt.results_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)

    # Device
    device = select_device(opt.device, batch_size=root_opt.viton_batch_size)

    # Model
    pb_warp_model = PBAFWM(45, opt.align_corners).to(device)
    pb_warp_model.eval()
    if os.path.exists(opt.pb_warp_load_final_checkpoint):
        pb_warp_ckpt = get_ckpt(opt.pb_warp_load_final_checkpoint)
        load_ckpt(pb_warp_model, pb_warp_ckpt)
        print_log(log_path, f'Load pretrained parser-based warp from {opt.pb_warp_load_final_checkpoint}')
        
    pb_gen_model = ResUnetGenerator(8, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d).to(device)
    pb_gen_model.eval()
    
    if os.path.exists(opt.pb_gen_load_final_checkpoint):
        pb_gen_ckpt = get_ckpt(opt.pb_gen_load_final_checkpoint)
        load_ckpt(pb_gen_model, pb_gen_ckpt)
        print_log(log_path, f'Load pretrained parser-based gen from {opt.pb_gen_load_final_checkpoint}')
        
    pf_warp_model = AFWM(3, opt.align_corners).to(device)
    pf_warp_model.eval()
    
    if os.path.exists(opt.pf_warp_load_final_checkpoint):
        pf_warp_ckpt = get_ckpt(opt.pf_warp_load_final_checkpoint)
        load_ckpt(pf_warp_model, pf_warp_ckpt)
        print_log(log_path, f'Load pretrained parser-free warp from {opt.pf_warp_load_final_checkpoint}')
        
    pf_gen_model = MobileNetV2_unet(7, 4).to(device)
    pf_gen_model.eval()
    if os.path.exists(opt.pf_gen_load_final_checkpoint):
        pf_gen_ckpt = get_ckpt(opt.pf_gen_load_final_checkpoint)
        load_ckpt(pf_gen_model, pf_gen_ckpt)
    print_log(log_path, f'Load pretrained parser-free gen from {opt.pf_gen_load_final_checkpoint}')

    # Dataset
    train_data = LoadVITONDataset(root_opt, phase='train', size=(256, 192))
    train_loader = DataLoader(
        train_data, batch_size=opt.viton_batch_size, shuffle=True, num_workers=root_opt.workers
    )

    for data in train_loader:
        (
            human_map,
            preserve_map,
            real_image,
            clothes_un,
            warped_cloth_un,
            p_tryon_un,
            clothes,
            warped_cloth,
            p_tryon,
        ) = visualize_train_process(
            opt,
            data,
            models={
                'pb_warp': pb_warp_model,
                'pb_gen': pb_gen_model,
                'pf_warp': pf_warp_model,
                'pf_gen': pf_gen_model,
            },
            device=device,
        )

        tv.utils.save_image(
            human_map,
            os.path.join(samples_dir,'human_map.jpg'),
            nrow=int(1),
            normalize=True,
            value_range=(-1, 1),
        )
        tv.utils.save_image(
            preserve_map,
            os.path.join(samples_dir,'preserve_map.jpg'),
            nrow=int(1),
            normalize=True,
            value_range=(-1, 1),
        )
        tv.utils.save_image(
            real_image,
            os.path.join(samples_dir, 'real_image.jpg'),
            nrow=int(1),
            normalize=True,
            value_range=(-1, 1),
        )
        tv.utils.save_image(
            clothes_un,
            os.path.join(samples_dir,'clothes_un.jpg'),
            nrow=int(1),
            normalize=True,
            value_range=(-1, 1),
        )
        tv.utils.save_image(
            warped_cloth_un,
            os.path.join(samples_dir, 'warped_cloth_un.jpg'),
            nrow=int(1),
            normalize=True,
            value_range=(-1, 1),
        )
        tv.utils.save_image(
            p_tryon_un,
            os.path.join(samples_dir, 'p_tryon_un.jpg'),
            nrow=int(1),
            normalize=True,
            value_range=(-1, 1),
        )
        tv.utils.save_image(
            clothes, os.path.join(samples_dir,'clothes.jpg'), nrow=int(1), normalize=True, value_range=(-1, 1)
        )
        tv.utils.save_image(
            warped_cloth,
            os.path.join(samples_dir ,'warped_cloth.jpg'),
            nrow=int(1),
            normalize=True,
            value_range=(-1, 1),
        )
        tv.utils.save_image(
            p_tryon,os.path.join(samples_dir, 'p_tryon.jpg'), nrow=int(1), normalize=True, value_range=(-1, 1)
        )

        break

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

    parser.pb_warp_load_step_checkpoint_dir = parser.pb_warp_load_step_checkpoint_dir.format(root_opt.parser_based_warp_experiment_from_run, root_opt.parser_based_warp_experiment_from_dir)
    parser.pb_warp_load_step_checkpoint = os.path.join(parser.pb_warp_load_step_checkpoint_dir, parser.pb_warp_load_step_checkpoint)
    parser.pb_warp_load_step_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pb_warp_load_step_checkpoint)

    parser.pb_warp_save_final_checkpoint_dir = parser.pb_warp_save_final_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    parser.pb_warp_save_final_checkpoint = os.path.join(parser.pb_warp_save_final_checkpoint_dir, parser.pb_warp_save_final_checkpoint)
    parser.pb_warp_save_final_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pb_warp_save_final_checkpoint)

    parser.pb_warp_load_final_checkpoint_dir = parser.pb_warp_load_final_checkpoint_dir.format(root_opt.parser_based_warp_experiment_from_run, root_opt.parser_based_warp_experiment_from_dir)
    parser.pb_warp_load_final_checkpoint = os.path.join(parser.pb_warp_load_final_checkpoint_dir, parser.pb_warp_load_final_checkpoint)
    parser.pb_warp_load_final_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pb_warp_load_final_checkpoint)
    
    # Parser Based Gen
    parser.pb_gen_save_step_checkpoint_dir = parser.pb_gen_save_step_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    parser.pb_gen_save_step_checkpoint = os.path.join(parser.pb_gen_save_step_checkpoint_dir, parser.pb_gen_save_step_checkpoint)
    parser.pb_gen_save_step_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pb_gen_save_step_checkpoint)
    
    

    parser.pb_gen_load_step_checkpoint_dir = parser.pb_gen_load_step_checkpoint_dir.format(root_opt.parser_based_gen_experiment_from_run, root_opt.parser_based_gen_experiment_from_dir)
    parser.pb_gen_load_step_checkpoint = os.path.join(parser.pb_gen_load_step_checkpoint_dir, parser.pb_gen_load_step_checkpoint)
    parser.pb_gen_load_step_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pb_gen_load_step_checkpoint)
    

    parser.pb_gen_save_final_checkpoint_dir = parser.pb_gen_save_final_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    parser.pb_gen_save_final_checkpoint = os.path.join(parser.pb_gen_save_final_checkpoint_dir, parser.pb_gen_save_final_checkpoint)
    parser.pb_gen_save_final_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pb_gen_save_final_checkpoint)

    parser.pb_gen_load_final_checkpoint_dir = parser.pb_gen_load_final_checkpoint_dir.format(root_opt.parser_based_gen_experiment_from_run, root_opt.parser_based_gen_experiment_from_dir)
    parser.pb_gen_load_final_checkpoint = os.path.join(parser.pb_gen_load_final_checkpoint_dir, parser.pb_gen_load_final_checkpoint)
    parser.pb_gen_load_final_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pb_gen_load_final_checkpoint)
    
    # Parser Free Warping
    parser.pf_warp_save_step_checkpoint_dir = parser.pf_warp_save_step_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    parser.pf_warp_save_step_checkpoint = os.path.join(parser.pf_warp_save_step_checkpoint_dir, parser.pf_warp_save_step_checkpoint)
    parser.pf_warp_save_step_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pf_warp_save_step_checkpoint)

    parser.pf_warp_load_step_checkpoint_dir = parser.pf_warp_load_step_checkpoint_dir.format(root_opt.parser_free_warp_experiment_from_run, root_opt.parser_free_warp_experiment_from_dir)
    parser.pf_warp_load_step_checkpoint = os.path.join(parser.pf_warp_load_step_checkpoint_dir, parser.pf_warp_load_step_checkpoint)
    parser.pf_warp_load_step_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pf_warp_load_step_checkpoint)

    parser.pf_warp_save_final_checkpoint_dir = parser.pf_warp_save_final_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    parser.pf_warp_save_final_checkpoint = os.path.join(parser.pf_warp_save_final_checkpoint_dir, parser.pf_warp_save_final_checkpoint)
    parser.pf_warp_save_final_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pf_warp_save_final_checkpoint)

    parser.pf_warp_load_final_checkpoint_dir = parser.pf_warp_load_final_checkpoint_dir.format(root_opt.parser_free_warp_experiment_from_run, root_opt.parser_free_warp_experiment_from_dir)
    parser.pf_warp_load_final_checkpoint = os.path.join(parser.pf_warp_load_final_checkpoint_dir, parser.pf_warp_load_final_checkpoint)
    parser.pf_warp_load_final_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pf_warp_load_final_checkpoint)
    
    # Parser Free Gen
    parser.pf_gen_save_step_checkpoint_dir = parser.pf_gen_save_step_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    parser.pf_gen_save_step_checkpoint = os.path.join(parser.pf_gen_save_step_checkpoint_dir, parser.pf_gen_save_step_checkpoint)
    parser.pf_gen_save_step_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pf_gen_save_step_checkpoint)
    
    # parser.pf_gen_load_step_checkpoint_dir = parser.pf_gen_load_step_checkpoint_dir.format(root_opt.parser_free_gen_experiment_from_run, root_opt.parser_free_gen_experiment_from_dir)
    # parser.pf_gen_load_step_checkpoint = os.path.join(parser.pf_gen_load_step_checkpoint_dir, parser.pf_gen_load_step_checkpoint)
    
    parser.pf_gen_save_final_checkpoint_dir = parser.pf_gen_save_final_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    parser.pf_gen_save_final_checkpoint = os.path.join(parser.pf_gen_save_final_checkpoint_dir, parser.pf_gen_save_final_checkpoint)
    parser.pf_gen_save_final_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pf_gen_save_final_checkpoint)
    
    parser.pf_gen_load_final_checkpoint_dir = parser.pf_gen_load_final_checkpoint_dir.format(root_opt.parser_free_gen_experiment_from_run, root_opt.parser_free_gen_experiment_from_dir)
    parser.pf_gen_load_final_checkpoint = os.path.join(parser.pf_gen_load_final_checkpoint_dir, parser.pf_gen_load_final_checkpoint)
    parser.pf_gen_load_final_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pf_gen_load_final_checkpoint)
    
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
    return parser

def visualize_(opt, root_opt):
    opt,root_opt = process_opt(opt, root_opt)
    _visualize_(opt, root_opt)

