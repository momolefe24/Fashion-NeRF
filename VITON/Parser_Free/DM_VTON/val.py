import shutil
from pathlib import Path
import argparse
import cupy
import torch
import torch.nn.functional as F
import torchvision as tv
from torch.utils.data import DataLoader
from tqdm import tqdm

from VITON.Parser_Free.DM_VTON.dataloader.viton_dataset import LoadVITONDataset
from VITON.Parser_Free.DM_VTON.models.generators.mobile_unet import MobileNetV2_unet
from VITON.Parser_Free.DM_VTON.models.warp_modules.mobile_afwm import MobileAFWM as AFWM
from VITON.Parser_Free.DM_VTON.opt.test_opt import TestOptions
from VITON.Parser_Free.DM_VTON.utils.general import print_log
from VITON.Parser_Free.DM_VTON.utils.metrics import calculate_fid_given_paths
from VITON.Parser_Free.DM_VTON.utils.torch_utils import get_ckpt, load_ckpt, select_device
fix = lambda path: os.path.normpath(path)
import os 

def run_val_pf(
    opt, root_opt, data_loader, models, align_corners, device, img_dir, save_dir, log_path, save_img=True
):
    warp_model, gen_model = models['warp'], models['gen']
    metrics = {}


    result_dir = save_dir
    tryon_dir = os.path.join(result_dir, 'try_on')
    visualize_dir = os.path.join(result_dir, 'visualize')
    if not os.path.exists(tryon_dir):
        os.makedirs(tryon_dir)
        os.makedirs(visualize_dir)
    

    # testidate
    with torch.no_grad():
        # seen, dt = 0, (Profile(device=device), Profile(device=device), Profile(device=device))

        for idx, data in enumerate(tqdm(data_loader)):
            # Prepare data
            # with dt[0]:
            real_image = data['image'].to(device)
            clothes = data['color'].to(device)
            edge = data['edge'].to(device)
            edge = (edge > 0.5).float()
            clothes = clothes * edge

            # Warp
            # with dt[1]:
            with cupy.cuda.Device(int(device.split(':')[-1])):
                flow_out = warp_model(
                    real_image, clothes, edge
                )  # edge is only for parameter replacement during train, does not work in val
                (
                    warped_cloth,
                    last_flow,
                    cond_fea_all,
                    warp_fea_all,
                    flow_all,
                    delta_list,
                    x_all,
                    x_edge_all,
                    delta_x_all,
                    delta_y_all,
                ) = flow_out
                warped_edge = F.grid_sample(
                    edge,
                    last_flow.permute(0, 2, 3, 1),
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=align_corners,
                )
                if root_opt.dataset_name == 'Rail':
                    binary_mask = (warped_edge > 0.5).float()
                    warped_cloth = warped_cloth * binary_mask

            # Gen
            # with dt[2]:
            gen_inputs = torch.cat([real_image, warped_cloth, warped_edge], 1)
            gen_outputs = gen_model(gen_inputs)
            p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
            p_rendered = torch.tanh(p_rendered)
            m_composite = torch.sigmoid(m_composite)
            m_composite = m_composite * warped_edge
            p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

            # seen += len(p_tryon)

            # Save images
            for j in range(len(data['p_name'])):
                p_name = data['p_name'][j]

                tv.utils.save_image(
                    p_tryon[j],
                    os.path.join(tryon_dir,p_name),
                    nrow=int(1),
                    normalize=True,
                    value_range=(-1, 1),
                )

                combine = torch.cat(
                    [real_image[j].float(), clothes[j], warped_cloth[j], p_tryon[j]], -1
                ).squeeze()
                tv.utils.save_image(
                    combine,
                    os.path.join(visualize_dir,p_name),
                    nrow=int(1),
                    normalize=True,
                    value_range=(-1, 1),
                )
    if not os.path.exists(str(img_dir)):
        os.makedirs(str(img_dir))
    fid = calculate_fid_given_paths(
        paths=[str(img_dir), str(tryon_dir)],
        batch_size=50,
        device=device,
    )

    if not save_img:
        shutil.rmtree(result_dir)

    # FID
    metrics['fid'] = fid

    # Log
    metrics_str = 'Metric, {}'.format(', '.join([f'{k}: {v}' for k, v in metrics.items()]))
    print_log(log_path, metrics_str)

    return metrics


def _visualize_(opt, root_opt):
    # Device
    device = select_device(opt.device, batch_size=root_opt.viton_batch_size)
    log_path = os.path.join(opt.results_dir, 'log.txt')

    # Model
    warp_model = AFWM(3, opt.align_corners).to(device)
    warp_model.eval()
    if os.path.exists(opt.pf_warp_load_final_checkpoint):
        warp_ckpt = get_ckpt(opt.pf_warp_load_final_checkpoint)
        load_ckpt(warp_model, warp_ckpt)
        print_log(log_path, f'Load pretrained parser-free warp from {opt.pf_warp_checkpoint}')
        
        
    gen_model = MobileNetV2_unet(7, 4).to(device)
    gen_model.eval()
    if os.path.exists(opt.pf_gen_load_final_checkpoint):
        gen_ckpt = get_ckpt(opt.pf_gen_load_final_checkpoint)
        load_ckpt(gen_model, gen_ckpt)
        print_log(log_path, f'Load pretrained parser-free gen from {opt.pf_gen_load_final_checkpoint}')
    
    img_dir = os.path.join(root_opt.root_dir, root_opt.original_dir, 'test_img')
    # Dataloader
    test_data = LoadVITONDataset(path=opt.dataroot, phase='test', size=(256, 192))
    data_loader = DataLoader(
        test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers
    )

    run_val_pf(
        opt, root_opt, 
        data_loader=data_loader,
        models={'warp': warp_model, 'gen': gen_model},
        align_corners=opt.align_corners,
        device=device,
        log_path=log_path,
        save_dir=opt.results_dir,
        img_dir=img_dir,
        save_img=True,
    )



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

def validate_(opt, root_opt):
    opt,root_opt = process_opt(opt, root_opt)
    _visualize_(opt, root_opt)