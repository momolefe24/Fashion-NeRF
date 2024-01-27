import argparse
import os
import random
from pathlib import Path
from tensorboardX import SummaryWriter
# import debugpy
# debugpy.listen(5678)
# print("Ready")
# debugpy.wait_for_client()
# debugpy.breakpoint()
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import wandb
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

# from dataset.dresscode import DressCodeDataset
from VITON.Parser_Based.Ladi_VTON.src.dataset.dresscode import DressCodeDataset
from VITON.Parser_Based.Ladi_VTON.src.dataset.vitonhd import VitonHDDataset
from VITON.Parser_Based.Ladi_VTON.src.models.ConvNet_TPS import ConvNet_TPS
from VITON.Parser_Based.Ladi_VTON.src.models.UNet import UNetVanilla
from VITON.Parser_Based.Ladi_VTON.src.utils.vgg_loss import VGGLoss

fix = lambda path: os.path.normpath(path)

   
def get_wandb_image(image, wandb):
    if image.max() <= 1.0:
        image = image*255 
    image_numpy = image.permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
    return wandb.Image(image_numpy)     


def print_log(log_path, content, to_print=True):
    import os
    if os.path.exists(log_path):
        with open(log_path, 'a') as f:
            f.write(content)
            f.write('\n')

        if to_print:
            print(content)
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def compute_metric(dataloader: DataLoader, tps: ConvNet_TPS, criterion_l1: nn.L1Loss, criterion_vgg: VGGLoss,
                   refinement: UNetVanilla = None, height: int = 512, width: int = 384) -> tuple[
    float, float, list[list]]:
    """
    Perform inference on the given dataloader and compute the L1 and VGG loss between the warped cloth and the
    ground truth image.
    """
    tps.eval()
    if refinement:
        refinement.eval()

    running_loss = 0.
    vgg_running_loss = 0
    for step, inputs in enumerate(tqdm(dataloader)):
        cloth = inputs['cloth'].to(device)
        image = inputs['image'].to(device)
        im_cloth = inputs['im_cloth'].to(device)
        im_mask = inputs['im_mask'].to(device)
        pose_map = inputs.get('dense_uv')
        if pose_map is None:
            pose_map = inputs['pose_map']
        pose_map = pose_map.to(device)

        # TPS parameters prediction
        # For sake of performance, the TPS parameters are predicted on a low resolution image
        low_cloth = torchvision.transforms.functional.resize(cloth, (256, 192),
                                                             torchvision.transforms.InterpolationMode.BILINEAR,
                                                             antialias=True)
        low_im_mask = torchvision.transforms.functional.resize(im_mask, (256, 192),
                                                               torchvision.transforms.InterpolationMode.BILINEAR,
                                                               antialias=True)
        low_pose_map = torchvision.transforms.functional.resize(pose_map, (256, 192),
                                                                torchvision.transforms.InterpolationMode.BILINEAR,
                                                                antialias=True)

        # TPS parameters prediction
        agnostic = torch.cat([low_im_mask, low_pose_map], 1)

        low_grid, theta, rx, ry, cx, cy, rg, cg = tps(low_cloth, agnostic)

        # We upsample the grid to the original image size and warp the cloth using the predicted TPS parameters
        highres_grid = torchvision.transforms.functional.resize(low_grid.permute(0, 3, 1, 2),
                                                                size=(height, width),
                                                                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                                                                antialias=True).permute(0, 2, 3, 1)
        warped_cloth = F.grid_sample(cloth, highres_grid, padding_mode='border')

        if refinement:
            # Refine the warped cloth using the refinement network
            warped_cloth = torch.cat([im_mask, pose_map, warped_cloth], 1)
            warped_cloth = refinement(warped_cloth)

        # Compute the loss
        loss = criterion_l1(warped_cloth, im_cloth)
        running_loss += loss.item()
        if criterion_vgg:
            vgg_loss = criterion_vgg(warped_cloth, im_cloth)
            vgg_running_loss += vgg_loss.item()
        break
    visual = [[image, cloth, im_cloth, warped_cloth.clamp(-1, 1)]]
    loss = running_loss / (step + 1)
    vgg_loss = vgg_running_loss / (step + 1)
    return loss, vgg_loss, visual


def training_loop_tps(dataloader: DataLoader, tps: ConvNet_TPS, optimizer_tps: torch.optim.Optimizer,
                      criterion_l1: nn.L1Loss, scaler: torch.cuda.amp.GradScaler, const_weight: float) -> tuple[
    float, float, float, list[list]]:
    """
    Training loop for the TPS network. Note that the TPS is trained on a low resolution image for sake of performance.
    """
    tps.train()
    running_loss = 0.
    running_l1_loss = 0.
    running_const_loss = 0.
    for step, inputs in enumerate(tqdm(dataloader)):  # Yield images with low resolution (256x192)
        low_cloth = inputs['cloth'].to(device, non_blocking=True)
        low_image = inputs['image'].to(device, non_blocking=True)
        low_im_cloth = inputs['im_cloth'].to(device, non_blocking=True)
        low_im_mask = inputs['im_mask'].to(device, non_blocking=True)

        low_pose_map = inputs.get('dense_uv')
        if low_pose_map is None:  # If the dataset does not provide dense UV maps, use the pose map (keypoints) instead
            low_pose_map = inputs['pose_map']
        low_pose_map = low_pose_map.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            # TPS parameters prediction
            agnostic = torch.cat([low_im_mask, low_pose_map], 1)
            low_grid, theta, rx, ry, cx, cy, rg, cg = tps(low_cloth, agnostic)

            # Warp the cloth using the predicted TPS parameters
            low_warped_cloth = F.grid_sample(low_cloth, low_grid, padding_mode='border')

            # Compute the loss
            l1_loss = criterion_l1(low_warped_cloth, low_im_cloth)
            const_loss = torch.mean(rx + ry + cx + cy + rg + cg)

            loss = l1_loss + const_loss * const_weight

        # Update the parameters
        optimizer_tps.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer_tps)
        scaler.update()

        running_loss += loss.item()
        running_l1_loss += l1_loss.item()
        running_const_loss += const_loss.item()
        break

    visual = [[low_image, low_cloth, low_im_cloth, low_warped_cloth.clamp(-1, 1)]]
    loss = running_loss / (step + 1)
    l1_loss = running_l1_loss / (step + 1)
    const_loss = running_const_loss / (step + 1)
    return loss, l1_loss, const_loss, visual


def training_loop_refinement(dataloader: DataLoader, tps: ConvNet_TPS, refinement: UNetVanilla,
                             optimizer_ref: torch.optim.Optimizer, criterion_l1: nn.L1Loss, criterion_vgg: VGGLoss,
                             l1_weight: float, vgg_weight: float, scaler: torch.cuda.amp.GradScaler, height=512,
                             width=384) -> tuple[float, float, float, list[list]]:
    """
    Training loop for the refinement network. Note that the refinement network is trained on a high resolution image
    """
    tps.eval()
    refinement.train()
    running_loss = 0.
    running_l1_loss = 0.
    running_vgg_loss = 0.
    for step, inputs in enumerate(tqdm(dataloader)):
        cloth = inputs['cloth'].to(device)
        image = inputs['image'].to(device)
        im_cloth = inputs['im_cloth'].to(device)
        im_mask = inputs['im_mask'].to(device)

        pose_map = inputs.get('dense_uv')
        if pose_map is None:  # If the dataset does not provide dense UV maps, use the pose map (keypoints) instead
            pose_map = inputs['pose_map']
        pose_map = pose_map.to(device)

        # Resize the inputs to the low resolution for the TPS network
        low_cloth = torchvision.transforms.functional.resize(cloth, (256, 192),
                                                             torchvision.transforms.InterpolationMode.BILINEAR,
                                                             antialias=True)
        low_im_mask = torchvision.transforms.functional.resize(im_mask, (256, 192),
                                                               torchvision.transforms.InterpolationMode.BILINEAR,
                                                               antialias=True)
        low_pose_map = torchvision.transforms.functional.resize(pose_map, (256, 192),
                                                                torchvision.transforms.InterpolationMode.BILINEAR,
                                                                antialias=True)

        with torch.cuda.amp.autocast():
            # TPS parameters prediction
            agnostic = torch.cat([low_im_mask, low_pose_map], 1)

            low_grid, theta, rx, ry, cx, cy, rg, cg = tps(low_cloth, agnostic)
            low_warped_cloth = F.grid_sample(cloth, low_grid, padding_mode='border')

            # We upsample the grid to the original image size and warp the cloth using the predicted TPS parameters
            highres_grid = torchvision.transforms.functional.resize(low_grid.permute(0, 3, 1, 2),
                                                                    size=(height, width),
                                                                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                                                                    antialias=True).permute(0, 2, 3, 1)

            warped_cloth = F.grid_sample(cloth, highres_grid, padding_mode='border')

            # Refine the warped cloth using the refinement network
            warped_cloth = torch.cat([im_mask, pose_map, warped_cloth], 1)
            warped_cloth = refinement(warped_cloth)

            # Compute the loss
            l1_loss = criterion_l1(warped_cloth, im_cloth)
            vgg_loss = criterion_vgg(warped_cloth, im_cloth)

            loss = l1_loss * l1_weight + vgg_loss * vgg_weight
        break
        # Update the parameters
        optimizer_ref.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer_ref)
        scaler.update()

        running_loss += loss.item()
        running_l1_loss += l1_loss.item()
        running_vgg_loss += vgg_loss.item()

    visual = [[image, cloth, im_cloth, low_warped_cloth.clamp(-1, 1)]]
    loss = running_loss / (step + 1)
    l1_loss = running_l1_loss / (step + 1)
    vgg_loss = running_vgg_loss / (step + 1)
    return loss, l1_loss, vgg_loss, visual


@torch.no_grad()
def extract_images(dataloader: DataLoader, tps: ConvNet_TPS, refinement: UNetVanilla, save_path: str, height: int = 512,
                   width: int = 384) -> None:
    """
    Extracts the images using the trained networks and saves them to the save_path
    """
    tps.eval()
    refinement.eval()

    # running_loss = 0.
    for step, inputs in enumerate(tqdm(dataloader)):
        c_name = inputs['c_name']
        im_name = inputs['im_name']
        cloth = inputs['cloth'].to(device)
        category = inputs.get('category')
        im_mask = inputs['im_mask'].to(device)
        pose_map = inputs.get('dense_uv')
        if pose_map is None:
            pose_map = inputs['pose_map']
        pose_map = pose_map.to(device)

        # Resize the inputs to the low resolution for the TPS network
        low_cloth = torchvision.transforms.functional.resize(cloth, (256, 192),
                                                             torchvision.transforms.InterpolationMode.BILINEAR,
                                                             antialias=True)
        low_im_mask = torchvision.transforms.functional.resize(im_mask, (256, 192),
                                                               torchvision.transforms.InterpolationMode.BILINEAR,
                                                               antialias=True)
        low_pose_map = torchvision.transforms.functional.resize(pose_map, (256, 192),
                                                                torchvision.transforms.InterpolationMode.BILINEAR,
                                                                antialias=True)

        # TPS parameters prediction
        agnostic = torch.cat([low_im_mask, low_pose_map], 1)

        low_grid, theta, rx, ry, cx, cy, rg, cg = tps(low_cloth, agnostic)

        # We upsample the grid to the original image size and warp the cloth using the predicted TPS parameters
        highres_grid = torchvision.transforms.functional.resize(low_grid.permute(0, 3, 1, 2),
                                                                size=(height, width),
                                                                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                                                                antialias=True).permute(0, 2, 3, 1)

        warped_cloth = F.grid_sample(cloth, highres_grid, padding_mode='border')

        # Refine the warped cloth using the refinement network
        warped_cloth = torch.cat([im_mask, pose_map, warped_cloth], 1)
        warped_cloth = refinement(warped_cloth)

        warped_cloth = (warped_cloth + 1) / 2
        warped_cloth = warped_cloth.clamp(0, 1)

        # Save the images
        for cname, iname, warpclo, cat in zip(c_name, im_name, warped_cloth, category):
            if not os.path.exists(os.path.join(save_path, cat)):
                os.makedirs(os.path.join(save_path, cat))
            save_image(warpclo, os.path.join(save_path, cat, iname.replace(".jpg", "") + "_" + cname),
                       quality=95)
        break
def get_root_experiment_runs(root_opt):
    root_opt.experiment_run = root_opt.experiment_run.format(root_opt.experiment_number, root_opt.run_number)
    root_opt.experiment_from_run = root_opt.experiment_from_run.format(root_opt.experiment_from_number, root_opt.run_from_number)
    root_opt.tps_experiment_from_run = root_opt.tps_experiment_from_run.format(root_opt.tps_experiment_from_number, root_opt.tps_run_from_number)
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
    
    # TOCG
    root_opt.tps_experiment_from_dir = root_opt.tps_experiment_from_dir.format(root_opt.VITON_Type, root_opt.VITON_Name, root_opt.tps_load_from_model)
    root_opt.tps_experiment_from_dir = os.path.join(root_opt.tps_experiment_from_dir, root_opt.VITON_Model)
    
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
    # ================================= TOCG =================================
    opt.tps_save_step_checkpoint_dir = opt.tps_save_step_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    opt.tps_save_step_checkpoint_dir = fix(opt.tps_save_step_checkpoint_dir)
    opt.tps_save_step_checkpoint = os.path.join(opt.tps_save_step_checkpoint_dir, opt.tps_save_step_checkpoint)
    
    opt.tps_save_final_checkpoint_dir = opt.tps_save_final_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    opt.tps_save_final_checkpoint_dir = fix(opt.tps_save_final_checkpoint_dir)
    opt.tps_save_final_checkpoint = os.path.join(opt.tps_save_final_checkpoint_dir, opt.tps_save_final_checkpoint)
    
    opt.tps_load_final_checkpoint_dir = opt.tps_load_final_checkpoint_dir.format(root_opt.vgg_experiment_from_run, root_opt.tps_experiment_from_dir)
    opt.tps_load_final_checkpoint_dir = fix(opt.tps_load_final_checkpoint_dir)
    opt.tps_load_final_checkpoint = os.path.join(opt.tps_load_final_checkpoint_dir, opt.tps_load_final_checkpoint)

    if not last_step:
        opt.tps_load_step_checkpoint_dir = opt.tps_load_step_checkpoint_dir.format(root_opt.tps_experiment_from_run, root_opt.tps_experiment_from_dir)
    else:
        opt.tps_load_step_checkpoint_dir = opt.tps_load_step_checkpoint_dir.format(root_opt.tps_experiment_from_run, root_opt.this_viton_save_to_dir)
    opt.tps_load_step_checkpoint_dir = fix(opt.tps_load_step_checkpoint_dir)
    if not last_step:
        opt.tps_load_step_checkpoint = os.path.join(opt.tps_load_step_checkpoint_dir, opt.tps_load_step_checkpoint)
    else:
        if os.path.isdir(opt.tps_load_step_checkpoint_dir.format(root_opt.tps_experiment_from_run, root_opt.this_viton_save_to_dir)):
            os_list = os.listdir(opt.tps_load_step_checkpoint_dir.format(root_opt.tps_experiment_from_run, root_opt.this_viton_save_to_dir))
            os_list = [string for string in os_list if "tps" in string]
            last_step = sorted(os_list, key=sort_digit)[-1]
            opt.tps_load_step_checkpoint = os.path.join(opt.tps_load_step_checkpoint_dir, last_step)
        
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

def _train_ladi_vton_tps_(opt, root_opt,  wandb=None):

    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(log_dir = os.path.join(opt.tensorboard_dir, opt.VITON_Model))
    with open(os.path.join(root_opt.experiment_run_yaml, f"{root_opt.experiment_run.replace('/','_')}_{root_opt.opt_vton_yaml}"), 'w') as outfile:
        yaml.dump(vars(opt), outfile, default_flow_style=False)
    # Directories
    log_path = os.path.join(opt.results_dir, 'log.txt')
    if not os.path.exists(opt.results_dir):
        os.makedirs(opt.results_dir)
        with open(log_path, 'w') as file:
            file.write(f"Hello, this is experiment {root_opt.experiment_run} \n")

    dataset_output_list = ['c_name', 'im_name', 'cloth', 'image', 'im_cloth', 'im_mask', 'pose_map', 'category']
    if opt.dense:
        dataset_output_list.append('dense_uv')

    # Training dataset and dataloader
    dataroot = os.path.join(root_opt.root_dir, root_opt.original_dir)
    if opt.dataset == "vitonhd":
        dataset_train = VitonHDDataset(opt, root_opt, phase='train',
                                       outputlist=dataset_output_list,
                                       dataroot_path=dataroot,
                                       size=(opt.height, opt.width))
    elif opt.dataset == "dresscode":
        dataset_train = DressCodeDataset(dataroot_path=opt.dresscode_dataroot,
                                         phase='train',
                                         outputlist=dataset_output_list,
                                         size=(opt.height, opt.width))
    else:
        raise NotImplementedError("Dataset should be either vitonhd or dresscode")

    dataset_train.__getitem__(0)
    dataloader_train = DataLoader(batch_size=opt.viton_batch_size,
                                  dataset=dataset_train,
                                  shuffle=True,
                                  num_workers=opt.viton_workers)

    # Validation dataset and dataloader
    if opt.dataset == "vitonhd":
        dataset_test_paired = VitonHDDataset(opt, root_opt, phase='test',
                                             dataroot_path=dataroot,
                                             outputlist=dataset_output_list, size=(opt.height, opt.width))

        dataset_test_unpaired = VitonHDDataset(opt, root_opt, phase='test',
                                               order='unpaired',
                                               dataroot_path=dataroot,
                                               outputlist=dataset_output_list, size=(opt.height, opt.width))

    elif opt.dataset == "dresscode":
        dataset_test_paired = DressCodeDataset(dataroot_path=opt.dresscode_dataroot,
                                               phase='test',
                                               outputlist=dataset_output_list, size=(opt.height, opt.width))

        dataset_test_unpaired = DressCodeDataset(phase='test',
                                                 order='unpaired',
                                                 dataroot_path=opt.dresscode_dataroot,
                                                 outputlist=dataset_output_list, size=(opt.height, opt.width))

    else:
        raise NotImplementedError("Dataset should be either vitonhd or dresscode")

    dataloader_test_paired = DataLoader(batch_size=opt.viton_batch_size,
                                        dataset=dataset_test_paired,
                                        shuffle=True,
                                        num_workers=opt.viton_workers, drop_last=True)

    dataloader_test_unpaired = DataLoader(batch_size=opt.viton_batch_size,
                                          dataset=dataset_test_unpaired,
                                          shuffle=True,
                                          num_workers=opt.viton_workers, drop_last=True)

    # Define TPS and refinement network
    input_nc = 5 if opt.dense else 21
    n_layer = 3
    tps = ConvNet_TPS(256, 192, input_nc, n_layer).to(device)

    refinement = UNetVanilla(
        n_channels=8 if opt.dense else 24,
        n_classes=3,
        bilinear=True).to(device)

    # Define optimizer, scaler and loss
    optimizer_tps = torch.optim.Adam(tps.parameters(), lr=opt.lr, betas=(0.5, 0.99))
    optimizer_ref = torch.optim.Adam(list(refinement.parameters()), lr=opt.lr, betas=(0.5, 0.99))

    scaler = torch.cuda.amp.GradScaler()
    criterion_l1 = nn.L1Loss()

    if opt.vgg_weight > 0:
        criterion_vgg = VGGLoss().to(device)
    else:
        criterion_vgg = None

    start_epoch = 0
    last_step = root_opt.load_last_step if type(root_opt.load_last_step) == bool else eval(root_opt.load_last_step)
    
    if last_step:
        print('Loading full checkpoint')
        print_log(log_path, f'Load pretrained model from {opt.tps_load_step_checkpoint}')
        state_dict = torch.load(opt.tps_load_step_checkpoint)
        tps.load_state_dict(state_dict['tps'])
        refinement.load_state_dict(state_dict['refinement'])
        optimizer_tps.load_state_dict(state_dict['optimizer_tps'])
        optimizer_ref.load_state_dict(state_dict['optimizer_ref'])
        start_epoch = state_dict['epoch']
    elif os.path.exists(opt.tps_load_final_checkpoint):
        print('Loading full checkpoint')
        print_log(log_path, f'Load pretrained model from {opt.tps_load_final_checkpoint}')
        state_dict = torch.load(opt.tps_load_final_checkpoint)
        tps.load_state_dict(state_dict['tps'])
        refinement.load_state_dict(state_dict['refinement'])
        optimizer_tps.load_state_dict(state_dict['optimizer_tps'])
        optimizer_ref.load_state_dict(state_dict['optimizer_ref'])
        start_epoch = state_dict['epoch']

        if opt.only_extraction:
            print("Extracting warped cloth images...")
            extraction_dataset_paired = torch.utils.data.ConcatDataset([dataset_test_paired, dataset_train])
            extraction_dataloader_paired = DataLoader(batch_size=opt.viton_batch_size,
                                                      dataset=extraction_dataset_paired,
                                                      shuffle=False,
                                                      num_workers=opt.viton_workers,
                                                      drop_last=False)

            
            warped_cloth_root = opt.results_dir

            # save_name_paired = warped_cloth_root / 'warped_cloths' / opt.dataset
            save_name_paired = os.path.join(warped_cloth_root, 'warped_cloths', opt.dataset)

            extract_images(extraction_dataloader_paired, tps, refinement, save_name_paired, opt.height, opt.width)

            extraction_dataset = dataset_test_unpaired
            extraction_dataloader_paired = DataLoader(batch_size=opt.viton_batch_size,
                                                      dataset=extraction_dataset,
                                                      shuffle=False,
                                                      num_workers=opt.viton_workers)

            save_name_unpaired = os.path.join(warped_cloth_root, 'warped_cloths_unpaired', opt.dataset)
            extract_images(extraction_dataloader_paired, tps, refinement, save_name_unpaired, opt.height, opt.width)
            exit()

    if opt.only_extraction and not os.path.exists(
            opt.tps_load_final_checkpoint):
        print("No checkpoint found, before extracting warped cloth images, please train the model first.")
        exit()

    # Training loop for TPS training
    # Set training dataset height and width to (256, 192) since the TPS is trained using a lower resolution
    dataset_train.height = 256
    dataset_train.width = 192
    # for e in range(start_epoch, opt.epochs_tps):
    for e in range(start_epoch, opt.epochs_tps):
        print(f"Epoch {e}/{opt.epochs_tps}")
        print('train')
        train_loss, train_l1_loss, train_const_loss, visual = training_loop_tps(
            dataloader_train,
            tps,
            optimizer_tps,
            criterion_l1,
            scaler,
            opt.const_weight)

        # Compute loss on paired test set
        print('paired test')
        running_loss, vgg_running_loss, visual = compute_metric(
            dataloader_test_paired,
            tps,
            criterion_l1,
            criterion_vgg,
            refinement=None,
            height=opt.height,
            width=opt.width)

        imgs = torchvision.utils.make_grid(torch.cat(visual[0]), nrow=len(visual[0][0]), padding=2, normalize=True,
                                           range=None, scale_each=False, pad_value=0)

        # Compute loss on unpaired test set
        print('unpaired test')
        running_loss_unpaired, vgg_running_loss_unpaired, visual = compute_metric(
            dataloader_test_unpaired,
            tps,
            criterion_l1,
            criterion_vgg,
            refinement=None,
            height=opt.height,
            width=opt.width)

        imgs_unpaired = torchvision.utils.make_grid(torch.cat(visual[0]), nrow=len(visual[0][0]), padding=2,
                                                    normalize=True, range=None,
                                                    scale_each=False, pad_value=0)

        # Log to wandb
        if wandb is not None:
            wandb.log({
                'train/loss': train_loss,
                'train/l1_loss': train_l1_loss,
                'train/const_loss': train_const_loss,
                'train/vgg_loss': 0,
                'eval/eval_loss_paired': running_loss,
                'eval/eval_vgg_loss_paired': vgg_running_loss,
                'eval/eval_loss_unpaired': running_loss_unpaired,
                'eval/eval_vgg_loss_unpaired': vgg_running_loss_unpaired,
                'images_paired': wandb.Image(imgs),
                'images_unpaired': wandb.Image(imgs_unpaired),
            })

        # Save checkpoint
        os.makedirs(opt.tps_save_step_checkpoint_dir, exist_ok=True)
        torch.save({
            'epoch': e + 1,
            'tps': tps.state_dict(),
            'refinement': refinement.state_dict(),
            'optimizer_tps': optimizer_tps.state_dict(),
            'optimizer_ref': optimizer_ref.state_dict(),
        }, opt.tps_save_step_checkpoint % (e + 1))
        break
    scaler = torch.cuda.amp.GradScaler()  # Initialize scaler again for refinement

    # Training loop for refinement
    # Set training dataset height and width to (opt.height, opt.width) since the refinement is trained using a higher resolution
    dataset_train.height = opt.height
    dataset_train.width = opt.width
    for e in range(max(start_epoch, opt.epochs_tps), max(start_epoch, opt.epochs_tps) + opt.epochs_refinement):
        print(f"Epoch {e}/{max(start_epoch, opt.epochs_tps) + opt.epochs_refinement}")
        train_loss, train_l1_loss, train_vgg_loss, visual = training_loop_refinement(
            dataloader_train,
            tps,
            refinement,
            optimizer_ref,
            criterion_l1,
            criterion_vgg,
            opt.l1_weight,
            opt.vgg_weight,
            scaler,
            opt.height,
            opt.width)

        # Compute loss on paired test set
        running_loss, vgg_running_loss, visual = compute_metric(
            dataloader_test_paired,
            tps,
            criterion_l1,
            criterion_vgg,
            refinement=refinement,
            height=opt.height,
            width=opt.width)

        imgs = torchvision.utils.make_grid(torch.cat(visual[0]), nrow=len(visual[0][0]), padding=2, normalize=True,
                                           range=None, scale_each=False, pad_value=0)

        # Compute loss on unpaired test set
        running_loss_unpaired, vgg_running_loss_unpaired, visual = compute_metric(
            dataloader_test_unpaired,
            tps,
            criterion_l1,
            criterion_vgg,
            refinement=refinement,
            height=opt.height,
            width=opt.width)

        imgs_unpaired = torchvision.utils.make_grid(torch.cat(visual[0]), nrow=len(visual[0][0]), padding=2,
                                                    normalize=True, range=None,
                                                    scale_each=False, pad_value=0)

        # Log to wandb
        if wandb is not None:
            wandb.log({
                'train/loss': train_loss,
                'train/l1_loss': train_l1_loss,
                'train/const_loss': 0,
                'train/vgg_loss': train_vgg_loss,
                'eval/eval_loss_paired': running_loss,
                'eval/eval_vgg_loss_paired': vgg_running_loss,
                'eval/eval_loss_unpaired': running_loss_unpaired,
                'eval/eval_vgg_loss_unpaired': vgg_running_loss_unpaired,
                'images_paired': wandb.Image(imgs),
                'images_unpaired': wandb.Image(imgs_unpaired),
            })
        # break
        # Save checkpoint
        os.makedirs(opt.tps_save_step_checkpoint_dir, exist_ok=True)
        
        torch.save({
            'epoch': e + 1,
            'tps': tps.state_dict(),
            'refinement': refinement.state_dict(),
            'optimizer_tps': optimizer_tps.state_dict(),
            'optimizer_ref': optimizer_ref.state_dict(),
        }, opt.tps_save_step_checkpoint % (e + 1))
        break
    # Extract warped cloth images at the end of training
    print("Extracting warped cloth images...")
    extraction_dataset_paired = torch.utils.data.ConcatDataset([dataset_test_paired, dataset_train])
    extraction_dataloader_paired = DataLoader(batch_size=opt.viton_batch_size,
                                              dataset=extraction_dataset_paired,
                                              shuffle=False,
                                              num_workers=opt.viton_workers,
                                              drop_last=False)

    warped_cloth_root = opt.results_dir

    # save_name_paired = warped_cloth_root / 'warped_cloths' / opt.dataset
    save_name_paired = os.path.join(warped_cloth_root, 'warped_cloths', opt.dataset)
    extract_images(extraction_dataloader_paired, tps, refinement, save_name_paired, opt.height, opt.width)

    extraction_dataset = dataset_test_unpaired
    extraction_dataloader_paired = DataLoader(batch_size=opt.viton_batch_size,
                                              dataset=extraction_dataset,
                                              shuffle=False,
                                              num_workers=opt.viton_workers)

    # save_name_unpaired = warped_cloth_root / 'warped_cloths_unpaired' / opt.dataset
    save_name_unpaired = os.path.join(warped_cloth_root, 'warped_cloths_unpaired', opt.dataset)
    extract_images(extraction_dataloader_paired, tps, refinement, save_name_unpaired, opt.height, opt.width)
    os.makedirs(opt.tps_save_final_checkpoint_dir, exist_ok=True)
    torch.save({
        'epoch': e + 1,
        'tps': tps.state_dict(),
        'refinement': refinement.state_dict(),
        'optimizer_tps': optimizer_tps.state_dict(),
        'optimizer_ref': optimizer_ref.state_dict(),
    }, opt.tps_save_final_checkpoint)

import subprocess
def train_ladi_vton_tps_(opt, root_opt, run_wandb=False):
    # os.system('conda activate ladi-vton && conda env list | grep ladi')
    command = "source ~/.bashrc && conda activate ladi-vton && conda env list | grep ladi"
    subprocess.run(command, shell=True, executable='/bin/bash')
    opt,root_opt = process_opt(opt, root_opt)
    if run_wandb:
        import wandb
        wandb.login()
        wandb.init(
        project="Fashion-NeRF",
        entity='prime_lab',
        notes=f"{root_opt.question}",
        tags=[f"{root_opt.experiment_run}"],
        config=vars(opt)
        )
    else:
        wandb = None
    if wandb is not None:
        temp_opt = vars(opt)
        temp_opt['wandb_name'] = wandb.run.name
        opt = argparse.Namespace(**temp_opt)
    _train_ladi_vton_tps_(opt, root_opt,  wandb=wandb)
