import torch
import torch.nn as nn
import yaml
from torchvision.utils import make_grid
from VITON.Parser_Based.HR_VITON.networks import make_grid as mkgrid
from VITON.Parser_Based.HR_VITON.sync_batchnorm import DataParallelWithCallback
from preprocessing.segment_anything.segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor  
import argparse
import os
import time
from dataset import FashionDataLoader, FashionNeRFDataset
from VITON.Parser_Based.HR_VITON.networks import ConditionGenerator, VGGLoss, GANLoss, load_checkpoint, save_checkpoint, define_D
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tensorboardX import SummaryWriter
from VITON.Parser_Based.HR_VITON.utils import *
from torch.utils.data import Subset
import matplotlib.pyplot as plt
from torchvision.utils import save_image
fix = lambda path: os.path.normpath(path)
opt,root_opt,wandb,sweep_id =None, None, None,None
   
def get_wandb_image(image, wandb):
    if image.max() <= 1.0:
        image = image*255 
    image_numpy = image.permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
    return wandb.Image(image_numpy)     
        
def get_root_experiment_runs(root_opt):
    root_opt.experiment_run = root_opt.experiment_run.format(root_opt.experiment_number, root_opt.run_number)
    root_opt.experiment_from_run = root_opt.experiment_from_run.format(root_opt.experiment_from_number, root_opt.run_from_number)
    root_opt.tocg_experiment_from_run = root_opt.tocg_experiment_from_run.format(root_opt.tocg_experiment_from_number, root_opt.tocg_run_from_number)
    root_opt.tocg_discriminator_experiment_from_run = root_opt.tocg_discriminator_experiment_from_run.format(root_opt.tocg_discriminator_experiment_from_number, root_opt.tocg_discriminator_run_from_number)
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
    root_opt.tocg_experiment_from_dir = root_opt.tocg_experiment_from_dir.format(root_opt.VITON_Type, root_opt.VITON_Name, root_opt.tocg_load_from_model)
    root_opt.tocg_experiment_from_dir = os.path.join(root_opt.tocg_experiment_from_dir, root_opt.VITON_Model)
    
    # TOCG discriminator
    root_opt.tocg_discriminator_experiment_from_dir = root_opt.tocg_discriminator_experiment_from_dir.format(root_opt.VITON_Type, root_opt.VITON_Name, root_opt.tocg_discriminator_load_from_model)
    root_opt.tocg_discriminator_experiment_from_dir = os.path.join(root_opt.tocg_discriminator_experiment_from_dir, root_opt.VITON_Model)    
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
    parser.val_count = root_opt.val_count
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
    if last_step:
        opt.tocg_load_step_checkpoint_dir = opt.tocg_load_step_checkpoint_dir.format(root_opt.tocg_experiment_from_run, root_opt.tocg_experiment_from_dir)
    else:
        opt.tocg_load_step_checkpoint_dir = opt.tocg_load_step_checkpoint_dir.format(root_opt.tocg_experiment_from_run, root_opt.this_viton_save_to_dir)
    opt.tocg_load_step_checkpoint_dir = fix(opt.tocg_load_step_checkpoint_dir)
    if not last_step:
        opt.tocg_load_step_checkpoint = os.path.join(opt.tocg_load_step_checkpoint_dir, opt.tocg_load_step_checkpoint)
    else:
        if os.path.isdir(opt.tocg_load_step_checkpoint_dir):
            os_list = os.listdir(opt.tocg_load_step_checkpoint_dir.format(root_opt.tocg_experiment_from_run, root_opt.this_viton_save_to_dir))
            os_list = [string for string in os_list if "tocg" in string and "discriminator" not in string]
            last_step = sorted(os_list, key=sort_digit)[-1]
            opt.tocg_load_step_checkpoint = os.path.join(opt.tocg_load_step_checkpoint_dir, last_step)
    opt.tocg_load_step_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.tocg_load_step_checkpoint)
    opt.tocg_load_step_checkpoint_dir = os.path.join("/",*opt.tocg_load_step_checkpoint.split("/")[:-1])
    # ================================= TOCG DISCRIMINATOR =================================
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
    
    opt.tocg_discriminator_load_final_checkpoint_dir = opt.tocg_discriminator_load_final_checkpoint_dir.format(root_opt.tocg_discriminator_experiment_from_run, root_opt.tocg_discriminator_experiment_from_dir)
    opt.tocg_discriminator_load_final_checkpoint_dir = fix(opt.tocg_discriminator_load_final_checkpoint_dir)
    opt.tocg_discriminator_load_final_checkpoint = os.path.join(opt.tocg_discriminator_load_final_checkpoint_dir, opt.tocg_discriminator_load_final_checkpoint)
    opt.tocg_discriminator_load_final_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.tocg_discriminator_load_final_checkpoint)
    opt.tocg_discriminator_load_final_checkpoint_dir = os.path.join("/",*opt.tocg_discriminator_load_final_checkpoint.split("/")[:-1])
    
    if last_step:
        opt.tocg_discriminator_load_step_checkpoint_dir = opt.tocg_discriminator_load_step_checkpoint_dir.format(root_opt.tocg_discriminator_experiment_from_run, root_opt.tocg_discriminator_experiment_from_dir)
    else:
        opt.tocg_discriminator_load_step_checkpoint_dir = opt.tocg_discriminator_load_step_checkpoint_dir.format(root_opt.tocg_discriminator_experiment_from_run, root_opt.this_viton_save_to_dir)
    opt.tocg_discriminator_load_step_checkpoint_dir = fix(opt.tocg_discriminator_load_step_checkpoint_dir)
    if not last_step:
        opt.tocg_discriminator_load_step_checkpoint = os.path.join(opt.tocg_discriminator_load_step_checkpoint_dir, opt.tocg_discriminator_load_step_checkpoint)
    else:
        if os.path.isdir(opt.tocg_discriminator_load_step_checkpoint_dir):
            os_list = os.listdir(opt.tocg_discriminator_load_step_checkpoint_dir.format(root_opt.tocg_discriminator_experiment_from_run, root_opt.this_viton_save_to_dir))
            os_list = [string for string in os_list if "tocg_discriminator" in string]
            last_step = sorted(os_list, key=sort_digit)[-1]
            opt.tocg_discriminator_load_step_checkpoint = os.path.join(opt.tocg_discriminator_load_step_checkpoint_dir, last_step)
    opt.tocg_discriminator_load_step_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.tocg_discriminator_load_step_checkpoint)
    opt.tocg_discriminator_load_step_checkpoint_dir = os.path.join("/",*opt.tocg_discriminator_load_step_checkpoint.split("/")[:-1])
    return opt

def get_root_opt_results_dir(parser, root_opt):
    root_opt.transforms_dir = root_opt.transforms_dir.format(root_opt.dataset_name)
    parser.tensorboard_dir = parser.tensorboard_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    parser.results_dir = parser.results_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    return parser, root_opt

def process_opt(opt, root_opt):
    parser = opt
    parser = argparse.Namespace(**parser)
    root_opt = get_root_experiment_runs(root_opt)
    root_opt = get_root_opt_experiment_dir(root_opt)
    parser = get_root_opt_checkpoint_dir(parser, root_opt)
    parser, root_opt = get_root_opt_results_dir(parser, root_opt)    
    parser = copy_root_opt_to_opt(parser, root_opt)
    return parser, root_opt


to = lambda inputs,name,i: inputs[name][i].permute(1,2,0).cpu().numpy()
to2 = lambda inputs,name1,name2, i: inputs[name1][name2][i].permute(1,2,0).cpu().numpy()
to3 = lambda inputs,i: inputs[i].permute(1,2,0).detach().cpu().numpy()

def iou_metric(y_pred_batch, y_true_batch):
    B = y_pred_batch.shape[0]
    iou = 0
    for i in range(B):
        y_pred = y_pred_batch[i]
        y_true = y_true_batch[i]
        # y_pred is not one-hot, so need to threshold it
        y_pred = y_pred > 0.5
        
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()

    
        intersection = torch.sum(y_pred[y_true == 1])
        union = torch.sum(y_pred) + torch.sum(y_true)

    
        iou += (intersection + 1e-7) / (union - intersection + 1e-7) / B
    return iou

def remove_overlap(seg_out, warped_cm):
    
    assert len(warped_cm.shape) == 4
    
    warped_cm = warped_cm - (torch.cat([seg_out[:, 1:3, :, :], seg_out[:, 5:, :, :]], dim=1)).sum(dim=1, keepdim=True) * warped_cm
    return warped_cm



# train_model(opt, root_opt, train_loader,test_loader, validation_loader, board, tocg, D, wandb)
def train_model(opt,root_opt, train_loader, test_loader, validation_loader, board, tocg, D, wandb=None):
    tocg.cuda()
    D.cuda()
    tocg.train()
    D.train()

    # criterion
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss(opt)
    if opt.fp16:
        criterionGAN = GANLoss(use_lsgan=True, tensor=torch.cuda.HalfTensor)
    else :
        if root_opt.cuda:
            criterionGAN = GANLoss(use_lsgan=True, tensor=torch.cuda.FloatTensor)
        else:
            criterionGAN = GANLoss(use_lsgan=True, tensor=torch.Tensor)
    
    # optimizer
    optimizer_G = torch.optim.Adam(tocg.parameters(), lr=opt.G_lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=opt.D_lr, betas=(0.5, 0.999))
    
    tocg = DataParallelWithCallback(tocg, device_ids=[opt.device])
    D = DataParallelWithCallback(D, device_ids=[opt.device])

    for step in tqdm(range(opt.niter + opt.niter_decay)):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        # input1
        c_paired = inputs['cloth']['paired'].cuda()
        cm_paired = inputs['cloth_mask']['paired'].cuda()
        # cm_paired = torch.FloatTensor((cm_paired.detach().cpu().numpy() > 0.5).astype(np.float32)).cuda()
        # input2
        parse_agnostic = inputs['parse_agnostic'].cuda()
        densepose = inputs['densepose'].cuda()
        openpose = inputs['pose'].cuda()
        # GT
        label_onehot = inputs['parse_onehot'].cuda()  # CE
        label = inputs['parse'].cuda()  # GAN loss
        parse_cloth_mask = inputs['pcm'].cuda()  # L1
        im_c = inputs['parse_cloth'].cuda()  # VGG
        # visualization
        im = inputs['image']

        # inputs
        if len(cm_paired.size()) == 5:
            cm_paired = cm_paired.squeeze(1)
            cm_paired = cm_paired.permute(0,3,1,2)
        input1 = torch.cat([c_paired, cm_paired], 1)
        input2 = torch.cat([parse_agnostic, densepose], 1)

        # forward
        if opt.segment_anything and step >= (opt.niter - opt.niter * 0.2):
            flow_list, fake_segmap, warped_cloth_paired, warped_clothmask_paired = tocg(opt, input1, input2, im_c=im_c)
        else:
            flow_list, fake_segmap, warped_cloth_paired, warped_clothmask_paired = tocg(opt, input1, input2)
        
        warped_cm_onehot = torch.FloatTensor((warped_clothmask_paired.detach().cpu().numpy() > 0.5).astype(np.float32)).cuda()
        # fake segmap cloth channel * warped clothmask
        if opt.clothmask_composition != 'no_composition':
            if opt.clothmask_composition == 'detach':
                cloth_mask = torch.ones_like(fake_segmap.detach())
                cloth_mask[:, 3:4, :, :] = warped_cm_onehot
                fake_segmap = fake_segmap * cloth_mask
                
            if opt.clothmask_composition == 'warp_grad':
                cloth_mask = torch.ones_like(fake_segmap.detach())
                cloth_mask[:, 3:4, :, :] = warped_clothmask_paired
                fake_segmap = fake_segmap * cloth_mask
        if opt.occlusion:
            warped_clothmask_paired = remove_overlap(F.softmax(fake_segmap, dim=1), warped_clothmask_paired)
            warped_cloth_paired = warped_cloth_paired * warped_clothmask_paired + torch.ones_like(warped_cloth_paired) * (1-warped_clothmask_paired)

        # generated fake cloth mask & misalign mask
        fake_clothmask = (torch.argmax(fake_segmap.detach(), dim=1, keepdim=True) == 3).long()
        misalign = fake_clothmask - warped_cm_onehot
        misalign[misalign < 0.0] = 0.0
        
        # loss warping
        loss_l1_cloth = criterionL1(warped_clothmask_paired, parse_cloth_mask)
        loss_vgg = criterionVGG(warped_cloth_paired, im_c)

        loss_tv = 0
        
        if opt.edgeawaretv == 'no_edge':
            if not opt.lasttvonly:
                for flow in flow_list:
                    y_tv = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :]).mean()
                    x_tv = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean()
                    loss_tv = loss_tv + y_tv + x_tv
            else:
                for flow in flow_list[-1:]:
                    y_tv = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :]).mean()
                    x_tv = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean()
                    loss_tv = loss_tv + y_tv + x_tv
        else:
            if opt.edgeawaretv == 'last_only':
                flow = flow_list[-1]
                warped_clothmask_paired_down = F.interpolate(warped_clothmask_paired, flow.shape[1:3], mode='bilinear')
                y_tv = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :])
                x_tv = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
                mask_y = torch.exp(-150*torch.abs(warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, 1:, :, :] - warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, :-1, :, :]))
                mask_x = torch.exp(-150*torch.abs(warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, :, 1:, :] - warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, :, :-1, :]))
                y_tv = y_tv * mask_y
                x_tv = x_tv * mask_x
                y_tv = y_tv.mean()
                x_tv = x_tv.mean()
                loss_tv = loss_tv + y_tv + x_tv
                
            elif opt.edgeawaretv == 'weighted':
                for i in range(5):
                    flow = flow_list[i]
                    warped_clothmask_paired_down = F.interpolate(warped_clothmask_paired, flow.shape[1:3], mode='bilinear')
                    y_tv = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :])
                    x_tv = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
                    mask_y = torch.exp(-150*torch.abs(warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, 1:, :, :] - warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, :-1, :, :]))
                    mask_x = torch.exp(-150*torch.abs(warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, :, 1:, :] - warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, :, :-1, :]))
                    y_tv = y_tv * mask_y
                    x_tv = x_tv * mask_x
                    y_tv = y_tv.mean() / (2 ** (4-i))
                    x_tv = x_tv.mean() / (2 ** (4-i))
                    loss_tv = loss_tv + y_tv + x_tv

            if opt.tocg_add_lasttv:
                for flow in flow_list[-1:]:
                    y_tv = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :]).mean()
                    x_tv = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean()
                    loss_tv = loss_tv + y_tv + x_tv
            

        N, _, iH, iW = c_paired.size()
        # Intermediate flow loss
        if opt.interflowloss:
            for i in range(len(flow_list)-1):
                flow = flow_list[i]
                N, fH, fW, _ = flow.size()
                grid = mkgrid(N, iH, iW, opt)
                flow = F.interpolate(flow.permute(0, 3, 1, 2), size = c_paired.shape[2:], mode=opt.upsample).permute(0, 2, 3, 1)
                flow_norm = torch.cat([flow[:, :, :, 0:1] / ((fW - 1.0) / 2.0), flow[:, :, :, 1:2] / ((fH - 1.0) / 2.0)], 3)
                warped_c = F.grid_sample(c_paired, flow_norm + grid, padding_mode='border')
                warped_cm = F.grid_sample(cm_paired, flow_norm + grid, padding_mode='border')
                warped_cm = remove_overlap(F.softmax(fake_segmap, dim=1), warped_cm)
                loss_l1_cloth += criterionL1(warped_cm, parse_cloth_mask) / (2 ** (4-i))
                loss_vgg += criterionVGG(warped_c, im_c) / (2 ** (4-i))

        # loss segmentation
        # generator
        CE_loss = cross_entropy2d(fake_segmap, label_onehot.transpose(0, 1)[0].long())
        
        if opt.no_GAN_loss:
            loss_G = (10 * loss_l1_cloth + loss_vgg + opt.tvlambda * loss_tv) + (CE_loss * opt.CElamda)
            # step
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()
        
        else:
            fake_segmap_softmax = torch.softmax(fake_segmap, 1)

            pred_segmap = D(torch.cat((input1.detach(), input2.detach(), fake_segmap_softmax), dim=1))
            
            loss_G_GAN = criterionGAN(pred_segmap, True)
            
            if not opt.G_D_seperate:  
                # discriminator
                fake_segmap_pred = D(torch.cat((input1.detach(), input2.detach(), fake_segmap_softmax.detach()),dim=1))
                real_segmap_pred = D(torch.cat((input1.detach(), input2.detach(), label),dim=1))
                loss_D_fake = criterionGAN(fake_segmap_pred, False)
                loss_D_real = criterionGAN(real_segmap_pred, True)

                # loss sum
                loss_G = (opt.loss_l1_cloth_lambda * loss_l1_cloth + loss_vgg +opt.tvlambda * loss_tv) + (CE_loss * opt.CElamda + loss_G_GAN * opt.GANlambda)  # warping + seg_generation
                loss_D = loss_D_fake + loss_D_real

                # step
                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()
                
                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()
                
            else: # train G first after that train D
                # loss G sum
                loss_G = (opt.loss_l1_cloth_lambda * loss_l1_cloth + loss_vgg + opt.tvlambda * loss_tv) + (CE_loss * opt.CElamda + loss_G_GAN * opt.GANlambda)  # warping + seg_generation
                
                # step G
                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()
                
                # discriminator
                with torch.no_grad():
                    _, fake_segmap, _, _ = tocg(opt, input1, input2)
                fake_segmap_softmax = torch.softmax(fake_segmap, 1)
                
                # loss discriminator
                fake_segmap_pred = D(torch.cat((input1.detach(), input2.detach(), fake_segmap_softmax.detach()),dim=1))
                real_segmap_pred = D(torch.cat((input1.detach(), input2.detach(), label),dim=1))
                loss_D_fake = criterionGAN(fake_segmap_pred, False)
                loss_D_real = criterionGAN(real_segmap_pred, True)
                
                loss_D = loss_D_fake + loss_D_real
                
                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()
        # Vaildation
        if (step + 1) % opt.val_count == 0:
            validate_tocg(opt, step, tocg,D, validation_loader,board,wandb)
            tocg.train()
        
        # tensorboard
        if (step + 1) % opt.display_count == 0:
            # loss G
            board.add_scalar('warping_loss', loss_G.item(), step + 1)
            board.add_scalar('l1_cloth', loss_l1_cloth.item(), step + 1)
            board.add_scalar('vgg', loss_vgg.item(), step + 1)
            board.add_scalar('total_variation_loss', loss_tv.item(), step + 1)
            board.add_scalar('cross_entropy_loss', CE_loss.item(), step + 1)
# Wandb     
            if not opt.no_GAN_loss:
                board.add_scalar('gan', loss_G_GAN.item(), step + 1)
                # loss D
                board.add_scalar('discriminator', loss_D.item(), step + 1)
                board.add_scalar('pred_real', loss_D_real.item(), step + 1)
                board.add_scalar('pred_fake', loss_D_fake.item(), step + 1)
            save_image(warped_cloth_paired, os.path.join(opt.results_dir, f'warped_cloth_paired_{step}.png'))
            grid = make_grid([(c_paired[0].cpu() / 2 + 0.5), (cm_paired[0].cpu()).expand(3, -1, -1), visualize_segmap(parse_agnostic.cpu()), ((densepose.cpu()[0]+1)/2),
                              (im_c[0].cpu() / 2 + 0.5), parse_cloth_mask[0].cpu().expand(3, -1, -1), (warped_cloth_paired[0].cpu().detach() / 2 + 0.5), (warped_cm_onehot[0].cpu().detach()).expand(3, -1, -1),
                              visualize_segmap(label.cpu()), visualize_segmap(fake_segmap.cpu()), (im[0]/2 +0.5), (misalign[0].cpu().detach()).expand(3, -1, -1)],
                                nrow=4)
            board.add_images('train_images', grid.unsqueeze(0), step + 1)
            if wandb is not None:
                my_table = wandb.Table(columns=['Grid', 'Segmentation Map For Label','Segmentation Map For Fake Data','Warped Cloth','Misalign'])
                grid_wandb = get_wandb_image(grid, wandb)
                segmap_for_label = get_wandb_image(visualize_segmap(label.cpu()), wandb)
                segmap_for_fake_data = get_wandb_image(visualize_segmap(fake_segmap.cpu()), wandb)
                warped_cloth = get_wandb_image((warped_cloth_paired[0].cpu().detach() / 2 + 0.5), wandb)
                misalign_cloth = get_wandb_image((misalign[0].cpu().detach()).expand(3, -1, -1), wandb)
                my_table.add_data(grid_wandb, segmap_for_label,segmap_for_fake_data,warped_cloth,misalign_cloth)
                wandb.log({'Table': my_table, 'warping_loss': loss_G.item()
                ,'l1_cloth':loss_l1_cloth.item()
                ,'vgg':loss_vgg.item()
                ,'total_variation_loss':loss_tv.item()
                ,'cross_entropy_loss':CE_loss.item()})
            
            inputs = test_loader.next_batch()
            c_paired = inputs['cloth'][opt.test_datasetting].cuda()
            cm_paired = inputs['cloth_mask'][opt.test_datasetting].cuda()
            cm_paired = torch.FloatTensor((cm_paired.detach().cpu().numpy() > 0.5).astype(np.float32)).cuda()
            # input2
            parse_agnostic = inputs['parse_agnostic'].cuda()
            densepose = inputs['densepose'].cuda()
            openpose = inputs['pose'].cuda()
            # GT
            label_onehot = inputs['parse_onehot'].cuda()  # CE
            label = inputs['parse'].cuda()  # GAN loss
            parse_cloth_mask = inputs['pcm'].cuda()  # L1
            im_c = inputs['parse_cloth'].cuda()  # VGG
            # visualization
            im = inputs['image']

            tocg.eval()
            with torch.no_grad():
                # inputs
                input1 = torch.cat([c_paired, cm_paired], 1)
                input2 = torch.cat([parse_agnostic, densepose], 1)

                # forward
                flow_list, fake_segmap, warped_cloth_paired, warped_clothmask_paired = tocg(opt, input1, input2)
                warped_cm_onehot = torch.FloatTensor(
                    (warped_clothmask_paired.detach().cpu().numpy() > 0.5).astype(np.float32)).cuda()
                if opt.clothmask_composition != 'no_composition':
                    if opt.clothmask_composition == 'detach':
                        cloth_mask = torch.ones_like(fake_segmap)
                        cloth_mask[:,3:4, :, :] = warped_cm_onehot
                        fake_segmap = fake_segmap * cloth_mask
                        
                    if opt.clothmask_composition == 'warp_grad':
                        cloth_mask = torch.ones_like(fake_segmap)
                        cloth_mask[:,3:4, :, :] = warped_clothmask_paired
                        fake_segmap = fake_segmap * cloth_mask
                if opt.occlusion:
                    warped_clothmask_paired = remove_overlap(F.softmax(fake_segmap, dim=1), warped_clothmask_paired)
                    warped_cloth_paired = warped_cloth_paired * warped_clothmask_paired + torch.ones_like(warped_cloth_paired) * (1-warped_clothmask_paired)
                
                # generated fake cloth mask & misalign mask
                fake_clothmask = (torch.argmax(fake_segmap.detach(), dim=1, keepdim=True) == 3).long()
                misalign = fake_clothmask - warped_cm_onehot
                misalign[misalign < 0.0] = 0.0
                
                for i in range(opt.num_test_visualize):
                    grid = make_grid([(c_paired[i].cpu() / 2 + 0.5), (cm_paired[i].cpu()).expand(3, -1, -1), visualize_segmap(parse_agnostic.cpu(), batch=i), ((densepose.cpu()[i]+1)/2),
                                    (im_c[i].cpu() / 2 + 0.5), parse_cloth_mask[i].cpu().expand(3, -1, -1), (warped_cloth_paired[i].cpu().detach() / 2 + 0.5), (warped_cm_onehot[i].cpu().detach()).expand(3, -1, -1),
                                    visualize_segmap(label.cpu(), batch=i), visualize_segmap(fake_segmap.cpu(), batch=i), (im[i]/2 +0.5), (misalign[i].cpu().detach()).expand(3, -1, -1)],
                                        nrow=4)
                    board.add_images(f'test_images/{i}', grid.unsqueeze(0), step + 1)
                tocg.train()
        
        # display
        if (step + 1) % opt.save_period == 0:
            t = time.time() - iter_start_time
            if not opt.no_GAN_loss:
                print("step: %8d, time: %.3f\nloss G: %.4f, L1_cloth loss: %.4f, VGG loss: %.4f, TV loss: %.4f CE: %.4f, G GAN: %.4f\nloss D: %.4f, D real: %.4f, D fake: %.4f"
                    % (step + 1, t, loss_G.item(), loss_l1_cloth.item(), loss_vgg.item(), loss_tv.item(), CE_loss.item(), loss_G_GAN.item(), loss_D.item(), loss_D_real.item(), loss_D_fake.item()), flush=True)
            if not os.path.exists(opt.tocg_save_step_checkpoint_dir):
                os.makedirs(opt.tocg_save_step_checkpoint_dir)
            save_checkpoint(tocg,opt.tocg_save_step_checkpoint % (step + 1), opt)
            if not os.path.exists(opt.tocg_discriminator_save_step_checkpoint_dir):
                os.makedirs(opt.tocg_discriminator_save_step_checkpoint_dir)
            save_checkpoint(D,opt.tocg_discriminator_save_step_checkpoint % (step + 1), opt)
        # break


def validate_tocg(opt, step, tocg,D, validation_loader,board,wandb):
    tocg.eval()
    # criterion
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss(opt)
    if opt.fp16:
        criterionGAN = GANLoss(use_lsgan=True, tensor=torch.cuda.HalfTensor)
    else :
        if root_opt.cuda:
            criterionGAN = GANLoss(use_lsgan=True, tensor=torch.cuda.FloatTensor)
        else:
            criterionGAN = GANLoss(use_lsgan=True, tensor=torch.Tensor)
    
    tocg = DataParallelWithCallback(tocg, device_ids=[opt.device])
    D = DataParallelWithCallback(D, device_ids=[opt.device])
    iou_list = []
    with torch.no_grad():
        for cnt in range(20//opt.viton_batch_size):
            inputs = validation_loader.next_batch()
            if root_opt.cuda:
                # input1
                c_paired = inputs['cloth']['paired'].cuda()
                cm_paired = inputs['cloth_mask']['paired'].cuda()
                # cm_paired = torch.FloatTensor((cm_paired.detach().cpu().numpy() > 0.5).astype(np.float32)).cuda()
                # input2
                parse_agnostic = inputs['parse_agnostic'].cuda()
                densepose = inputs['densepose'].cuda()
                openpose = inputs['pose'].cuda()
                # GT
                label_onehot = inputs['parse_onehot'].cuda()  # CE
                label = inputs['parse'].cuda()  # GAN loss
                parse_cloth_mask = inputs['pcm'].cuda()  # L1
                im_c = inputs['parse_cloth'].cuda()  # VGG
            else:
                c_paired = inputs['cloth']['paired']
                cm_paired = inputs['cloth_mask']['paired']
                # cm_paired = torch.FloatTensor(
                #     (cm_paired.detach().cpu().numpy() > 0.5).astype(np.float32))
                # input2
                parse_agnostic = inputs['parse_agnostic']
                densepose = inputs['densepose']
                openpose = inputs['pose']
                # GT
                label_onehot = inputs['parse_onehot']  # CE
                label = inputs['parse']  # GAN loss
                parse_cloth_mask = inputs['pcm']  # L1
                im_c = inputs['parse_cloth']  # VGG
            # visualization
            im = inputs['image']
            
        # inputs
        if len(cm_paired.size()) == 5:
            cm_paired = cm_paired.squeeze(1)
            cm_paired = cm_paired.permute(0,3,1,2)
        input1 = torch.cat([c_paired, cm_paired], 1)
        input2 = torch.cat([parse_agnostic, densepose], 1)

        # forward
        if opt.segment_anything and step >= (opt.niter + opt.niter_decay) // 3:
            flow_list, fake_segmap, warped_cloth_paired, warped_clothmask_paired = tocg(opt, input1, input2, im_c=im_c)
        else:
            flow_list, fake_segmap, warped_cloth_paired, warped_clothmask_paired = tocg(opt, input1, input2)
        
        warped_cm_onehot = torch.FloatTensor((warped_clothmask_paired.detach().cpu().numpy() > 0.5).astype(np.float32)).cuda()
        # fake segmap cloth channel * warped clothmask
        if opt.clothmask_composition != 'no_composition':
            if opt.clothmask_composition == 'detach':
                cloth_mask = torch.ones_like(fake_segmap.detach())
                cloth_mask[:, 3:4, :, :] = warped_cm_onehot
                fake_segmap = fake_segmap * cloth_mask
                
            if opt.clothmask_composition == 'warp_grad':
                cloth_mask = torch.ones_like(fake_segmap.detach())
                cloth_mask[:, 3:4, :, :] = warped_clothmask_paired
                fake_segmap = fake_segmap * cloth_mask
        
        if opt.occlusion:
            warped_clothmask_paired = remove_overlap(F.softmax(fake_segmap, dim=1), warped_clothmask_paired)
            warped_cloth_paired = warped_cloth_paired * warped_clothmask_paired + torch.ones_like(warped_cloth_paired) * (1-warped_clothmask_paired)

        # generated fake cloth mask & misalign mask
        fake_clothmask = (torch.argmax(fake_segmap.detach(), dim=1, keepdim=True) == 3).long()
        misalign = fake_clothmask - warped_cm_onehot
        misalign[misalign < 0.0] = 0.0
        
        # loss warping
        loss_l1_cloth = criterionL1(warped_clothmask_paired, parse_cloth_mask)
        loss_vgg = criterionVGG(warped_cloth_paired, im_c)

        loss_tv = 0
        
        if opt.edgeawaretv == 'no_edge':
            if not opt.lasttvonly:
                for flow in flow_list:
                    y_tv = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :]).mean()
                    x_tv = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean()
                    loss_tv = loss_tv + y_tv + x_tv
            else:
                for flow in flow_list[-1:]:
                    y_tv = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :]).mean()
                    x_tv = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean()
                    loss_tv = loss_tv + y_tv + x_tv
        else:
            if opt.edgeawaretv == 'last_only':
                flow = flow_list[-1]
                warped_clothmask_paired_down = F.interpolate(warped_clothmask_paired, flow.shape[1:3], mode='bilinear')
                y_tv = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :])
                x_tv = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
                mask_y = torch.exp(-150*torch.abs(warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, 1:, :, :] - warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, :-1, :, :]))
                mask_x = torch.exp(-150*torch.abs(warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, :, 1:, :] - warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, :, :-1, :]))
                y_tv = y_tv * mask_y
                x_tv = x_tv * mask_x
                y_tv = y_tv.mean()
                x_tv = x_tv.mean()
                loss_tv = loss_tv + y_tv + x_tv
                
            elif opt.edgeawaretv == 'weighted':
                for i in range(5):
                    flow = flow_list[i]
                    warped_clothmask_paired_down = F.interpolate(warped_clothmask_paired, flow.shape[1:3], mode='bilinear')
                    y_tv = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :])
                    x_tv = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
                    mask_y = torch.exp(-150*torch.abs(warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, 1:, :, :] - warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, :-1, :, :]))
                    mask_x = torch.exp(-150*torch.abs(warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, :, 1:, :] - warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, :, :-1, :]))
                    y_tv = y_tv * mask_y
                    x_tv = x_tv * mask_x
                    y_tv = y_tv.mean() / (2 ** (4-i))
                    x_tv = x_tv.mean() / (2 ** (4-i))
                    loss_tv = loss_tv + y_tv + x_tv

            if opt.tocg_add_lasttv:
                for flow in flow_list[-1:]:
                    y_tv = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :]).mean()
                    x_tv = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean()
                    loss_tv = loss_tv + y_tv + x_tv
            

        N, _, iH, iW = c_paired.size()
        # Intermediate flow loss
        if opt.interflowloss:
            for i in range(len(flow_list)-1):
                flow = flow_list[i]
                N, fH, fW, _ = flow.size()
                grid = mkgrid(N, iH, iW, opt)
                flow = F.interpolate(flow.permute(0, 3, 1, 2), size = c_paired.shape[2:], mode=opt.upsample).permute(0, 2, 3, 1)
                flow_norm = torch.cat([flow[:, :, :, 0:1] / ((fW - 1.0) / 2.0), flow[:, :, :, 1:2] / ((fH - 1.0) / 2.0)], 3)
                warped_c = F.grid_sample(c_paired, flow_norm + grid, padding_mode='border')
                warped_cm = F.grid_sample(cm_paired, flow_norm + grid, padding_mode='border')
                warped_cm = remove_overlap(F.softmax(fake_segmap, dim=1), warped_cm)
                loss_l1_cloth += criterionL1(warped_cm, parse_cloth_mask) / (2 ** (4-i))
                loss_vgg += criterionVGG(warped_c, im_c) / (2 ** (4-i))
        # loss segmentation
        # generator
        CE_loss = cross_entropy2d(fake_segmap, label_onehot.transpose(0, 1)[0].long())
        
        if opt.no_GAN_loss:
            loss_G = (10 * loss_l1_cloth + loss_vgg + opt.tvlambda * loss_tv) + (CE_loss * opt.CElamda)
        else:
            fake_segmap_softmax = torch.softmax(fake_segmap, 1)
            pred_segmap = D(torch.cat((input1.detach(), input2.detach(), fake_segmap_softmax), dim=1))
            loss_G_GAN = criterionGAN(pred_segmap, True)      
            if not opt.G_D_seperate:  
                # discriminator
                fake_segmap_pred = D(torch.cat((input1.detach(), input2.detach(), fake_segmap_softmax.detach()),dim=1))
                real_segmap_pred = D(torch.cat((input1.detach(), input2.detach(), label),dim=1))
                loss_D_fake = criterionGAN(fake_segmap_pred, False)
                loss_D_real = criterionGAN(real_segmap_pred, True)

                # loss sum
                loss_G = (opt.loss_l1_cloth_lambda * loss_l1_cloth + loss_vgg +opt.tvlambda * loss_tv) + (CE_loss * opt.CElamda + loss_G_GAN * opt.GANlambda)  # warping + seg_generation
                loss_D = loss_D_fake + loss_D_real
                
            else: # train G first after that train D
                # loss G sum
                loss_G = (opt.loss_l1_cloth_lambda * loss_l1_cloth + loss_vgg + opt.tvlambda * loss_tv) + (CE_loss * opt.CElamda + loss_G_GAN * opt.GANlambda)  # warping + seg_generation
            
                
                # discriminator
                with torch.no_grad():
                    _, fake_segmap, _, _ = tocg(opt, input1, input2)
                fake_segmap_softmax = torch.softmax(fake_segmap, 1)
                
                # loss discriminator
                fake_segmap_pred = D(torch.cat((input1.detach(), input2.detach(), fake_segmap_softmax.detach()),dim=1))
                real_segmap_pred = D(torch.cat((input1.detach(), input2.detach(), label),dim=1))
                loss_D_fake = criterionGAN(fake_segmap_pred, False)
                loss_D_real = criterionGAN(real_segmap_pred, True)
                
                loss_D = loss_D_fake + loss_D_real
                
            board.add_scalar('val_warping_loss', loss_G.item(), step + 1)
            board.add_scalar('val_l1_cloth', loss_l1_cloth.item(), step + 1)
            board.add_scalar('val_vgg', loss_vgg.item(), step + 1)
            board.add_scalar('val_total_variation_loss', loss_tv.item(), step + 1)
            board.add_scalar('val_cross_entropy', CE_loss.item(), step + 1)
# Wandb     
            if not opt.no_GAN_loss:
                board.add_scalar('val_gan', loss_G_GAN.item(), step + 1)
                # loss D
                board.add_scalar('val_discriminator', loss_D.item(), step + 1)
                board.add_scalar('val_pred_real', loss_D_real.item(), step + 1)
                board.add_scalar('val_pred_fake', loss_D_fake.item(), step + 1)
            if not os.path.exists(os.path.join(opt.results_dir,'val')):
                os.makedirs(os.path.join(opt.results_dir,'val'))
            save_image(warped_cloth_paired, os.path.join(opt.results_dir,'val', f'warped_cloth_paired_{step}.png'))
            grid = make_grid([(c_paired[0].cpu() / 2 + 0.5), (cm_paired[0].cpu()).expand(3, -1, -1), visualize_segmap(parse_agnostic.cpu()), ((densepose.cpu()[0]+1)/2),
                              (im_c[0].cpu() / 2 + 0.5), parse_cloth_mask[0].cpu().expand(3, -1, -1), (warped_cloth_paired[0].cpu().detach() / 2 + 0.5), (warped_cm_onehot[0].cpu().detach()).expand(3, -1, -1),
                              visualize_segmap(label.cpu()), visualize_segmap(fake_segmap.cpu()), (im[0]/2 +0.5), (misalign[0].cpu().detach()).expand(3, -1, -1)],
                                nrow=4)
            board.add_images('valid_images', grid.unsqueeze(0), step + 1)
            if wandb is not None:
                my_table = wandb.Table(columns=['Grid', 'Segmentation Map For Label','Segmentation Map For Fake Data','Warped Cloth','Misalign'])
                grid_wandb = get_wandb_image(grid, wandb)
                segmap_for_label = get_wandb_image(visualize_segmap(label.cpu()), wandb)
                segmap_for_fake_data = get_wandb_image(visualize_segmap(fake_segmap.cpu()), wandb)
                warped_cloth = get_wandb_image((warped_cloth_paired[0].cpu().detach() / 2 + 0.5), wandb)
                misalign_cloth = get_wandb_image((misalign[0].cpu().detach()).expand(3, -1, -1), wandb)
                my_table.add_data(grid_wandb, segmap_for_label,segmap_for_fake_data,warped_cloth,misalign_cloth)
                wandb.log({'Val_Table': my_table, 'val_warping_loss': loss_G.item()
                ,'val_l1_cloth':loss_l1_cloth.item()
                ,'val_vgg':loss_vgg.item()
                ,'val_total_variation_loss':loss_tv.item()
                ,'val_cross_entropy':CE_loss.item()})
                
            # calculate iou
            iou = iou_metric(F.softmax(fake_segmap, dim=1).detach(), label)
            iou_list.append(iou.item())
            wandb.log({'val_iou':np.mean(iou_list) })
            board.add_scalar('val_iou', np.mean(iou_list), step + 1)
            
          
def print_log(log_path, content, to_print=True):
    import os
    if os.path.exists(log_path):
        with open(log_path, 'a') as f:
            f.write(content)
            f.write('\n')

        if to_print:
            print(content)
             
             
def get_sam(opt):
    sam = sam_model_registry[opt.sam_model_type](checkpoint=opt.sam_checkpoint) 
    if opt.cuda:
        sam.to(device="cuda") 
    return sam

def train_hrviton_tocg_(opt_, root_opt_, run_wandb=False, sweep=None):
    global opt, root_opt, wandb,sweep_id
    opt,root_opt = process_opt(opt_, root_opt_)
    sweep_id = None
    if sweep is not None:
        import wandb 
        wandb.login()
        sweep_id = wandb.sweep(sweep=sweep, project="Fashion-NeRF-Sweep")
        wandb.agent(sweep_id,_train_hrviton_tocg_sweep,count=5)
    elif run_wandb:
        import wandb
        wandb.login()
        wandb.init(project="Fashion-NeRF", entity='prime_lab', notes=f"question: {opt.question}, intent: {opt.intent}", tags=[f"{root_opt.experiment_run}"], config=vars(opt))
        temp_opt = vars(opt)
        temp_opt['wandb_name'] = wandb.run.name
        opt = argparse.Namespace(**temp_opt)
        _train_hrviton_tocg_()
    else:
        wandb = None
        _train_hrviton_tocg_()
 
 
def _train_hrviton_tocg_sweep():
    if wandb is not None:
        with wandb.init(project="Fashion-NeRF-Sweep", entity='prime_lab', notes=f"question: {opt.question}, intent: {opt.intent}", tags=[f"{root_opt.experiment_run}"], config=vars(opt)):
            _train_hrviton_tocg_()
            
def _train_hrviton_tocg_():
    global opt, root_opt, wandb,sweep_id
    if sweep_id is not None:
        opt = wandb.config
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(log_dir = opt.tensorboard_dir)
    torch.cuda.set_device(opt.device)
    experiment_string = f"{root_opt.experiment_run.replace('/','_')}_{root_opt.opt_vton_yaml.replace('yaml/','')}"
    with open(os.path.join(root_opt.experiment_run_yaml, experiment_string), 'w') as outfile:
        yaml.dump(vars(opt), outfile, default_flow_style=False)
    # Directories
    log_path = os.path.join(opt.results_dir, 'log.txt')
    if not os.path.exists(opt.results_dir):
        os.makedirs(opt.results_dir)
        with open(log_path, 'w') as file:
            file.write(f"Hello, this is experiment {root_opt.experiment_run} \n")
            
    train_dataset = FashionNeRFDataset(root_opt, opt, viton=True, model='viton')
    # train_dataset.__getitem__(0)
    train_loader = FashionDataLoader(train_dataset, root_opt.viton_batch_size, root_opt.viton_workers, True)

    test_dataset = FashionNeRFDataset(root_opt, opt, viton=True, mode='test', model='viton')
    test_loader = FashionDataLoader(test_dataset, opt.num_test_visualize, 1, False)
    validation_dataset = Subset(test_dataset, np.arange(50))
    validation_loader = FashionDataLoader(validation_dataset, opt.num_test_visualize, root_opt.viton_workers, False)

    # Model
    input1_nc = 4  # cloth + cloth-mask
    input2_nc = opt.semantic_nc + 3  # parse_agnostic + densepose
    
    sam = None
    if opt.segment_anything:
        sam = get_sam(opt)
    
    
    tocg = ConditionGenerator(opt, input1_nc=4, input2_nc=input2_nc, output_nc=opt.output_nc, ngf=96, norm_layer=nn.BatchNorm2d,segment_anything=sam)
    D = define_D(input_nc=input1_nc + input2_nc + opt.output_nc, Ddownx2 = opt.Ddownx2, Ddropout = opt.Ddropout, n_layers_D=3, spectral = opt.spectral, num_D = opt.num_D)

    # Load Checkpoint
    last_step = root_opt.load_last_step if type(root_opt.load_last_step) == bool else eval(root_opt.load_last_step)
    if last_step:
        load_checkpoint(tocg, opt.tocg_load_step_checkpoint)
        print_log(log_path, f'Load pretrained model from {opt.tocg_load_step_checkpoint}')
        
        load_checkpoint(D, opt.tocg_discriminator_load_step_checkpoint)
        print_log(log_path, f'Load pretrained discimrinator model from {opt.tocg_discriminator_load_step_checkpoint}')
    elif os.path.exists(opt.tocg_load_final_checkpoint):
        load_checkpoint(tocg, opt.tocg_load_final_checkpoint)
        print_log(log_path, f'Load pretrained model from {opt.tocg_load_final_checkpoint}')
        
        load_checkpoint(D, opt.tocg_discriminator_load_final_checkpoint)
        print_log(log_path, f'Load pretrained dicriminator model from {opt.tocg_discriminator_load_final_checkpoint}')
    

    # Train
    train_model(opt, root_opt, train_loader,test_loader, validation_loader, board, tocg, D, wandb)

    # Save Checkpoint
    if wandb is not None:
        wandb.finish()
    if not os.path.exists(opt.tocg_save_final_checkpoint_dir):
        os.makedirs(opt.tocg_save_final_checkpoint_dir)
    save_checkpoint(tocg,opt.tocg_save_final_checkpoint, opt)
    if not os.path.exists(opt.tocg_discriminator_save_final_checkpoint_dir):
        os.makedirs(opt.tocg_discriminator_save_final_checkpoint_dir)
    save_checkpoint(D,opt.tocg_discriminator_save_final_checkpoint , opt)
    print("Finished training %s!" % opt.VITON_Name)

