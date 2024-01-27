import time
# from VITON.Parser_Based.ACGPN2.ACGPN_inference.data.data_loader import CreateDataLoader
from VITON.Parser_Based.ACGPN.ACGPN_train.data.data_loader import CreateDataLoader
from VITON.Parser_Based.ACGPN.ACGPN_inference.models.models import create_model
from VITON.Parser_Based.ACGPN.ACGPN_inference.util import util
import os
import numpy as np
import torch
import argparse
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import cv2
# writer = SummaryWriter('runs/G1G2')
SIZE=320
NC=14
fix = lambda path: os.path.normpath(path)
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

def generate_label_color(opt, inputs):
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
    M_f=M_f.astype(np.int)
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
def changearm(old_label):
    label=old_label
    arm1=torch.FloatTensor((old_label.cpu().numpy()==11).astype(np.int))
    arm2=torch.FloatTensor((old_label.cpu().numpy()==13).astype(np.int))
    noise=torch.FloatTensor((old_label.cpu().numpy()==7).astype(np.int))
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
    parser.device = int(root_opt.device)
    parser.datamode = root_opt.datamode
    parser.dataset_name = root_opt.dataset_name
    parser.warp_load_from_model = root_opt.warp_load_from_model
    parser.load_last_step = root_opt.load_last_step if type(root_opt.load_last_step) == bool else eval(root_opt.load_last_step)
    parser.run_wandb = root_opt.run_wandb
    parser.viton_batch_size = root_opt.viton_batch_size
    parser.save_period = root_opt.save_period
    parser.print_step = root_opt.print_step
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
    
    opt.vgg_save_final_checkpoint_dir = opt.vgg_save_final_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    opt.vgg_save_final_checkpoint_dir = fix(opt.vgg_save_final_checkpoint_dir)
    opt.vgg_save_final_checkpoint = os.path.join(opt.vgg_save_final_checkpoint_dir, opt.vgg_save_final_checkpoint)
    
    opt.vgg_load_final_checkpoint_dir = opt.vgg_load_final_checkpoint_dir.format(root_opt.vgg_experiment_from_run, root_opt.vgg_experiment_from_dir)
    opt.vgg_load_final_checkpoint_dir = fix(opt.vgg_load_final_checkpoint_dir)
    opt.vgg_load_final_checkpoint = os.path.join(opt.vgg_load_final_checkpoint_dir, opt.vgg_load_final_checkpoint)
    opt.vgg_load_final_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.vgg_load_final_checkpoint)
    
   # ================================= G1 ================================= 
    opt.g1_save_step_checkpoint_dir = opt.g1_save_step_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    opt.g1_save_step_checkpoint = os.path.join(opt.g1_save_step_checkpoint_dir, opt.g1_save_step_checkpoint)
    opt.g1_save_step_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.g1_save_step_checkpoint)
    
    
    opt.g1_save_final_checkpoint_dir = opt.g1_save_final_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    opt.g1_save_final_checkpoint_dir = fix(opt.g1_save_final_checkpoint_dir)
    opt.g1_save_final_checkpoint = os.path.join(opt.g1_save_final_checkpoint_dir, opt.g1_save_final_checkpoint)
    opt.g1_save_final_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.g1_save_final_checkpoint)
    
    
    opt.g1_load_final_checkpoint_dir = opt.g1_load_final_checkpoint_dir.format(root_opt.g1_experiment_from_run, root_opt.g1_experiment_from_dir)
    opt.g1_load_final_checkpoint_dir = fix(opt.g1_load_final_checkpoint_dir)
    opt.g1_load_final_checkpoint = os.path.join(opt.g1_load_final_checkpoint_dir, opt.g1_load_final_checkpoint)
    opt.g1_load_final_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.g1_load_final_checkpoint)
    
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
   # ================================= G2 ================================= 
    opt.g2_save_step_checkpoint_dir = opt.g2_save_step_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    opt.g2_save_step_checkpoint_dir = fix(opt.g2_save_step_checkpoint_dir)
    opt.g2_save_step_checkpoint = os.path.join(opt.g2_save_step_checkpoint_dir, opt.g2_save_step_checkpoint)
    opt.g2_save_step_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.g2_save_step_checkpoint)
    
    opt.g2_save_final_checkpoint_dir = opt.g2_save_final_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    opt.g2_save_final_checkpoint_dir = fix(opt.g2_save_final_checkpoint_dir)
    opt.g2_save_final_checkpoint = os.path.join(opt.g2_save_final_checkpoint_dir, opt.g2_save_final_checkpoint)
    opt.g2_save_final_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.g2_save_final_checkpoint)
    
    opt.g2_load_final_checkpoint_dir = opt.g2_load_final_checkpoint_dir.format(root_opt.g2_experiment_from_run, root_opt.g2_experiment_from_dir)
    opt.g2_load_final_checkpoint_dir = fix(opt.g2_load_final_checkpoint_dir)
    opt.g2_load_final_checkpoint = os.path.join(opt.g2_load_final_checkpoint_dir, opt.g2_load_final_checkpoint)
    opt.g2_load_final_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.g2_load_final_checkpoint)
    
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
    
   # ================================= G ================================= 
    opt.g_save_step_checkpoint_dir = opt.g_save_step_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    opt.g_save_step_checkpoint_dir = fix(opt.g_save_step_checkpoint_dir)
    opt.g_save_step_checkpoint = os.path.join(opt.g_save_step_checkpoint_dir, opt.g_save_step_checkpoint)
    opt.g_save_step_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.g_save_step_checkpoint)
    

    opt.g_save_final_checkpoint_dir = opt.g_save_final_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    opt.g_save_final_checkpoint_dir = fix(opt.g_save_final_checkpoint_dir)
    opt.g_save_final_checkpoint = os.path.join(opt.g_save_final_checkpoint_dir, opt.g_save_final_checkpoint)
    opt.g_save_final_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.g_save_final_checkpoint)
    
    opt.g_load_final_checkpoint_dir = opt.g_load_final_checkpoint_dir.format(root_opt.g_experiment_from_run, root_opt.g_experiment_from_dir)
    opt.g_load_final_checkpoint_dir = fix(opt.g_load_final_checkpoint_dir)
    opt.g_load_final_checkpoint = os.path.join(opt.g_load_final_checkpoint_dir, opt.g_load_final_checkpoint)
    opt.g_load_final_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.g_load_final_checkpoint)
    
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
   # ================================= d ================================= 
    opt.d_save_step_checkpoint_dir = opt.d_save_step_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    opt.d_save_step_checkpoint_dir = fix(opt.d_save_step_checkpoint_dir)
    opt.d_save_step_checkpoint = os.path.join(opt.d_save_step_checkpoint_dir, opt.d_save_step_checkpoint)
    opt.d_save_step_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.d_save_step_checkpoint)    
    
    opt.d_save_final_checkpoint_dir = opt.d_save_final_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    opt.d_save_final_checkpoint_dir = fix(opt.d_save_final_checkpoint_dir)
    opt.d_save_final_checkpoint = os.path.join(opt.d_save_final_checkpoint_dir, opt.d_save_final_checkpoint)
    opt.d_save_final_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.d_save_final_checkpoint)    
    
    opt.d_load_final_checkpoint_dir = opt.d_load_final_checkpoint_dir.format(root_opt.d_experiment_from_run, root_opt.d_experiment_from_dir)
    opt.d_load_final_checkpoint_dir = fix(opt.d_load_final_checkpoint_dir)
    opt.d_load_final_checkpoint = os.path.join(opt.d_load_final_checkpoint_dir, opt.d_load_final_checkpoint)
    opt.d_load_final_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.d_load_final_checkpoint)    
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
   # ================================= Unet ================================= 
   
    opt.unet_save_step_checkpoint_dir = opt.unet_save_step_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    opt.unet_save_step_checkpoint_dir = fix(opt.unet_save_step_checkpoint_dir)
    opt.unet_save_step_checkpoint = os.path.join(opt.unet_save_step_checkpoint_dir, opt.unet_save_step_checkpoint)
    opt.unet_save_step_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.unet_save_step_checkpoint) 
    
    opt.unet_save_final_checkpoint_dir = opt.unet_save_final_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    opt.unet_save_final_checkpoint_dir = fix(opt.unet_save_final_checkpoint_dir)
    opt.unet_save_final_checkpoint = os.path.join(opt.unet_save_final_checkpoint_dir, opt.unet_save_final_checkpoint)
    opt.unet_save_final_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.unet_save_final_checkpoint)
    
    opt.unet_load_final_checkpoint_dir = opt.unet_load_final_checkpoint_dir.format(root_opt.unet_experiment_from_run, root_opt.unet_experiment_from_dir)
    opt.unet_load_final_checkpoint = os.path.join(opt.unet_load_final_checkpoint_dir, opt.unet_load_final_checkpoint)
    opt.unet_load_final_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.unet_load_final_checkpoint)
    
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
        
    if root_opt.datamode == 'train':
        opt.isTrain = True
    else:
        opt.isTrain = False
    return opt

def test_acgpn_(opt, root_opt):
    opt,root_opt = process_opt(opt, root_opt)
    print("Start to test %s!")
    _test_acgpn_(opt, root_opt)

def _test_acgpn_(opt, root_opt):
    writer = SummaryWriter(opt.tensorboard_dir)
    data_loader = CreateDataLoader(opt, root_opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('# Inference images = %d' % dataset_size)

    model = create_model(opt)
    start_epoch, epoch_iter = 1, 0
    step = 0
    total_steps = (start_epoch-1) * dataset_size + epoch_iter


    step = 0

    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size
        for i, data in enumerate(dataset, start=epoch_iter):

            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            save_fake = True
            t_mask = torch.FloatTensor((data['label'].cpu().numpy() == 7).astype(np.float))
            mask_clothes = torch.FloatTensor((data['label'].cpu().numpy() == 4).astype(np.int))
            mask_fore = torch.FloatTensor((data['label'].cpu().numpy() > 0).astype(np.int))
            img_fore = data['image'] * mask_fore
            img_fore_wc = img_fore * mask_fore
            all_clothes_label = changearm(data['label'])



            ############## Forward Pass ######################
            fake_image, warped_cloth, refined_cloth = model(Variable(data['label'].cuda()), Variable(data['edge'].cuda()), Variable(img_fore.cuda()), Variable(
            mask_clothes.cuda()), Variable(data['color'].cuda()), Variable(all_clothes_label.cuda()), Variable(data['image'].cuda()), Variable(data['pose'].cuda()), Variable(data['image'].cuda()), Variable(mask_fore.cuda()))

            # make output folders
            output_dir = os.path.join(opt.results_dir, opt.phase)
            fake_image_dir = os.path.join(output_dir, 'try-on')
            os.makedirs(fake_image_dir, exist_ok=True)
            warped_cloth_dir = os.path.join(output_dir, 'warped_cloth')
            os.makedirs(warped_cloth_dir, exist_ok=True)
            refined_cloth_dir = os.path.join(output_dir, 'refined_cloth')
            os.makedirs(refined_cloth_dir, exist_ok=True)
            
            # save output
            for j in range(opt.batchSize):
                print("Saving", data['name'][j])
                util.save_tensor_as_image(fake_image[j],
                                        os.path.join(fake_image_dir, data['name'][j]))
                util.save_tensor_as_image(warped_cloth[j],
                                        os.path.join(warped_cloth_dir, data['name'][j]))
                util.save_tensor_as_image(refined_cloth[j],
                                        os.path.join(refined_cloth_dir, data['name'][j]))
                if epoch_iter >= dataset_size:
                    break
        
        # end of epoch 
        iter_end_time = time.time()
        print('End of epoch %d / %d \t Time Taken: %d sec' %
            (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        break