import torch
import torch.nn as nn

from torchvision.utils import make_grid, save_image
from dataset import FashionDataLoader, FashionNeRFDataset
import argparse
import os
import time
from VITON.Parser_Based.HR_VITON.cp_dataset import CPDatasetTest, CPDataLoader
from VITON.Parser_Based.HR_VITON.networks import ConditionGenerator, load_checkpoint, define_D
from tqdm import tqdm
from tensorboardX import SummaryWriter
from VITON.Parser_Based.HR_VITON.utils import *
from VITON.Parser_Based.HR_VITON.get_norm_const import D_logit

fix = lambda path: os.path.normpath(path)

def test(opt, test_loader, tocg, D=None):
    # Model
    tocg.cuda()
    tocg.eval()
    if D is not None:
        D.cuda()
        D.eval()
    prediction_dir = os.path.join(opt.results_dir, 'prediction')
    ground_truth_dir = os.path.join(opt.results_dir, 'ground_truth')
    ground_truth_mask_dir = os.path.join(opt.results_dir, 'ground_truth_mask')
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)
    if not os.path.exists(ground_truth_dir):
        os.makedirs(ground_truth_dir)
    if not os.path.exists(ground_truth_mask_dir):
        os.makedirs(ground_truth_mask_dir)
    num = 0
    iter_start_time = time.time()
    if D is not None:
        D_score = []
    for inputs in test_loader.data_loader:
        
        # input1
        c_paired = inputs['cloth']['paired'].cuda()
        cm_paired = inputs['cloth_mask']['paired'].cuda()
        cm_paired = torch.FloatTensor((cm_paired.detach().cpu().numpy() > 0.5).astype(np.float)).cuda()
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

        with torch.no_grad():
            # inputs
            input1 = torch.cat([c_paired, cm_paired], 1)
            input2 = torch.cat([parse_agnostic, densepose], 1)

            # forward
            # flow_list, fake_segmap, warped_cloth_paired, warped_clothmask_paired = tocg(opt, input1, input2)
            
            
            if opt.segment_anything:
                flow_list, fake_segmap, warped_cloth_paired, warped_clothmask_paired = tocg(opt, input1, input2, im_c=im_c)
            else:
                flow_list, fake_segmap, warped_cloth_paired, warped_clothmask_paired = tocg(opt, input1, input2)
                
            # warped cloth mask one hot 
            warped_cm_onehot = torch.FloatTensor((warped_clothmask_paired.detach().cpu().numpy() > 0.5).astype(np.float)).cuda()
            
            if opt.clothmask_composition != 'no_composition':
                if opt.clothmask_composition == 'detach':
                    cloth_mask = torch.ones_like(fake_segmap)
                    cloth_mask[:,3:4, :, :] = warped_cm_onehot
                    fake_segmap = fake_segmap * cloth_mask
                    
                if opt.clothmask_composition == 'warp_grad':
                    cloth_mask = torch.ones_like(fake_segmap)
                    cloth_mask[:,3:4, :, :] = warped_clothmask_paired
                    fake_segmap = fake_segmap * cloth_mask
            if D is not None:
                fake_segmap_softmax = F.softmax(fake_segmap, dim=1)
                pred_segmap = D(torch.cat((input1.detach(), input2.detach(), fake_segmap_softmax), dim=1))
                score = D_logit(pred_segmap)
                # score = torch.exp(score) / opt.norm_const
                score = (score / (1 - score)) / opt.norm_const
                print("prob0", score)
                for i in range(cm_paired.shape[0]):
                    name = inputs['c_name']['paired'][i].replace('.jpg', '.png')
                    D_score.append((name, score[i].item()))
            
            
            # generated fake cloth mask & misalign mask
            fake_clothmask = (torch.argmax(fake_segmap.detach(), dim=1, keepdim=True) == 3).long()
            misalign = fake_clothmask - warped_cm_onehot
            misalign[misalign < 0.0] = 0.0
            image_name = os.path.join(prediction_dir, inputs['im_name'][0])
            ground_truth_image_name = os.path.join(ground_truth_dir, inputs['im_name'][0])
            ground_truth_mask_name = os.path.join(ground_truth_mask_dir, inputs['im_name'][0])
            save_image((warped_cloth_paired.cpu().detach() / 2 + 0.5), image_name)
            save_image((im_c.cpu().detach() / 2 + 0.5), ground_truth_image_name)
            save_image((parse_cloth_mask.cpu().detach() / 2 + 0.5), ground_truth_mask_name)
        num += c_paired.shape[0]
        print(num)
    if D is not None:
        D_score.sort(key=lambda x: x[1], reverse=True)
        # Save D_score
        for name, score in D_score:
            f = open(os.path.join(opt.results_dir,'test','rejection_prob.txt'), 'a')
            f.write(name + ' ' + str(score) + '\n')
            f.close()
    print(f"Test time {time.time() - iter_start_time}")

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
    parser.results_dir = os.path.join(parser.results_dir, parser.datamode)
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
    parser.fp16 = root_opt.fp16
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


def test_hrviton_tocg_(opt, root_opt):
    opt,root_opt = process_opt(opt, root_opt)
    print("Start to test %s!")
    _test_hrviton_tocg_(opt, root_opt)
    


def _test_hrviton_tocg_(opt, root_opt):
    # create test dataset & loader
    test_dataset = FashionNeRFDataset(root_opt, opt, viton=True, mode='test', model='viton')
    test_dataset.__getitem__(0)
    test_loader = FashionDataLoader(test_dataset, 1, 1, False)

    # Model
    input1_nc = 4  # cloth + cloth-mask
    input2_nc = opt.semantic_nc + 3  # parse_agnostic + densepose
    tocg = ConditionGenerator(opt, input1_nc=input1_nc, input2_nc=input2_nc, output_nc=opt.output_nc, ngf=96, norm_layer=nn.BatchNorm2d)
    if os.path.exists(opt.tocg_discriminator_load_final_checkpoint):
        D = define_D(input_nc=input1_nc + input2_nc + opt.output_nc, Ddownx2 = opt.Ddownx2, Ddropout = opt.Ddropout, n_layers_D=3, spectral = opt.spectral, num_D = opt.num_D)
    else:
        D = None
    # Load Checkpoint
    load_checkpoint(tocg, opt.tocg_load_final_checkpoint)
    if os.path.exists(opt.tocg_discriminator_load_final_checkpoint):
        load_checkpoint(D, opt.tocg_discriminator_load_final_checkpoint)
    # Train
    test(opt, test_loader,tocg, D=D)

    print("Finished testing!")