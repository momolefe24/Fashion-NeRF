import torch
import torch.nn as nn

from torchvision.utils import make_grid, save_image
from dataset import FashionDataLoader, FashionNeRFDataset
import argparse
import os
import time
from VITON.Parser_Based.HR_VITON.cp_dataset import CPDatasetTest, CPDataLoader
from VITON.Parser_Based.HR_VITON.networks import ConditionGenerator, load_checkpoint, define_D
from VITON.Parser_Based.HR_VITON.utils import condition_process_opt
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

def test_hrviton_tocg_(opt, root_opt):
    opt,root_opt = condition_process_opt(opt, root_opt)
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