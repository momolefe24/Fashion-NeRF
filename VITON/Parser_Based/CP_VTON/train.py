#coding=utf-8
import torch
import yaml
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import os
import time
from VITON.Parser_Based.CP_VTON.cp_dataset import CPDataset, CPDataLoader
from VITON.Parser_Based.CP_VTON.networks import GMM, UnetGenerator, VGGLoss, load_checkpoint, save_checkpoint
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from VITON.Parser_Based.CP_VTON.visualization import board_add_image, board_add_images
from VITON.Parser_Based.CP_VTON.utils import *
opt,root_opt,wandb,sweep_id =None, None, None,None
   
def get_wandb_image(image, wandb):
    if image.max() <= 1.0:
        image = image*255 
    image_numpy = image.permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
    return wandb.Image(image_numpy)     
        
def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print('no checkpoint')
        raise
    log = model.load_state_dict(torch.load(checkpoint_path), strict=False)

def train_gmm(opt, train_loader,validation_loader, model, board, wandb=None):
    model.cuda()
    model.train()

    # criterion
    criterionL1 = nn.L1Loss()
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    
    
    for step in range(opt.niter + opt.niter_decay):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()
            
        im = inputs['image'].cuda()
        im_pose = inputs['pose_image'].cuda()
        im_h = inputs['head'].cuda()
        shape = inputs['shape'].cuda()
        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        im_c =  inputs['parse_cloth'].cuda()
        im_cm = inputs['parse_cloth_mask'].cuda()
        im_g = inputs['grid_image'].cuda()
            
        grid, theta = model(agnostic, c)
        warped_cloth = F.grid_sample(c, grid, padding_mode='border')
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
        warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')


        if opt.clip_warping:
            warped_cloth = warped_cloth * warped_mask + torch.ones_like(warped_cloth) * (1 - warped_mask)
        visuals = [ [im_h, shape, im_pose], 
                   [c, warped_cloth, im_c], 
                   [warped_grid, (warped_cloth+im)*0.5, im]]
        
        loss = criterionL1(warped_cloth, im_c)    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            board.add_scalar('warping_loss', loss.item(), step+1)
            board.add_image('Image', im[0], step+1)
            board.add_image('Pose Image', im_pose[0], step+1)
            board.add_image('Clothing', c[0], step+1)
            board.add_image('Parse Clothing', im_c[0], step+1)
            board.add_image('Parse Clothing Mask', im_cm[0], step+1)
            board.add_image('Warped Cloth', warped_cloth[0], step+1)
            board.add_image('Warped Cloth Mask', warped_mask[0], step+1)
            if wandb is not None:
                my_table = wandb.Table(columns=['Image', 'Pose Image','Clothing','Parse Clothing','Parse Clothing Mask','Warped Cloth','Warped Cloth Mask'])
                real_image_wandb = get_wandb_image(im[0], wandb=wandb) # 'Image'
                pose_image_wandb = get_wandb_image(im_pose[0], wandb=wandb) #'Pose Image'
                cloth_image_wandb = get_wandb_image(c[0], wandb=wandb) # 'Clothing'
                imc_image_wandb = get_wandb_image(im_c[0], wandb=wandb)  #'Parse Clothing'
                imc_mask_image_wandb = get_wandb_image(im_cm[0].cpu().expand(3, -1, -1), wandb=wandb)  #'Parse Clothing'
                warped_cloth_image_wandb = get_wandb_image((warped_cloth[0].cpu().detach() / 2 + 0.5), wandb=wandb) # 'Warped Cloth'
                warped_cloth_mask_wandb = get_wandb_image(warped_mask[0], wandb=wandb) # 'Warped Cloth Mask'
                my_table.add_data(real_image_wandb, pose_image_wandb, cloth_image_wandb, 
                    imc_image_wandb, imc_mask_image_wandb, warped_cloth_image_wandb, warped_cloth_mask_wandb
                )
                wandb.log({'warping_loss': 
                    loss.item(),
                'Table':my_table })
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %4f' % (step+1, t, loss.item()), flush=True)
        
        if (step + 1) % opt.val_count == 0:
            validate_gmm(validation_loader, model,board, wandb=wandb)
            model.train()
        if (step+1) % opt.save_period == 0:
            save_checkpoint(model, opt.gmm_save_step_checkpoint % (step+1))
        # break

def validate_gmm(validation_loader,model,board,wandb=wandb):
    model.cuda()
    model.eval()

    # criterion
    criterionL1 = nn.L1Loss()

    inputs = validation_loader.next_batch()
        
    im = inputs['image'].cuda()
    im_pose = inputs['pose_image'].cuda()
    im_h = inputs['head'].cuda()
    shape = inputs['shape'].cuda()
    agnostic = inputs['agnostic'].cuda()
    c = inputs['cloth'].cuda()
    cm = inputs['cloth_mask'].cuda()
    im_c =  inputs['parse_cloth'].cuda()
    im_cm =  inputs['parse_cloth_mask'].cuda()
    im_g = inputs['grid_image'].cuda()
        
    grid, theta = model(agnostic, c)
    warped_cloth = F.grid_sample(c, grid, padding_mode='border')
    warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
    warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')
    if opt.clip_warping:
        warped_cloth = warped_cloth * warped_mask + torch.ones_like(warped_cloth) * (1 - warped_mask)
    loss = criterionL1(warped_cloth, im_c) 
    
    board.add_scalar('Val/warping_loss', loss.item())
    board.add_image('Val/Image', im[0])
    board.add_image('Val/Pose Image', im_pose[0])
    board.add_image('Val/Clothing', c[0])
    board.add_image('Val/Parse Clothing', im_c[0])
    board.add_image('Val/Parse Clothing Mask', im_cm[0])
    board.add_image('Val/Warped Cloth', warped_cloth[0])
    board.add_image('Val/Warped Cloth Mask', warped_mask[0])
    if wandb is not None:
        my_table = wandb.Table(columns=['Image', 'Pose Image','Clothing','Parse Clothing','Parse Clothing Mask','Warped Cloth','Warped Cloth Mask'])
        real_image_wandb = get_wandb_image(im[0], wandb=wandb) # 'Image'
        pose_image_wandb = get_wandb_image(im_pose[0], wandb=wandb) #'Pose Image'
        cloth_image_wandb = get_wandb_image(c[0], wandb=wandb) # 'Clothing'
        imc_image_wandb = get_wandb_image(im_c[0], wandb=wandb)  #'Parse Clothing'
        imc_mask_image_wandb = get_wandb_image(im_cm[0].cpu().expand(3, -1, -1), wandb=wandb)  #'Parse Clothing'
        warped_cloth_image_wandb = get_wandb_image(warped_cloth[0], wandb=wandb) # 'Warped Cloth'
        warped_cloth_mask_wandb = get_wandb_image(warped_mask[0], wandb=wandb) # 'Warped Cloth Mask'
        my_table.add_data(real_image_wandb, pose_image_wandb, cloth_image_wandb, 
            imc_image_wandb, imc_mask_image_wandb, warped_cloth_image_wandb, warped_cloth_mask_wandb
        )
        wandb.log({'val_warping_loss': loss.item(),
        'Val_Table':my_table })

def train_tom(opt, train_loader, validation_loader, gmm_model, model, board, wandb=None):
    model.cuda()
    model.train()
    
    gmm_model.cuda()
    gmm_model.eval()
    
    # criterion
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()
    criterionMask = nn.L1Loss()
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    
    for step in range(opt.niter + opt.niter_decay):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()
            
        im = inputs['image'].cuda()
        im_pose = inputs['pose_image']
        im_h = inputs['head']
        shape = inputs['shape']

        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        
        
        grid, theta = gmm_model(agnostic, c)
        warped_cloth = F.grid_sample(c, grid, padding_mode='border')
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
        
        if opt.clip_warping:
            warped_cloth = warped_cloth * warped_mask + torch.ones_like(warped_cloth) * (1 - warped_mask)
        outputs = model(torch.cat([agnostic, warped_cloth],1))
        p_rendered, m_composite = torch.split(outputs, 3,1)
        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)
        p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

        visuals = [ [im_h, shape, im_pose], 
                   [warped_cloth, warped_mask*2-1, m_composite*2-1], 
                   [p_rendered, p_tryon, im]]
            
        loss_l1 = criterionL1(p_tryon, im)
        loss_vgg = criterionVGG(p_tryon, im)
        loss_mask = criterionMask(m_composite, warped_mask)
        loss = loss_l1 + loss_vgg + loss_mask
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            board.add_image('Real Image', im[0], 0)
            board.add_image('Head Image', im_h[0], 0)
            board.add_image('Cloth', c[0], 0)
            board.add_image('Cloth Mask', (cm*2-1)[0], 0)
            board.add_image('Warped Cloth', warped_cloth[0], 0)
            board.add_image('Warped Cloth Mask', (warped_mask*2-1)[0], 0)
            board.add_image('Composite Mask', (m_composite*2-1)[0], 0)
            board.add_image('Rendered Image', p_rendered[0], 0)
            board.add_image('Try-On', p_tryon[0], 0)
            board.add_scalar('composition_loss', loss.item(), step+1)
            board.add_scalar('l1_composition_loss', loss_l1.item(), step+1)
            board.add_scalar('vgg_composition_loss', loss_vgg.item(), step+1)
            board.add_scalar('mask_composition_loss', loss_mask.item(), step+1)
            if wandb is not None:
                my_table = wandb.Table(columns=['Image','Head Image','Cloth','Cloth Mask','Warped Cloth','Warped Cloth Mask','Composite Mask','Rendered Image','Try-On'])
                real_image_wandb = get_wandb_image(im[0], wandb=wandb)
                head_image_wandb = get_wandb_image(im_h[0], wandb=wandb)
                cloth_image_wandb = get_wandb_image(c[0], wandb=wandb)
                cm_2 = cm*2-1
                cloth_mask_image_wandb = get_wandb_image(cm_2[0], wandb=wandb)
                
                warped_cloth_image_wandb = get_wandb_image(warped_cloth[0], wandb=wandb)
                warped_cm_2 = warped_mask*2-1
                warped_cloth_mask_image_wandb = get_wandb_image(warped_cm_2[0], wandb=wandb)
                
                m_2 = m_composite*2-1
                m_composite_image_wandb = get_wandb_image(m_2[0], wandb=wandb)
                rendered_image_wandb = get_wandb_image(p_rendered[0], wandb=wandb)
                try_on_wandb = get_wandb_image(p_tryon[0], wandb=wandb)
                my_table.add_data(real_image_wandb,head_image_wandb,cloth_image_wandb,cloth_mask_image_wandb,
                               warped_cloth_image_wandb   ,warped_cloth_mask_image_wandb,
                                  m_composite_image_wandb,rendered_image_wandb,try_on_wandb)
                wandb.log({'l1_composition_loss': loss_l1.item(),
                           'vgg_composition_loss':loss_vgg.item(),
                           'mask_composition_loss':loss_mask.item(),
                           'composition_loss':loss.item(),
                'Table':my_table })
            t = time.time() - iter_start_time
            message = 'step: %8d, time: %.3f, loss: %.4f, l1: %.4f, vgg: %.4f, mask: %.4f' % (step+1, t, loss.item(), loss_l1.item(),  loss_vgg.item(), loss_mask.item())
            print('step: %8d, time: %.3f, loss: %.4f, l1: %.4f, vgg: %.4f, mask: %.4f' 
                    % (step+1, t, loss.item(), loss_l1.item(), 
                    loss_vgg.item(), loss_mask.item()), flush=True)
            print_log(os.path.join(opt.results_dir, 'log.txt'), message)
            
        if (step + 1) % opt.val_count == 0:
            validate_tom(validation_loader, model, gmm_model, board, wandb=wandb)
            model.train()
        if (step+1) % opt.save_period == 0:
            print_log(os.path.join(opt.results_dir, 'log.txt'), f'Save pretrained model from {opt.tom_load_step_checkpoint}')
            save_checkpoint(model, opt.tom_save_step_checkpoint % (step+1))


def validate_tom(validation_loader,model,gmm_model, board, wandb=wandb):
    model.cuda()
    model.eval()
    
    gmm_model.cuda()
    gmm_model.eval()
    
    # criterion
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()
    criterionMask = nn.L1Loss()
    
    for step in range(opt.niter + opt.niter_decay):
        inputs = validation_loader.next_batch()
            
        im = inputs['image'].cuda()
        im_pose = inputs['pose_image']
        im_h = inputs['head']
        shape = inputs['shape']

        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        
        
        grid, theta = gmm_model(agnostic, c)
        warped_cloth = F.grid_sample(c, grid, padding_mode='border')
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
        
        
        outputs = model(torch.cat([agnostic, warped_cloth],1))
        p_rendered, m_composite = torch.split(outputs, 3,1)
        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)
        p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

        visuals = [ [im_h, shape, im_pose], 
                   [warped_cloth, warped_mask*2-1, m_composite*2-1], 
                   [p_rendered, p_tryon, im]]
            
        loss_l1 = criterionL1(p_tryon, im)
        loss_vgg = criterionVGG(p_tryon, im)
        loss_mask = criterionMask(m_composite, warped_mask)
        loss = loss_l1 + loss_vgg + loss_mask
        if opt.clip_warping:
            warped_cloth = warped_cloth * warped_mask + torch.ones_like(warped_cloth) * (1 - warped_mask)
        board_add_images(board, 'combine', visuals, step+1)
        board.add_image('Val/Real Image', im[0], 0)
        board.add_image('Val/Head Image', im_h[0], 0)
        board.add_image('Val/Cloth', c[0], 0)
        board.add_image('Val/Cloth Mask', (cm*2-1)[0], 0)
        board.add_image('Val/Warped Cloth', warped_cloth[0], 0)
        board.add_image('Val/Warped Cloth Mask', (warped_mask*2-1)[0], 0)
        board.add_image('Val/Composite Mask', (m_composite*2-1)[0], 0)
        board.add_image('Val/Rendered Image', p_rendered[0], 0)
        board.add_image('Val/Try-On', p_tryon[0], 0)
        board.add_scalar('Val/composition_loss', loss.item(), step+1)
        board.add_scalar('Val/l1_composition_loss', loss_l1.item(), step+1)
        board.add_scalar('Val/vgg_composition_loss', loss_vgg.item(), step+1)
        board.add_scalar('Val/mask_composition_loss', loss_mask.item(), step+1)
        if wandb is not None:
            my_table = wandb.Table(columns=['Image','Head Image','Cloth','Cloth Mask','Warped Cloth','Warped Cloth Mask','Composite Mask','Rendered Image','Try-On'])
            real_image_wandb = get_wandb_image(im[0], wandb=wandb)
            head_image_wandb = get_wandb_image(im_h[0], wandb=wandb)
            cloth_image_wandb = get_wandb_image(c[0], wandb=wandb)
            cm_2 = cm*2-1
            cloth_mask_image_wandb = get_wandb_image(cm_2[0], wandb=wandb)
            
            warped_cloth_image_wandb = get_wandb_image(warped_cloth[0], wandb=wandb)
            warped_cm_2 = warped_mask*2-1
            warped_cloth_mask_image_wandb = get_wandb_image(warped_cm_2[0], wandb=wandb)
            
            m_2 = m_composite*2-1
            m_composite_image_wandb = get_wandb_image(m_2[0], wandb=wandb)
            rendered_image_wandb = get_wandb_image(p_rendered[0], wandb=wandb)
            try_on_wandb = get_wandb_image(p_tryon[0], wandb=wandb)
            my_table.add_data(real_image_wandb,head_image_wandb,cloth_image_wandb,cloth_mask_image_wandb,
                            warped_cloth_image_wandb   ,warped_cloth_mask_image_wandb,
                                m_composite_image_wandb,rendered_image_wandb,try_on_wandb)
            wandb.log({'val_l1_composition_loss': loss_l1.item(),
                        'val_vgg_composition_loss':loss_vgg.item(),
                        'val_mask_composition_loss':loss_mask.item(),
                        'val_composition_loss':loss.item(),
                'val_Table':my_table })
        
            
def print_log(log_path, content, to_print=True):
    import os
    if os.path.exists(log_path):
        with open(log_path, 'a') as f:
            f.write(content)
            f.write('\n')

        if to_print:
            print(content)
            
def split_dataset(dataset,train_size=0.8):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    train_indices, validation_indices = train_test_split(indices, train_size=train_size)
    train_subset = Subset(dataset, train_indices)
    validation_subset = Subset(dataset, validation_indices)
    return train_subset, validation_subset


def _train_cpvton_sweep():
    if wandb is not None:
        with wandb.init(project="Fashion-NeRF-Sweep", entity='rail_lab', tags=[f"{root_opt.experiment_run}"], config=vars(opt)):
            _train_cpvton_()
            
def train_cpvton_(opt_, root_opt_, run_wandb=False, sweep=None):
    global opt, root_opt, wandb,sweep_id
    opt,root_opt = process_opt(opt_, root_opt_)
    sweep_id = None
    if sweep is not None:
        import wandb 
        wandb.login()
        sweep_id = wandb.sweep(sweep=sweep, project="Fashion-NeRF-Sweep")
        wandb.agent(sweep_id,_train_cpvton_sweep,count=3)
    elif run_wandb:
        import wandb
        wandb.login()
        wandb.init(project="Fashion-NeRF", entity='rail_lab', tags=[f"{root_opt.experiment_run}"], config=vars(opt))
        temp_opt = vars(opt)
        temp_opt['wandb_name'] = wandb.run.name
        opt = argparse.Namespace(**temp_opt)
        _train_cpvton_()
    else:
        wandb = None
        _train_cpvton_()
    

def make_dirs(opt):
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    if not os.path.exists(opt.results_dir):
        os.makedirs(opt.results_dir)
    if not os.path.exists(opt.gmm_save_final_checkpoint_dir):
        os.makedirs(opt.gmm_save_final_checkpoint_dir)
    if not os.path.exists(opt.tom_save_final_checkpoint_dir):
        os.makedirs(opt.tom_save_final_checkpoint_dir)
    if not os.path.exists(opt.gmm_save_step_checkpoint_dir):
        os.makedirs(opt.gmm_save_step_checkpoint_dir)
    if not os.path.exists(opt.tom_save_step_checkpoint_dir):
        os.makedirs(opt.tom_save_step_checkpoint_dir)
        
def _train_cpvton_():
    global opt, root_opt, wandb,sweep_id
    make_dirs(opt)
    if sweep_id is not None:
        opt.niter = wandb.config.niter
        opt.niter_decay = wandb.config.niter_decay
        opt.lr = wandb.config.lr
        opt.init_type = wandb.config.init_type
    board = SummaryWriter(log_dir = opt.tensorboard_dir)    
    experiment_string = f"{root_opt.experiment_run.replace('/','_')}_{root_opt.opt_vton_yaml.replace('yaml/','')}"
    with open(os.path.join(root_opt.experiment_run_yaml, experiment_string), 'w') as outfile:
        yaml.dump(vars(opt), outfile, default_flow_style=False)
    log_path = os.path.join(opt.results_dir, 'log.txt')
    with open(log_path, 'w') as file:
        file.write(f"Hello, this is experiment {root_opt.experiment_run} \n")
    
    train_dataset = CPDataset(root_opt, opt)
    train_dataset, validation_dataset = split_dataset(train_dataset)
    # create dataloader
    train_loader = CPDataLoader(opt, train_dataset)
    validation_loader = CPDataLoader(opt, validation_dataset)
    
    if opt.VITON_Model == 'GMM':
        model = GMM(opt)
        last_step = root_opt.load_last_step if type(root_opt.load_last_step) == bool else eval(root_opt.load_last_step)
        if last_step:
            load_checkpoint(model, opt.gmm_load_step_checkpoint)
            print_log(log_path, f'Load pretrained model from {opt.gmm_load_step_checkpoint}')
        elif os.path.exists(opt.gmm_load_final_checkpoint):
            load_checkpoint(model, opt.gmm_load_final_checkpoint)
            print_log(log_path, f'Load pretrained model from {opt.gmm_load_final_checkpoint}')
        train_gmm(opt, train_loader, validation_loader, model, board, wandb=wandb)
        save_checkpoint(model,opt.gmm_save_final_checkpoint)
    elif opt.VITON_Model == 'TOM':
        model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
        gmm_model = GMM(opt)
        last_step = root_opt.load_last_step if type(root_opt.load_last_step) == bool else eval(root_opt.load_last_step)
        if last_step:
            load_checkpoint(gmm_model, opt.gmm_load_step_checkpoint)
            print_log(log_path, f'Load pretrained gmm_model from {opt.gmm_load_step_checkpoint}')
        elif os.path.exists(opt.gmm_load_final_checkpoint):
            load_checkpoint(gmm_model, opt.gmm_load_final_checkpoint)
            print_log(log_path, f'Load pretrained gmm_model from {opt.gmm_load_final_checkpoint}')
        
        
        last_step = root_opt.load_last_step if type(root_opt.load_last_step) == bool else eval(root_opt.load_last_step)
        if last_step:
            load_checkpoint(model, opt.tom_load_step_checkpoint)
            print_log(log_path, f'Load pretrained model from {opt.tom_load_step_checkpoint}')
        elif os.path.exists(opt.tom_load_final_checkpoint):
            load_checkpoint(model, opt.tom_load_final_checkpoint)
            print_log(log_path, f'Load pretrained model from {opt.tom_load_final_checkpoint}')
        train_tom(opt, train_loader,validation_loader, gmm_model, model, board, wandb=wandb)
        save_checkpoint(model, opt.tom_save_final_checkpoint)
        
  
    print("Finished to train" )
