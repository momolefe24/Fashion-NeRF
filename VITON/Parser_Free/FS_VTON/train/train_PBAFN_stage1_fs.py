import time
import yaml
from options.train_options import TrainOptions
from models.networks import VGGLoss,save_checkpoint
from models.afwm import TVLoss,AFWM
import torch.nn as nn
import argparse
import torch.nn.functional as F
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from VITON.Parser_Free.DM_VTON.train_pb_warp import process_opt
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
import cv2
import datetime
fix = lambda path: os.path.normpath(path)
opt,root_opt,wandb,sweep_id =None, None, None,None
def get_wandb_image(image, wandb):
    if image.max() <= 1.0:
        image = image*255 
    image_numpy = image.permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
    return wandb.Image(image_numpy)  

def CreateDataset(opt, root_opt):
    from VITON.Parser_Free.PF_AFN.PF_AFN_train.data.aligned_dataset import AlignedDataset
    dataset = AlignedDataset()
    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt, root_opt)
    return dataset

def train_fsvton_pb_warp_(opt_, root_opt_, run_wandb=False, sweep=None):
    global opt, root_opt, wandb,sweep_id
    opt,root_opt = process_opt(opt_, root_opt_)
    sweep_id = None
    if sweep is not None:
        import wandb 
        wandb.login()
        sweep_id = wandb.sweep(sweep=sweep, project="Fashion-NeRF-Sweep")
        wandb.agent(sweep_id,_train_fsvton_pb_warp_sweep,count=3)
    elif run_wandb:
        import wandb
        wandb.login()
        wandb.init(project="Fashion-NeRF", entity='rail_lab',  tags=[f"{root_opt.experiment_run}"], config=vars(opt))
        temp_opt = vars(opt)
        temp_opt['wandb_name'] = wandb.run.name
        opt = argparse.Namespace(**temp_opt)
        _train_fsvton_pb_warp_()
    else:
        wandb = None
        _train_fsvton_pb_warp_()


def _train_fsvton_pb_warp_sweep():
    if wandb is not None:
        with wandb.init(project="Fashion-NeRF-Sweep", entity='rail_lab',  tags=[f"{root_opt.experiment_run}"], config=vars(opt)):
            _train_fsvton_pb_warp_()
            
def select_device(device='', batch_size=0):
    cpu = device == 'cpu'
    if not cpu and torch.cuda.is_available():  # prefer GPU if available
        arg = f'cuda:{device}'
    else:  # revert to CPU
        arg = 'cpu'
    return arg

def load_checkpoint(model, checkpoint_path,opt=None):
    if not os.path.exists(checkpoint_path):
        print('no checkpoint')
        raise
    log = model.load_state_dict(torch.load(checkpoint_path), strict=False)
    if opt is None:
        model.cuda()
        
def split_dataset(dataset,train_size=0.8):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    train_indices, validation_indices = train_test_split(indices, train_size=train_size)
    train_subset = Subset(dataset, train_indices)
    validation_subset = Subset(dataset, validation_indices)
    return train_subset, validation_subset

def make_dirs(opt):
    if not os.path.exists(opt.results_dir):
        os.makedirs(opt.results_dir)
    if not os.path.exists(opt.pb_warp_save_step_checkpoint_dir):
        os.makedirs(opt.pb_warp_save_step_checkpoint_dir)
    if not os.path.exists(os.path.join(opt.results_dir, 'val')):
        os.makedirs(os.path.join(opt.results_dir, 'val'))
    if not os.path.exists(opt.pb_warp_save_final_checkpoint_dir):
        os.makedirs(opt.pb_warp_save_final_checkpoint_dir)
        
def _train_fsvton_pb_warp_():
    global opt, root_opt, wandb,sweep_id
    make_dirs(opt)
    device = select_device(opt.device, batch_size=root_opt.viton_batch_size)
    writer = SummaryWriter(opt.tensorboard_dir)
    if sweep_id is not None:
        opt.lambda_loss_second_smooth = wandb.config.lambda_loss_second_smooth
        opt.lambda_loss_vgg = wandb.config.lambda_loss_vgg
        opt.lambda_loss_vgg_skin = wandb.config.lambda_loss_vgg_skin
        opt.lambda_loss_edge = wandb.config.lambda_loss_edge
        opt.lambda_loss_smooth = wandb.config.lambda_loss_smooth
        opt.lambda_loss_l1 = wandb.config.lambda_loss_l1
        opt.lambda_bg_loss_l1 = wandb.config.lambda_bg_loss_l1
        opt.lambda_loss_warp = wandb.config.lambda_loss_warp
        opt.lambda_loss_gen = wandb.config.lambda_loss_gen
        opt.lambda_cond_sup_loss = wandb.config.lambda_cond_sup_loss
        opt.lambda_warp_sup_loss = wandb.config.lambda_warp_sup_loss
        opt.lambda_loss_l1_skin = wandb.config.lambda_loss_l1_skin
        opt.lambda_loss_l1_mask = wandb.config.lambda_loss_l1_mask
        opt.align_corners = wandb.config.align_corners
        opt.optimizer = wandb.config.optimizer
        opt.epsilon = wandb.config.epsilon
        opt.momentum = wandb.config.momentum
        opt.lr = wandb.config.lr
        opt.pb_gen_lr = wandb.config.pb_gen_lr

    experiment_string = f"{root_opt.experiment_run.replace('/','_')}_{root_opt.opt_vton_yaml.replace('yaml/','')}"
    with open(os.path.join(root_opt.experiment_run_yaml, experiment_string), 'w') as outfile:
        yaml.dump(vars(opt), outfile, default_flow_style=False)

    
    start_epoch, epoch_iter = 1, 0

    train_data = CreateDataset(opt, root_opt)
    train_dataset, validation_dataset = split_dataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=opt.viton_batch_size, shuffle=True,num_workers=root_opt.workers)
    validation_loader = DataLoader(validation_dataset, batch_size=opt.viton_batch_size, shuffle=True, num_workers=root_opt.workers)
    
    dataset_size = len(train_loader)
    print('#training images = %d' % dataset_size)

    warp_model = AFWM(opt, 45)
    print(warp_model)
    warp_model.train()
    warp_model.cuda()
    warp_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(warp_model).to(device)

    if opt.isTrain and len(opt.gpu_ids):
        model = torch.nn.parallel.DistributedDataParallel(warp_model, device_ids=[opt.local_rank])

    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()

    params_warp = [p for p in model.parameters()]
    optimizer_warp = torch.optim.Adam(params_warp, lr=opt.lr, betas=(opt.beta1, 0.999))

    total_steps = (start_epoch-1) * dataset_size + epoch_iter
    total_valid_steps = (start_epoch-1) * dataset_size + epoch_iter
    step = 0
    step_per_batch = dataset_size

    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size
        train_batch(opt, train_loader, 
                    warp_model,step, total_steps,epoch,criterionL1,criterionVGG,optimizer_warp,
                    writer, step_per_batch)    
        if epoch % opt.val_count == 0:
            validate_batch(opt, root_opt, validation_loader, 
                    warp_model,total_valid_steps,epoch,criterionL1,criterionVGG,
                    writer)
    save_checkpoint(warp_model, opt.pb_warp_save_final_checkpoint)
    

def train_batch(opt, train_loader, warp_model,step, total_steps, epoch,criterionL1,criterionVGG,optimizer_warp, writer, step_per_batch):
    warp_model.train()
    total_loss_warping = 0
    dataset_size = len(train_loader)
    for i, data in enumerate(train_loader):
        iter_start_time = time.time()

        total_steps += 1
        epoch_iter += 1

        t_mask = torch.FloatTensor((data['label'].cpu().numpy()==7).astype(np.float))
        data['label'] = data['label']*(1-t_mask)+t_mask*4
        edge = data['edge']
        pre_clothes_edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int))
        clothes = data['color']
        clothes = clothes * pre_clothes_edge
        person_clothes_edge = torch.FloatTensor((data['label'].cpu().numpy()==4).astype(np.int))
        real_image = data['image']
        person_clothes = real_image * person_clothes_edge
        pose = data['pose']
        pose_map = data['pose_map']
        size = data['label'].size()
        oneHot_size1 = (size[0], 25, size[2], size[3])
        densepose = torch.cuda.FloatTensor(torch.Size(oneHot_size1)).zero_()
        densepose = densepose.scatter_(1,data['densepose'].data.long().cuda(),1.0)
        densepose_fore = data['densepose']/24.0
        face_mask = torch.FloatTensor((data['label'].cpu().numpy()==1).astype(np.int)) + torch.FloatTensor((data['label'].cpu().numpy()==12).astype(np.int))
        other_clothes_mask = torch.FloatTensor((data['label'].cpu().numpy()==5).astype(np.int)) + torch.FloatTensor((data['label'].cpu().numpy()==6).astype(np.int)) + \
                            torch.FloatTensor((data['label'].cpu().numpy()==8).astype(np.int)) + torch.FloatTensor((data['label'].cpu().numpy()==9).astype(np.int)) + \
                            torch.FloatTensor((data['label'].cpu().numpy()==10).astype(np.int))
        preserve_mask = torch.cat([face_mask,other_clothes_mask],1)
        concat = torch.cat([preserve_mask.cuda(),densepose,pose.cuda()],1)

        #import ipdb; ipdb.set_trace()

        flow_out = warp_model(concat.cuda(), clothes.cuda(), pre_clothes_edge.cuda())
        warped_cloth, last_flow, _1, _2, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all = flow_out
        warped_prod_edge = x_edge_all[4]

        epsilon = opt.epsilon
        loss_smooth = sum([TVLoss(x) for x in delta_list])
        loss_all = 0

        for num in range(5):
            cur_person_clothes = F.interpolate(person_clothes, scale_factor=0.5**(4-num), mode='bilinear')
            cur_person_clothes_edge = F.interpolate(person_clothes_edge, scale_factor=0.5**(4-num), mode='bilinear')
            loss_l1 = criterionL1(x_all[num], cur_person_clothes.cuda())
            loss_vgg = criterionVGG(x_all[num], cur_person_clothes.cuda())
            loss_edge = criterionL1(x_edge_all[num], cur_person_clothes_edge.cuda())
            b,c,h,w = delta_x_all[num].shape
            loss_flow_x = (delta_x_all[num].pow(2)+ epsilon*epsilon).pow(0.45)
            loss_flow_x = torch.sum(loss_flow_x)/(b*c*h*w)
            loss_flow_y = (delta_y_all[num].pow(2)+ epsilon*epsilon).pow(0.45)
            loss_flow_y = torch.sum(loss_flow_y)/(b*c*h*w)
            loss_second_smooth = loss_flow_x + loss_flow_y
            loss_all = loss_all + (num+1) * loss_l1 + (num + 1) * 0.2 * loss_vgg + (num+1) * 2 * loss_edge + (num + 1) * 6 * loss_second_smooth

        loss_all = opt.lambda_loss_smooth * loss_smooth + loss_all


        optimizer_warp.zero_grad()
        loss_all.backward()
        optimizer_warp.step()
        ############## Display results and errors ##########

        total_loss_warping += loss_all.item()
        if (step + 1) % opt.display_count == 0:
            a = real_image.float().cuda()
            b = person_clothes.cuda()
            c = clothes.cuda()
            d = torch.cat([densepose_fore.cuda(),densepose_fore.cuda(),densepose_fore.cuda()],1)
            e = warped_cloth
            f = torch.cat([warped_prod_edge,warped_prod_edge,warped_prod_edge],1)
            combine = torch.cat([a[0],b[0],c[0],d[0],e[0],f[0]], 2).squeeze()
            cv_img=(combine.permute(1,2,0).detach().cpu().numpy()+1)/2
            writer.add_image('combine', (combine.data + 1) / 2.0, step)
            writer.add_scalar('loss_all', loss_all, step)
            writer.add_scalar('warping_l1', loss_l1, step)
            writer.add_scalar('warping_vgg', loss_vgg, step)
            writer.add_image('Real Image', a[0], 0)
            writer.add_image('Pose Image', (pose_map[0].cpu() / 2 + 0.5), 0)
            writer.add_image('Clothing', c[0], 0)
            writer.add_image('Parse Clothing', (b[0].cpu() / 2 + 0.5), 0)
            writer.add_image('Parse Clothing Mask', person_clothes_edge[0].cpu().expand(3, -1, -1), 0)
            writer.add_image('Warped Cloth', (e[0].cpu().detach() / 2 + 0.5), 0)
            writer.add_image('Warped Cloth Mask', (f[0].cpu().detach()).expand(3, -1, -1), 0)
            if wandb is not None:
                my_table = wandb.Table(columns=['Image', 'Pose Image','Clothing','Parse Clothing','Parse Clothing Mask','Warped Cloth','Warped Cloth Mask'])
                real_image_wandb = get_wandb_image(a[0], wandb=wandb) # 'Image'
                pose_image_wandb = get_wandb_image((pose_map[0].cpu() / 2 + 0.5),wandb) # 'Pose Image'
                clothing_image_wandb = get_wandb_image(c[0], wandb=wandb)#'Clothing'
                parse_clothing_wandb = get_wandb_image((b[0].cpu() / 2 + 0.5), wandb) # 'Parse Clothing'
                parse_clothing_mask_wandb = get_wandb_image(person_clothes_edge[0].cpu().expand(3, -1, -1), wandb) # 'Parse Clothing Mask'
                warped_cloth_wandb = get_wandb_image((e[0].cpu().detach() / 2 + 0.5), wandb) # 'Warped Cloth'
                warped_cloth_mask_wandb = get_wandb_image((f[0].cpu().detach()).expand(3, -1, -1), wandb) # 'Warped Cloth Mask'
                
                my_table.add_data(real_image_wandb, pose_image_wandb, clothing_image_wandb, parse_clothing_wandb, parse_clothing_mask_wandb, warped_cloth_wandb, warped_cloth_mask_wandb)
                
                wandb.log({'warping_loss': loss_all,'warping_l1':loss_l1,'warping_vgg':loss_vgg,'Table':my_table })
            rgb=(cv_img*255).astype(np.uint8)
            bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(opt.results_dir, str(step)+'.jpg'),bgr)

        step += 1
        iter_end_time = time.time()
        iter_delta_time = iter_end_time - iter_start_time
        step_delta = (step_per_batch-step%step_per_batch) + step_per_batch*(opt.niter + opt.niter_decay-epoch)
        eta = iter_delta_time*step_delta
        eta = str(datetime.timedelta(seconds=int(eta)))
        time_stamp = datetime.datetime.now()
        now = time_stamp.strftime('%Y.%m.%d-%H:%M:%S')
        if step % 100 == 0:
            if opt.local_rank == 0:
                print('{}:{}:[step-{}]--[loss-{:.6f}]--[ETA-{}]'.format(now, epoch_iter,step, loss_all,eta))

        if epoch_iter >= dataset_size:
            break
    
    # end of epoch 
    iter_end_time = time.time()

    ### save model for this epoch
    if epoch % opt.save_period == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
        save_checkpoint(warp_model.module, os.path.join(opt.checkpoints_dir, opt.name, 'PBAFN_warp_epoch_%03d.pth' % (epoch+1)))

    if epoch > opt.niter:
        warp_model.module.update_learning_rate(optimizer_warp)


def validate_batch(opt, root_opt, validation_loader, warp_model,total_steps, epoch,criterionL1,criterionVGG, writer):
    warp_model.eval()
    total_loss_warping = 0
    for i, data in enumerate(validation_loader):
        iter_start_time = time.time()

        total_steps += 1
        epoch_iter += 1

        t_mask = torch.FloatTensor((data['label'].cpu().numpy()==7).astype(np.float))
        data['label'] = data['label']*(1-t_mask)+t_mask*4
        edge = data['edge']
        pre_clothes_edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int))
        clothes = data['color']
        clothes = clothes * pre_clothes_edge
        person_clothes_edge = torch.FloatTensor((data['label'].cpu().numpy()==4).astype(np.int))
        real_image = data['image']
        person_clothes = real_image * person_clothes_edge
        pose = data['pose']
        pose_map = data['pose_map']
        size = data['label'].size()
        oneHot_size1 = (size[0], 25, size[2], size[3])
        densepose = torch.cuda.FloatTensor(torch.Size(oneHot_size1)).zero_()
        densepose = densepose.scatter_(1,data['densepose'].data.long().cuda(),1.0)
        densepose_fore = data['densepose']/24.0
        face_mask = torch.FloatTensor((data['label'].cpu().numpy()==1).astype(np.int)) + torch.FloatTensor((data['label'].cpu().numpy()==12).astype(np.int))
        other_clothes_mask = torch.FloatTensor((data['label'].cpu().numpy()==5).astype(np.int)) + torch.FloatTensor((data['label'].cpu().numpy()==6).astype(np.int)) + \
                            torch.FloatTensor((data['label'].cpu().numpy()==8).astype(np.int)) + torch.FloatTensor((data['label'].cpu().numpy()==9).astype(np.int)) + \
                            torch.FloatTensor((data['label'].cpu().numpy()==10).astype(np.int))
        preserve_mask = torch.cat([face_mask,other_clothes_mask],1)
        concat = torch.cat([preserve_mask.cuda(),densepose,pose.cuda()],1)

        #import ipdb; ipdb.set_trace()

        flow_out = warp_model(concat.cuda(), clothes.cuda(), pre_clothes_edge.cuda())
        warped_cloth, last_flow, _1, _2, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all = flow_out
        warped_prod_edge = x_edge_all[4]

        epsilon = opt.epsilon
        loss_smooth = sum([TVLoss(x) for x in delta_list])
        loss_all = 0

        for num in range(5):
            cur_person_clothes = F.interpolate(person_clothes, scale_factor=0.5**(4-num), mode='bilinear')
            cur_person_clothes_edge = F.interpolate(person_clothes_edge, scale_factor=0.5**(4-num), mode='bilinear')
            loss_l1 = criterionL1(x_all[num], cur_person_clothes.cuda())
            loss_vgg = criterionVGG(x_all[num], cur_person_clothes.cuda())
            loss_edge = criterionL1(x_edge_all[num], cur_person_clothes_edge.cuda())
            b,c,h,w = delta_x_all[num].shape
            loss_flow_x = (delta_x_all[num].pow(2)+ epsilon*epsilon).pow(0.45)
            loss_flow_x = torch.sum(loss_flow_x)/(b*c*h*w)
            loss_flow_y = (delta_y_all[num].pow(2)+ epsilon*epsilon).pow(0.45)
            loss_flow_y = torch.sum(loss_flow_y)/(b*c*h*w)
            loss_second_smooth = loss_flow_x + loss_flow_y
            loss_all = loss_all + (num+1) * loss_l1 + (num + 1) * 0.2 * loss_vgg + (num+1) * 2 * loss_edge + (num + 1) * 6 * loss_second_smooth

        loss_all = opt.lambda_loss_smooth * loss_smooth + loss_all


        ############## Display results and errors ##########

        total_loss_warping += loss_all.item()
        if wandb is not None:
            a = real_image.float().cuda()
            b = person_clothes.cuda()
            c = clothes.cuda()
            d = torch.cat([densepose_fore.cuda(),densepose_fore.cuda(),densepose_fore.cuda()],1)
            e = warped_cloth
            f = torch.cat([warped_prod_edge,warped_prod_edge,warped_prod_edge],1)
            combine = torch.cat([a[0],b[0],c[0],d[0],e[0],f[0]], 2).squeeze()
            cv_img=(combine.permute(1,2,0).detach().cpu().numpy()+1)/2
            writer.add_image('combine', (combine.data + 1) / 2.0, epoch)
            writer.add_scalar('Val/loss_all', loss_all, epoch)
            writer.add_scalar('Val/warping_l1', loss_l1, epoch)
            writer.add_scalar('Val/warping_vgg', loss_vgg, epoch)
            writer.add_image('Val/Real Image', a[0], 0)
            writer.add_image('Val/Pose Image', (pose_map[0].cpu() / 2 + 0.5), 0)
            writer.add_image('Val/Clothing', c[0], 0)
            writer.add_image('Val/Parse Clothing', (b[0].cpu() / 2 + 0.5), 0)
            writer.add_image('Val/Parse Clothing Mask', person_clothes_edge[0].cpu().expand(3, -1, -1), 0)
            writer.add_image('Val/Warped Cloth', (e[0].cpu().detach() / 2 + 0.5), 0)
            writer.add_image('Val/Warped Cloth Mask', (f[0].cpu().detach()).expand(3, -1, -1), 0)
            if wandb is not None:
                my_table = wandb.Table(columns=['Image', 'Pose Image','Clothing','Parse Clothing','Parse Clothing Mask','Warped Cloth','Warped Cloth Mask'])
                real_image_wandb = get_wandb_image(a[0], wandb=wandb) # 'Image'
                pose_image_wandb = get_wandb_image((pose_map[0].cpu() / 2 + 0.5),wandb) # 'Pose Image'
                clothing_image_wandb = get_wandb_image(c[0], wandb=wandb)#'Clothing'
                parse_clothing_wandb = get_wandb_image((b[0].cpu() / 2 + 0.5), wandb) # 'Parse Clothing'
                parse_clothing_mask_wandb = get_wandb_image(person_clothes_edge[0].cpu().expand(3, -1, -1), wandb) # 'Parse Clothing Mask'
                warped_cloth_wandb = get_wandb_image((e[0].cpu().detach() / 2 + 0.5), wandb) # 'Warped Cloth'
                warped_cloth_mask_wandb = get_wandb_image((f[0].cpu().detach()).expand(3, -1, -1), wandb) # 'Warped Cloth Mask'
                
                my_table.add_data(real_image_wandb, pose_image_wandb, clothing_image_wandb, parse_clothing_wandb, parse_clothing_mask_wandb, warped_cloth_wandb, warped_cloth_mask_wandb)
                
                wandb.log({'val_warping_loss': loss_all,'val_warping_l1':loss_l1,'val_warping_vgg':loss_vgg,'Table':my_table })
            rgb=(cv_img*255).astype(np.uint8)
            bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(opt.results_dir, str(epoch)+'.jpg'),bgr)
