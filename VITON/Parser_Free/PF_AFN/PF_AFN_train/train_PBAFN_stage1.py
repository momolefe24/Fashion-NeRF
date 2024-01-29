import time
from VITON.Parser_Free.PF_AFN.PF_AFN_train.models.networks import VGGLoss,save_checkpoint, load_checkpoint_part_parallel, load_checkpoint_parallel
import argparse
import yaml
from VITON.Parser_Free.PF_AFN.PF_AFN_train.models.afwm import TVLoss,AFWM
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
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

def get_root_experiment_runs(root_opt):
    root_opt.experiment_run = root_opt.experiment_run.format(root_opt.experiment_number, root_opt.run_number)
    root_opt.experiment_from_run = root_opt.experiment_from_run.format(root_opt.experiment_from_number, root_opt.run_from_number)
    return root_opt


def get_root_opt_experiment_dir(root_opt):
    root_opt.rail_dir = root_opt.rail_dir.format(root_opt.dataset_name, root_opt.res, root_opt.datamode)
    root_opt.original_dir = root_opt.original_dir.format(root_opt.dataset_name, root_opt.res, root_opt.datamode)
    if root_opt.res == 'low_res':
        root_opt.original_dir = root_opt.original_dir.replace(root_opt.res, os.path.join(root_opt.res, root_opt.low_res_dataset_name))
    
    # Parser Based Warping
    root_opt.this_viton_save_to_dir = os.path.join(root_opt.this_viton_save_to_dir, root_opt.VITON_Model)
    root_opt.this_viton_load_from_dir = root_opt.this_viton_load_from_dir.format(root_opt.VITON_Type, root_opt.VITON_Name, root_opt.this_viton_load_from_dir)
    root_opt.this_viton_load_from_dir = os.path.join(root_opt.this_viton_load_from_dir, root_opt.VITON_Model)
    
    root_opt.warp_experiment_from_dir = root_opt.warp_experiment_from_dir.format(root_opt.VITON_Type, root_opt.VITON_Name, root_opt.warp_load_from_model)
    root_opt.warp_experiment_from_dir = os.path.join(root_opt.warp_experiment_from_dir, root_opt.VITON_Model)
    return root_opt

def get_root_opt_checkpoint_dir(parser, root_opt):
    last_step = root_opt.load_last_step if type(root_opt.load_last_step) == bool else eval(root_opt.load_last_step)
    sort_digit = lambda name: int(name.split('_')[-1].split('.')[0])
    parser.pb_warp_save_step_checkpoint_dir = parser.pb_warp_save_step_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    parser.pb_warp_save_step_checkpoint = os.path.join(parser.pb_warp_save_step_checkpoint_dir, parser.pb_warp_save_step_checkpoint)
    parser.pb_warp_save_step_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pb_warp_save_step_checkpoint)
    parser.pb_warp_save_step_checkpoint_dir = os.path.join("/",*parser.pb_warp_save_step_checkpoint.split("/")[:-1])
    

    parser.pb_warp_save_final_checkpoint_dir = parser.pb_warp_save_final_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    parser.pb_warp_save_final_checkpoint = os.path.join(parser.pb_warp_save_final_checkpoint_dir, parser.pb_warp_save_final_checkpoint)
    parser.pb_warp_save_final_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pb_warp_save_final_checkpoint)
    parser.pb_warp_save_final_checkpoint_dir = os.path.join("/",*parser.pb_warp_save_final_checkpoint.split("/")[:-1])
    
    
    parser.pb_warp_load_final_checkpoint_dir = parser.pb_warp_load_final_checkpoint_dir.format(root_opt.experiment_from_run, root_opt.warp_experiment_from_dir)
    parser.pb_warp_load_final_checkpoint = os.path.join(parser.pb_warp_load_final_checkpoint_dir, parser.pb_warp_load_final_checkpoint)
    parser.pb_warp_load_final_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pb_warp_load_final_checkpoint)
    parser.pb_warp_load_final_checkpoint_dir = os.path.join("/",*parser.pb_warp_load_final_checkpoint.split("/")[:-1])
    
    
    if not last_step:
        parser.pb_warp_load_step_checkpoint_dir = parser.pb_warp_load_step_checkpoint_dir.format(root_opt.experiment_from_run, root_opt.warp_experiment_from_dir)
    else:
        parser.pb_warp_load_step_checkpoint_dir = parser.pb_warp_load_step_checkpoint_dir.format(root_opt.experiment_from_run, root_opt.this_viton_save_to_dir)
    parser.pb_warp_load_step_checkpoint_dir = fix(parser.pb_warp_load_step_checkpoint_dir)
    if not last_step:
        parser.pb_warp_load_step_checkpoint = os.path.join(parser.pb_warp_load_step_checkpoint_dir, parser.pb_warp_load_step_checkpoint)
    else:
        if os.path.isdir(parser.pb_warp_load_step_checkpoint_dir.format(root_opt.experiment_from_run, root_opt.this_viton_save_to_dir)):
            os_list = os.listdir(parser.pb_warp_load_step_checkpoint_dir.format(root_opt.experiment_from_run, root_opt.this_viton_save_to_dir))
            os_list = [string for string in os_list if "pb_warp" in string]
            last_step = sorted(os_list, key=sort_digit)[-1]
            parser.pb_warp_load_step_checkpoint = os.path.join(parser.pb_warp_load_step_checkpoint_dir, last_step)
    parser.pb_warp_load_step_checkpoint = fix(parser.checkpoint_root_dir + "/" + parser.pb_warp_load_step_checkpoint)
    parser.pb_warp_load_step_checkpoint_dir = os.path.join("/", *parser.pb_warp_load_step_checkpoint.split("/")[:-1])
    return parser 

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
    parser.datamode = root_opt.datamode
    parser.load_last_step = root_opt.load_last_step if type(root_opt.load_last_step) == bool else eval(root_opt.load_last_step)
    parser.run_wandb = root_opt.run_wandb
    parser.viton_batch_size = root_opt.viton_batch_size
    parser.save_period = root_opt.save_period
    parser.seed = root_opt.seed
    parser.val_count = root_opt.val_count
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

def CreateDataset(opt, root_opt):
    from VITON.Parser_Free.PF_AFN.PF_AFN_train.data.aligned_dataset import AlignedDataset
    dataset = AlignedDataset()
    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt, root_opt)
    return dataset


def train_pfafn_pb_warp_(opt_, root_opt_, run_wandb=False, sweep=None):
    global opt, root_opt, wandb,sweep_id
    opt,root_opt = process_opt(opt_, root_opt_)
    sweep_id = None
    if sweep is not None:
        import wandb 
        wandb.login()
        sweep_id = wandb.sweep(sweep=sweep, project="Fashion-NeRF-Sweep")
        wandb.agent(sweep_id,_train_pfafn_pb_warp_sweep,count=5)
    elif run_wandb:
        import wandb
        wandb.login()
        wandb.init(project="Fashion-NeRF", entity='prime_lab', notes=f"question: {opt.question}, intent: {opt.intent}", tags=[f"{root_opt.experiment_run}"], config=vars(opt))
        temp_opt = vars(opt)
        temp_opt['wandb_name'] = wandb.run.name
        opt = argparse.Namespace(**temp_opt)
        _train_pfafn_pb_warp_()
    else:
        wandb = None
        _train_pfafn_pb_warp_()


def _train_pfafn_pb_warp_sweep():
    if wandb is not None:
        with wandb.init(project="Fashion-NeRF-Sweep", entity='prime_lab', notes=f"question: {opt.question}, intent: {opt.intent}", tags=[f"{root_opt.experiment_run}"], config=vars(opt)):
            _train_pfafn_pb_warp_()
            
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

def _train_pfafn_pb_warp_():
    global opt, root_opt, wandb,sweep_id
    if sweep_id is not None:
        opt = wandb.config
        
    experiment_string = f"{root_opt.experiment_run.replace('/','_')}_{root_opt.opt_vton_yaml.replace('yaml/','')}"
    with open(os.path.join(root_opt.experiment_run_yaml, experiment_string), 'w') as outfile:
        yaml.dump(vars(opt), outfile, default_flow_style=False)
    device = select_device(opt.device, batch_size=root_opt.viton_batch_size)

    
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
    if os.path.exists(opt.pb_warp_load_final_checkpoint_dir):
        load_checkpoint_part_parallel(opt, warp_model, opt.pb_warp_load_final_checkpoint)

    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()

    params_warp = [p for p in warp_model.parameters()]
    optimizer_warp = torch.optim.Adam(params_warp, lr=opt.lr, betas=(opt.beta1, 0.999))

    total_steps = (start_epoch-1) * dataset_size + epoch_iter
    total_valid_steps = (start_epoch-1) * dataset_size + epoch_iter
    step = 0
    step_per_batch = dataset_size
    
    writer = SummaryWriter(opt.tensorboard_dir)

    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size
        train_batch(opt, root_opt, train_loader, 
                    warp_model,total_steps,epoch,criterionL1,criterionVGG,optimizer_warp,
                    writer, step_per_batch,epoch_start_time)

        if epoch % opt.val_count == 0:
            validate_batch(opt, root_opt, validation_loader, 
                    warp_model,total_valid_steps,epoch,criterionL1,criterionVGG,
                    writer)
    
    if not os.path.exists(opt.pb_warp_save_final_checkpoint_dir):
        os.makedirs(opt.pb_warp_save_final_checkpoint_dir)
    save_checkpoint(warp_model, opt.pb_warp_save_final_checkpoint)
    
def validate_batch(opt, root_opt, validation_loader, warp_model,total_steps, epoch,criterionL1,criterionVGG, writer):
    warp_model.eval()
    total_loss_warping = 0
    for i, data in enumerate(validation_loader):
        total_steps += 1
    
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

        flow_out = warp_model(concat.cuda(), clothes.cuda(), pre_clothes_edge.cuda())
        warped_cloth, last_flow, _1, _2, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all = flow_out
        warped_prod_edge = x_edge_all[4]
        
        if root_opt.dataset_name == 'Rail' and epoch >0 :
            binary_mask = (warped_prod_edge > 0.5).float()
            warped_cloth = warped_cloth * binary_mask

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
        total_loss_warping += loss_all.item()
    
        writer.add_scalar('val_loss_all', loss_all, epoch)
        writer.add_scalar('val_loss_l1', loss_l1, epoch)
        writer.add_scalar('val_loss_vgg', loss_vgg, epoch)

        ############## Display results and errors ##########
        a = real_image.float().cuda()
        b = person_clothes.cuda()
        c = clothes.cuda()
        d = torch.cat([densepose_fore.cuda(),densepose_fore.cuda(),densepose_fore.cuda()],1)
        e = warped_cloth
        f = torch.cat([warped_prod_edge,warped_prod_edge,warped_prod_edge],1)
        combine = torch.cat([a[0],b[0],c[0],d[0],e[0],f[0]], 2).squeeze()
        cv_img=(combine.permute(1,2,0).detach().cpu().numpy()+1)/2
        writer.add_image('combine', (combine.data + 1) / 2.0, epoch)
        rgb=(cv_img*255).astype(np.uint8)
        if wandb is not None:
            my_table = wandb.Table(columns=['Combined Image','Real Image', 'Person Clothes' ,'Clothing','Warped Clothing', 'Warped Clothing Mask'])
            real_image_wandb = get_wandb_image(a[0], wandb=wandb)
            person_clothing_image_wandb = get_wandb_image(b[0], wandb=wandb)
            clothing_image_wandb = get_wandb_image(c[0], wandb=wandb)
            warped_wandb = get_wandb_image(e[0], wandb=wandb)
            warped_mask_wandb = get_wandb_image(f[0], wandb=wandb)
            my_table.add_data(wandb.Image((rgb).astype(np.uint8)), real_image_wandb,person_clothing_image_wandb,clothing_image_wandb,warped_wandb,warped_mask_wandb)
            wandb.log({'val_warping_loss': loss_all,'val_l1_cloth':loss_l1,'val_loss_vgg':loss_vgg,'Val_Table':my_table })
        bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        if not os.path.exists(os.path.join(opt.results_dir, 'val')):
            os.makedirs(os.path.join(opt.results_dir, 'val'))
        cv2.imwrite(os.path.join(opt.results_dir, 'val', f"{epoch}.jpg"),bgr)
    avg_total_loss_G = total_loss_warping / len(validation_loader)
    if wandb is not None:
        wandb.log({"val_total_avg_warping_loss":avg_total_loss_G})
      
def train_batch(opt, root_opt, train_loader, warp_model,total_steps, epoch,criterionL1,criterionVGG,optimizer_warp, writer, step_per_batch,epoch_start_time):
    warp_model.train()
    total_loss_warping = 0
    dataset_size = len(train_loader)
    with torch.no_grad():
        for i, data in enumerate(train_loader):
            iter_start_time = time.time()

            total_steps += 1
        
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

            flow_out = warp_model(concat.cuda(), clothes.cuda(), pre_clothes_edge.cuda())
            warped_cloth, last_flow, _1, _2, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all = flow_out
            warped_prod_edge = x_edge_all[4]
            
            if root_opt.dataset_name == 'Rail' and epoch >0 :
                binary_mask = (warped_prod_edge > 0.5).float()
                warped_cloth = warped_cloth * binary_mask

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
            total_loss_warping += loss_all.item()
        
            writer.add_scalar('loss_all', loss_all, step)

            optimizer_warp.zero_grad()
            loss_all.backward()
            optimizer_warp.step()
            ############## Display results and errors ##########
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
                rgb=(cv_img*255).astype(np.uint8)
                if wandb is not None:
                    my_table = wandb.Table(columns=['Combined Image','Real Image', 'Person Clothes' ,'Clothing','Warped Clothing', 'Warped Clothing Mask'])
                    real_image_wandb = get_wandb_image(a[0], wandb=wandb)
                    person_clothing_image_wandb = get_wandb_image(b[0], wandb=wandb)
                    clothing_image_wandb = get_wandb_image(c[0], wandb=wandb)
                    warped_wandb = get_wandb_image(e[0], wandb=wandb)
                    warped_mask_wandb = get_wandb_image(f[0], wandb=wandb)
                    my_table.add_data(wandb.Image((rgb).astype(np.uint8)), real_image_wandb,person_clothing_image_wandb,clothing_image_wandb,warped_wandb,warped_mask_wandb)
                    wandb.log({'warping_loss': loss_all,'l1_cloth':loss_l1,'loss_vgg':loss_vgg,'Table':my_table })
                bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(opt.results_dir, f"{step}.jpg"),bgr)

            step += 1
            iter_end_time = time.time()
            iter_delta_time = iter_end_time - iter_start_time
            step_delta = (step_per_batch-step%step_per_batch) + step_per_batch*(opt.niter + opt.niter_decay-epoch)
            eta = iter_delta_time*step_delta
            eta = str(datetime.timedelta(seconds=int(eta)))
            time_stamp = datetime.datetime.now()
            now = time_stamp.strftime('%Y.%m.%d-%H:%M:%S')
            if step % opt.print_step == 0:
                print('{}:{}:[step-{}]--[loss-{:.6f}]--[ETA-{}]'.format(now, epoch,step, loss_all,eta))

            if epoch >= dataset_size:
                break
        # break
    # end of epoch 
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
                (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_period == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
        if not os.path.exists(opt.pb_warp_save_step_checkpoint_dir):
            os.makedirs(opt.pb_warp_save_step_checkpoint_dir)
        save_checkpoint(warp_model, opt.pb_warp_save_step_checkpoint % (epoch+1))

    if epoch > opt.niter:
        warp_model.update_learning_rate(optimizer_warp)
        
    avg_total_loss_G = total_loss_warping / len(train_loader)
    if wandb is not None:
        wandb.log({"total_avg_warping_loss":avg_total_loss_G})