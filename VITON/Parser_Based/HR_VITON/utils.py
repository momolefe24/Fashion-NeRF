import torch
import argparse
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import numpy as np
import cv2
import os
fix = lambda path: os.path.normpath(path)

def get_clothes_mask(old_label) :
    clothes = torch.FloatTensor((old_label.cpu().numpy() == 3).astype(np.int))
    return clothes

def changearm(old_label):
    label=old_label
    arm1=torch.FloatTensor((old_label.cpu().numpy()==5).astype(np.int))
    arm2=torch.FloatTensor((old_label.cpu().numpy()==6).astype(np.int))
    label=label*(1-arm1)+arm1*3
    label=label*(1-arm2)+arm2*3
    return label

def gen_noise(shape):
    noise = np.zeros(shape, dtype=np.uint8)
    ### noise
    noise = cv2.randn(noise, 0, 255)
    noise = np.asarray(noise / 255, dtype=np.uint8)
    noise = torch.tensor(noise, dtype=torch.float32)
    return noise

def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht or w != wt:
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=weight, size_average=size_average, ignore_index=250
    )
    return loss

def ndim_tensor2im(image_tensor, imtype=np.uint8, batch=0):
    image_numpy = image_tensor[batch].cpu().float().numpy()
    result = np.argmax(image_numpy, axis=0)
    return result.astype(imtype)

def visualize_segmap(input, multi_channel=True, tensor_out=True, batch=0) :
    palette = [
        0, 0, 0, 128, 0, 0, 254, 0, 0, 0, 85, 0, 169, 0, 51,
        254, 85, 0, 0, 0, 85, 0, 119, 220, 85, 85, 0, 0, 85, 85,
        85, 51, 0, 52, 86, 128, 0, 128, 0, 0, 0, 254, 51, 169, 220,
        0, 254, 254, 85, 254, 169, 169, 254, 85, 254, 254, 0, 254, 169, 0
    ]
    input = input.detach()
    if multi_channel :
        input = ndim_tensor2im(input,batch=batch)
    else :
        input = input[batch][0].cpu()
        input = np.asarray(input)
        input = input.astype(np.uint8)
    input = Image.fromarray(input, 'P')
    input.putpalette(palette)

    if tensor_out :
        trans = transforms.ToTensor()
        return trans(input.convert('RGB'))

    return input

def pred_to_onehot(prediction) :
    size = prediction.shape
    prediction_max = torch.argmax(prediction, dim=1)
    oneHot_size = (size[0], 13, size[2], size[3])
    pred_onehot = torch.FloatTensor(torch.Size(oneHot_size)).zero_()
    pred_onehot = pred_onehot.scatter_(1, prediction_max.unsqueeze(1).data.long(), 1.0)
    return pred_onehot

def cal_miou(prediction, target) :
    size = prediction.shape
    target = target.cpu()
    prediction = pred_to_onehot(prediction.detach().cpu())
    list = [1,2,3,4,5,6,7,8]
    union = 0
    intersection = 0
    for b in range(size[0]) :
        for c in list :
            intersection += torch.logical_and(target[b,c], prediction[b,c]).sum()
            union += torch.logical_or(target[b,c], prediction[b,c]).sum()
    return intersection.item()/union.item()

def save_images(img_tensors, img_names, save_dir):
    for img_tensor, img_name in zip(img_tensors, img_names):
        tensor = (img_tensor.clone() + 1) * 0.5 * 255
        tensor = tensor.cpu().clamp(0, 255)

        try:
            array = tensor.numpy().astype('uint8')
        except:
            array = tensor.detach().numpy().astype('uint8')

        if array.shape[0] == 1:
            array = array.squeeze(0)
        elif array.shape[0] == 3:
            array = array.swapaxes(0, 1).swapaxes(1, 2)

        im = Image.fromarray(array)
        im.save(os.path.join(save_dir, img_name), format='JPEG')
        
        
def create_network(cls, opt):
    net = cls(opt)
    net.print_network()
    if len(opt.gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda()
    net.init_weights(opt.init_type, opt.init_variance)
    return net


def condition_get_root_experiment_runs(root_opt):
    root_opt.experiment_run = root_opt.experiment_run.format(root_opt.experiment_number, root_opt.run_number)
    root_opt.experiment_from_run = root_opt.experiment_from_run.format(root_opt.experiment_from_number, root_opt.run_from_number)
    root_opt.tocg_experiment_from_run = root_opt.tocg_experiment_from_run.format(root_opt.tocg_experiment_from_number, root_opt.tocg_run_from_number)
    root_opt.tocg_discriminator_experiment_from_run = root_opt.tocg_discriminator_experiment_from_run.format(root_opt.tocg_discriminator_experiment_from_number, root_opt.tocg_discriminator_run_from_number)
    return root_opt

def condition_get_root_opt_experiment_dir(root_opt):
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


def condition_get_root_opt_results_dir(parser, root_opt):
    root_opt.transforms_dir = root_opt.transforms_dir.format(root_opt.dataset_name)
    parser.tensorboard_dir = parser.tensorboard_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    parser.results_dir = parser.results_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    return parser, root_opt

def condition_copy_root_opt_to_opt(parser, root_opt):
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
    parser.datamode = root_opt.datamode
    parser.niter = root_opt.niter
    parser.niter_decay = root_opt.niter_decay
    parser.VITON_Type = root_opt.VITON_Type
    parser.VITON_selection_dir = parser.VITON_selection_dir.format(parser.VITON_Type, parser.VITON_Name)
    return parser

def condition_get_root_opt_checkpoint_dir(opt, root_opt):
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

def condition_get_root_opt_results_dir(parser, root_opt):
    root_opt.transforms_dir = root_opt.transforms_dir.format(root_opt.dataset_name)
    parser.tensorboard_dir = parser.tensorboard_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    parser.results_dir = parser.results_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    return parser, root_opt

def condition_process_opt(opt, root_opt):
    parser = opt
    parser = argparse.Namespace(**parser)
    root_opt = condition_get_root_experiment_runs(root_opt)
    root_opt = condition_get_root_opt_experiment_dir(root_opt)
    parser = condition_get_root_opt_checkpoint_dir(parser, root_opt)
    parser, root_opt = condition_get_root_opt_results_dir(parser, root_opt)    
    parser = condition_copy_root_opt_to_opt(parser, root_opt)
    return parser, root_opt




def generator_get_root_experiment_runs(root_opt):
    root_opt.experiment_run = root_opt.experiment_run.format(root_opt.experiment_number, root_opt.run_number)
    root_opt.experiment_from_run = root_opt.experiment_from_run.format(root_opt.experiment_from_number, root_opt.run_from_number)
    
    root_opt.tocg_experiment_from_run = root_opt.tocg_experiment_from_run.format(root_opt.tocg_experiment_from_number, root_opt.tocg_run_from_number)
    root_opt.tocg_discriminator_experiment_from_run = root_opt.tocg_discriminator_experiment_from_run.format(root_opt.tocg_discriminator_experiment_from_number, root_opt.tocg_discriminator_run_from_number)
    
    root_opt.gen_experiment_from_run = root_opt.gen_experiment_from_run.format(root_opt.gen_experiment_from_number, root_opt.gen_run_from_number)
    root_opt.gen_discriminator_experiment_from_run = root_opt.gen_discriminator_experiment_from_run.format(root_opt.gen_discriminator_experiment_from_number, root_opt.gen_discriminator_run_from_number)
    return root_opt

def generator_get_root_opt_experiment_dir(root_opt):
    root_opt.rail_dir = root_opt.rail_dir.format(root_opt.dataset_name, root_opt.res, root_opt.datamode)    
    root_opt.original_dir = root_opt.original_dir.format(root_opt.dataset_name, root_opt.res, root_opt.datamode)
    if root_opt.res == 'low_res':
        root_opt.original_dir = root_opt.original_dir.replace(root_opt.res, os.path.join(root_opt.res, root_opt.low_res_dataset_name))
    # Current model
    root_opt.this_viton_save_to_dir = os.path.join(root_opt.this_viton_save_to_dir, root_opt.VITON_Model)
    root_opt.this_viton_load_from_dir = root_opt.this_viton_load_from_dir.format(root_opt.VITON_Type, root_opt.VITON_Name, root_opt.this_viton_load_from_dir)
    root_opt.this_viton_load_from_dir = os.path.join(root_opt.this_viton_load_from_dir, root_opt.VITON_Model)
    
    # tocg
    root_opt.tocg_experiment_from_dir = root_opt.tocg_experiment_from_dir.format(root_opt.VITON_Type, root_opt.VITON_Name, root_opt.tocg_load_from_model)
    root_opt.tocg_experiment_from_dir = os.path.join(root_opt.tocg_experiment_from_dir, 'TOCG')
    
    # tocg discriminator
    root_opt.tocg_discriminator_experiment_from_dir = root_opt.tocg_discriminator_experiment_from_dir.format(root_opt.VITON_Type, root_opt.VITON_Name, root_opt.tocg_discriminator_load_from_model)
    root_opt.tocg_discriminator_experiment_from_dir = os.path.join(root_opt.tocg_discriminator_experiment_from_dir, root_opt.VITON_Model)    
    
    
    # gen
    root_opt.gen_experiment_from_dir = root_opt.gen_experiment_from_dir.format(root_opt.VITON_Type, root_opt.VITON_Name, root_opt.gen_load_from_model)
    root_opt.gen_experiment_from_dir = os.path.join(root_opt.gen_experiment_from_dir, root_opt.VITON_Model)
    
    # gen discriminator
    root_opt.gen_discriminator_experiment_from_dir = root_opt.gen_discriminator_experiment_from_dir.format(root_opt.VITON_Type, root_opt.VITON_Name, root_opt.gen_discriminator_load_from_model)
    root_opt.gen_discriminator_experiment_from_dir = os.path.join(root_opt.gen_discriminator_experiment_from_dir, root_opt.VITON_Model)    
    
    return root_opt


def generator_get_root_opt_results_dir(parser, root_opt):
    root_opt.transforms_dir = root_opt.transforms_dir.format(root_opt.dataset_name)
    parser.tensorboard_dir = parser.tensorboard_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    parser.results_dir = parser.results_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    return parser, root_opt

def generator_copy_root_opt_to_opt(parser, root_opt):
    parser.display_count = root_opt.display_count
    parser.cuda = root_opt.cuda
    parser.device = int(root_opt.device)
    parser.dataset_name = root_opt.dataset_name
    parser.warp_load_from_model = root_opt.warp_load_from_model
    parser.load_last_step = root_opt.load_last_step if type(root_opt.load_last_step) == bool else eval(root_opt.load_last_step)
    parser.run_wandb = root_opt.run_wandb
    parser.viton_batch_size = root_opt.viton_batch_size
    parser.save_period = root_opt.save_period
    parser.seed = root_opt.seed
    parser.print_step = root_opt.print_step
    parser.niter = root_opt.niter
    parser.niter_decay = root_opt.niter_decay
    parser.VITON_Type = root_opt.VITON_Type
    parser.VITON_selection_dir = parser.VITON_selection_dir.format(parser.VITON_Type, parser.VITON_Name)
    return parser

def generator_get_root_opt_checkpoint_dir(opt, root_opt):
    last_step = root_opt.load_last_step if type(root_opt.load_last_step) == bool else eval(root_opt.load_last_step)
    sort_digit = lambda name: int(name.split('_')[-1].split('.')[0])
    # ================================= tocg =================================
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
    
    if not last_step:
        opt.tocg_load_step_checkpoint_dir = opt.tocg_load_step_checkpoint_dir.format(root_opt.tocg_experiment_from_run, root_opt.tocg_experiment_from_dir)
    else:
        opt.tocg_load_step_checkpoint_dir = opt.tocg_load_step_checkpoint_dir.format(root_opt.tocg_experiment_from_run, root_opt.this_viton_save_to_dir)
    opt.tocg_load_step_checkpoint_dir = fix(opt.tocg_load_step_checkpoint_dir)
    if not last_step:
        opt.tocg_load_step_checkpoint = os.path.join(opt.tocg_load_step_checkpoint_dir, opt.tocg_load_step_checkpoint)
    else:
        if os.path.isdir(opt.tocg_load_step_checkpoint_dir.format(root_opt.tocg_experiment_from_run, root_opt.this_viton_save_to_dir)):
            os_list = os.listdir(opt.tocg_load_step_checkpoint_dir.format(root_opt.tocg_experiment_from_run, root_opt.this_viton_save_to_dir))
            os_list = [string for string in os_list if "tocg" in string]
            last_step = sorted(os_list, key=sort_digit)[-1]
            opt.tocg_load_step_checkpoint = os.path.join(opt.tocg_load_step_checkpoint_dir, last_step)
    opt.tocg_load_step_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.tocg_load_step_checkpoint)
    opt.tocg_load_step_checkpoint_dir = os.path.join("/",*opt.tocg_load_step_checkpoint.split("/")[:-1])
    # ================================= tocg DISCRIMINATOR =================================
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
    
    
    opt.tocg_discriminator_load_final_checkpoint_dir = opt.tocg_discriminator_load_final_checkpoint_dir.format(root_opt.tocg_experiment_from_run, root_opt.tocg_discriminator_experiment_from_dir)
    opt.tocg_discriminator_load_final_checkpoint_dir = fix(opt.tocg_discriminator_load_final_checkpoint_dir)
    opt.tocg_discriminator_load_final_checkpoint = os.path.join(opt.tocg_discriminator_load_final_checkpoint_dir, opt.tocg_discriminator_load_final_checkpoint)
    opt.tocg_discriminator_load_final_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.tocg_discriminator_load_final_checkpoint)
    opt.tocg_discriminator_load_final_checkpoint_dir = os.path.join("/",*opt.tocg_discriminator_load_final_checkpoint.split("/")[:-1])

    if not last_step:
        opt.tocg_discriminator_load_step_checkpoint_dir = opt.tocg_discriminator_load_step_checkpoint_dir.format(root_opt.tocg_discriminator_experiment_from_run, root_opt.tocg_discriminator_experiment_from_dir)
    else:
        opt.tocg_discriminator_load_step_checkpoint_dir = opt.tocg_discriminator_load_step_checkpoint_dir.format(root_opt.tocg_discriminator_experiment_from_run, root_opt.this_viton_save_to_dir)
    opt.tocg_discriminator_load_step_checkpoint_dir = fix(opt.tocg_discriminator_load_step_checkpoint_dir)
    if not last_step:
        opt.tocg_discriminator_load_step_checkpoint = os.path.join(opt.tocg_discriminator_load_step_checkpoint_dir, opt.tocg_discriminator_load_step_checkpoint)
    else:
        if os.path.isdir(opt.tocg_discriminator_load_step_checkpoint_dir.format(root_opt.tocg_discriminator_experiment_from_run, root_opt.this_viton_save_to_dir)):
            os_list = os.listdir(opt.tocg_discriminator_load_step_checkpoint_dir.format(root_opt.tocg_discriminator_experiment_from_run, root_opt.this_viton_save_to_dir))
            os_list = [string for string in os_list if "tocg_discriminator" in string]
            last_step = sorted(os_list, key=sort_digit)[-1]
            opt.tocg_discriminator_load_step_checkpoint = os.path.join(opt.tocg_discriminator_load_step_checkpoint_dir, last_step)
    opt.tocg_discriminator_load_step_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.tocg_discriminator_load_step_checkpoint)
    opt.tocg_discriminator_load_step_checkpoint_dir = os.path.join("/",*opt.tocg_discriminator_load_step_checkpoint.split("/")[:-1])
    # ================================= gen =================================
    opt.gen_save_step_checkpoint_dir = opt.gen_save_step_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    opt.gen_save_step_checkpoint_dir = fix(opt.gen_save_step_checkpoint_dir)
    opt.gen_save_step_checkpoint = os.path.join(opt.gen_save_step_checkpoint_dir, opt.gen_save_step_checkpoint)
    opt.gen_save_step_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.gen_save_step_checkpoint)
    opt.gen_save_step_checkpoint_dir = os.path.join("/",*opt.gen_save_step_checkpoint.split("/")[:-1])
    
    opt.gen_save_final_checkpoint_dir = opt.gen_save_final_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    opt.gen_save_final_checkpoint_dir = fix(opt.gen_save_final_checkpoint_dir)
    opt.gen_save_final_checkpoint = os.path.join(opt.gen_save_final_checkpoint_dir, opt.gen_save_final_checkpoint)
    opt.gen_save_final_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.gen_save_final_checkpoint)
    opt.gen_save_final_checkpoint_dir = os.path.join("/",*opt.gen_save_final_checkpoint.split("/")[:-1])
    
    opt.gen_load_final_checkpoint_dir = opt.gen_load_final_checkpoint_dir.format(root_opt.gen_experiment_from_run, root_opt.gen_experiment_from_dir)
    opt.gen_load_final_checkpoint_dir = fix(opt.gen_load_final_checkpoint_dir)
    opt.gen_load_final_checkpoint = os.path.join(opt.gen_load_final_checkpoint_dir, opt.gen_load_final_checkpoint)
    opt.gen_load_final_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.gen_load_final_checkpoint)
    opt.gen_load_final_checkpoint_dir = os.path.join("/",*opt.gen_load_final_checkpoint.split("/")[:-1])
    
    if not last_step:
        opt.gen_load_step_checkpoint_dir = opt.gen_load_step_checkpoint_dir.format(root_opt.gen_experiment_from_run, root_opt.gen_experiment_from_dir)
    else:
        opt.gen_load_step_checkpoint_dir = opt.gen_load_step_checkpoint_dir.format(root_opt.gen_experiment_from_run, root_opt.this_viton_save_to_dir)
    opt.gen_load_step_checkpoint_dir = fix(opt.gen_load_step_checkpoint_dir)
    if not last_step:
        opt.gen_load_step_checkpoint = os.path.join(opt.gen_load_step_checkpoint_dir, opt.gen_load_step_checkpoint)
    else:
        if os.path.isdir(opt.gen_load_step_checkpoint_dir.format(root_opt.gen_experiment_from_run, root_opt.this_viton_save_to_dir)):
            os_list = os.listdir(opt.gen_load_step_checkpoint_dir.format(root_opt.gen_experiment_from_run, root_opt.this_viton_save_to_dir))
            os_list = [string for string in os_list if "gen" in string]
            last_step = sorted(os_list, key=sort_digit)[-1]
            opt.gen_load_step_checkpoint = os.path.join(opt.gen_load_step_checkpoint_dir, last_step)
    opt.gen_load_step_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.gen_load_step_checkpoint)
    opt.gen_load_step_checkpoint_dir = os.path.join("/",*opt.gen_load_step_checkpoint.split("/")[:-1])
    # ================================= gen DISCRIMINATOR =================================
    opt.gen_discriminator_save_step_checkpoint_dir = opt.gen_discriminator_save_step_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    opt.gen_discriminator_save_step_checkpoint_dir = fix(opt.gen_discriminator_save_step_checkpoint_dir)
    opt.gen_discriminator_save_step_checkpoint = os.path.join(opt.gen_discriminator_save_step_checkpoint_dir, opt.gen_discriminator_save_step_checkpoint)
    opt.gen_discriminator_save_step_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.gen_discriminator_save_step_checkpoint)
    opt.gen_discriminator_save_step_checkpoint_dir = os.path.join("/",*opt.gen_discriminator_save_step_checkpoint.split("/")[:-1])
    
    opt.gen_discriminator_save_final_checkpoint_dir = opt.gen_discriminator_save_final_checkpoint_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    opt.gen_discriminator_save_final_checkpoint_dir = fix(opt.gen_discriminator_save_final_checkpoint_dir)
    opt.gen_discriminator_save_final_checkpoint = os.path.join(opt.gen_discriminator_save_final_checkpoint_dir, opt.gen_discriminator_save_final_checkpoint)
    opt.gen_discriminator_save_final_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.gen_discriminator_save_final_checkpoint)
    opt.gen_discriminator_save_final_checkpoint_dir = os.path.join("/",*opt.gen_discriminator_save_final_checkpoint.split("/")[:-1])
    
    opt.gen_discriminator_load_final_checkpoint_dir = opt.gen_discriminator_load_final_checkpoint_dir.format(root_opt.gen_experiment_from_run, root_opt.gen_discriminator_experiment_from_dir)
    opt.gen_discriminator_load_final_checkpoint_dir = fix(opt.gen_discriminator_load_final_checkpoint_dir)
    opt.gen_discriminator_load_final_checkpoint = os.path.join(opt.gen_discriminator_load_final_checkpoint_dir, opt.gen_discriminator_load_final_checkpoint)
    opt.gen_discriminator_load_final_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.gen_discriminator_load_final_checkpoint)
    opt.gen_discriminator_load_final_checkpoint_dir = os.path.join("/",*opt.gen_discriminator_load_final_checkpoint.split("/")[:-1])
    
    if not last_step:
        opt.gen_discriminator_load_step_checkpoint_dir = opt.gen_discriminator_load_step_checkpoint_dir.format(root_opt.gen_discriminator_experiment_from_run, root_opt.gen_discriminator_experiment_from_dir)
    else:
        opt.gen_discriminator_load_step_checkpoint_dir = opt.gen_discriminator_load_step_checkpoint_dir.format(root_opt.gen_discriminator_experiment_from_run, root_opt.this_viton_save_to_dir)
    opt.gen_discriminator_load_step_checkpoint_dir = fix(opt.gen_discriminator_load_step_checkpoint_dir)
    if not last_step:
        opt.gen_discriminator_load_step_checkpoint = os.path.join(opt.gen_discriminator_load_step_checkpoint_dir, opt.gen_discriminator_load_step_checkpoint)
    else:
        if os.path.isdir(opt.gen_discriminator_load_step_checkpoint_dir.format(root_opt.gen_discriminator_experiment_from_run, root_opt.this_viton_save_to_dir)):
            os_list = os.listdir(opt.gen_discriminator_load_step_checkpoint_dir.format(root_opt.gen_discriminator_experiment_from_run, root_opt.this_viton_save_to_dir))
            os_list = [string for string in os_list if "gen_discriminator" in string]
            last_step = sorted(os_list, key=sort_digit)[-1]
            opt.gen_discriminator_load_step_checkpoint = os.path.join(opt.gen_discriminator_load_step_checkpoint_dir, last_step)
    opt.gen_discriminator_load_step_checkpoint = fix(opt.checkpoint_root_dir + "/" + opt.gen_discriminator_load_step_checkpoint)
    opt.gen_discriminator_load_step_checkpoint_dir = os.path.join("/",*opt.gen_discriminator_load_step_checkpoint.split("/")[:-1])
    return opt

def generator_get_root_opt_results_dir(parser, root_opt):
    root_opt.transforms_dir = root_opt.transforms_dir.format(root_opt.dataset_name)
    parser.tensorboard_dir = parser.tensorboard_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    parser.results_dir = parser.results_dir.format(root_opt.experiment_run, root_opt.this_viton_save_to_dir)
    return parser, root_opt

def generator_copy_root_opt_to_opt(parser, root_opt):
    parser.display_count = root_opt.display_count
    parser.cuda = root_opt.cuda
    parser.device = int(root_opt.device)
    parser.dataset_name = root_opt.dataset_name
    parser.load_last_step = root_opt.load_last_step if type(root_opt.load_last_step) == bool else eval(root_opt.load_last_step)
    parser.run_wandb = root_opt.run_wandb
    parser.stage = root_opt.VITON_Model
    parser.seed = root_opt.seed
    parser.datamode = root_opt.datamode
    parser.val_count = root_opt.val_count
    parser.viton_batch_size = root_opt.viton_batch_size
    parser.save_period = root_opt.save_period
    parser.print_step = root_opt.print_step
    parser.niter = root_opt.niter
    parser.niter_decay = root_opt.niter_decay
    parser.VITON_Type = root_opt.VITON_Type
    parser.VITON_selection_dir = parser.VITON_selection_dir.format(parser.VITON_Type, parser.VITON_Name)
    return parser

def generator_process_opt(opt, root_opt):
    parser = opt
    parser = argparse.Namespace(**parser)
    root_opt = generator_get_root_experiment_runs(root_opt)
    root_opt = generator_get_root_opt_experiment_dir(root_opt)
    parser = generator_get_root_opt_checkpoint_dir(parser, root_opt)
    parser, root_opt = generator_get_root_opt_results_dir(parser, root_opt)    
    parser = generator_copy_root_opt_to_opt(parser, root_opt)
    return parser, root_opt