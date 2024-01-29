import time
# from VITON.Parser_Based.ACGPN2.ACGPN_inference.data.data_loader import CreateDataLoader
from VITON.Parser_Based.ACGPN.ACGPN_train.data.data_loader import CreateDataLoader
from VITON.Parser_Based.ACGPN.ACGPN_inference.models.models import create_model
from VITON.Parser_Based.ACGPN.ACGPN_inference.util import util
from VITON.Parser_Based.ACGPN.ACGPN_train.train import process_opt
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
    M_f=M_f.astype(int)
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
    arm1=torch.FloatTensor((old_label.cpu().numpy()==11).astype(int))
    arm2=torch.FloatTensor((old_label.cpu().numpy()==13).astype(int))
    noise=torch.FloatTensor((old_label.cpu().numpy()==7).astype(int))
    label=label*(1-arm1)+arm1*4
    label=label*(1-arm2)+arm2*4
    label=label*(1-noise)+noise*4
    return label

def test_acgpn_(opt, root_opt):
    opt,root_opt = process_opt(opt, root_opt)
    print("Start to test %s!")
    _test_acgpn_(opt, root_opt)

def _test_acgpn_(opt, root_opt):
    writer = SummaryWriter(opt.tensorboard_dir)
    data_loader = CreateDataLoader(opt, root_opt)
    dataset = data_loader.load_test_data()
    dataset_size = len(data_loader)
    print('# Inference images = %d' % dataset_size)

    model = create_model(opt)
    start_epoch, epoch_iter = 1, 0
    step = 0
    total_steps = (start_epoch-1) * dataset_size + epoch_iter


    step = 0
    prediction_dir = os.path.join(opt.results_dir, 'prediction')
    ground_truth_dir = os.path.join(opt.results_dir, 'ground_truth')
    ground_truth_mask_dir = os.path.join(opt.results_dir, 'ground_truth_mask')
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)
    if not os.path.exists(ground_truth_dir):
        os.makedirs(ground_truth_dir)
    if not os.path.exists(ground_truth_mask_dir):
        os.makedirs(ground_truth_mask_dir)
        
    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size
        for i, data in enumerate(dataset, start=epoch_iter):

            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            save_fake = True
            t_mask = torch.FloatTensor((data['label'].cpu().numpy() == 7).astype(float))
            mask_clothes = torch.FloatTensor((data['label'].cpu().numpy() == 4).astype(int))
            mask_fore = torch.FloatTensor((data['label'].cpu().numpy() > 0).astype(int))
            img_fore = data['image'] * mask_fore
            img_fore_wc = img_fore * mask_fore
            all_clothes_label = changearm(data['label'])



            ############## Forward Pass ######################
            losses, fake_image, real_image, input_label,L1_loss,style_loss,clothes_mask,CE_loss,rgb,alpha= model(Variable(data['label'].cuda()),Variable(data['edge'].cuda()),Variable(img_fore.cuda()),Variable(mask_clothes.cuda())
                                                                                                ,Variable(data['color'].cuda()),Variable(all_clothes_label.cuda()),Variable(data['image'].cuda()),Variable(data['pose'].cuda()) ,Variable(data['image'].cuda()) ,Variable(mask_fore.cuda()))

            # sum per device losses
            losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
            loss_dict = dict(zip(model.module.loss_names, losses))
            a = generate_label_color(generate_label_plain(input_label)).float().cuda()
            b = real_image.float().cuda()
            c = fake_image.float().cuda()
            d=torch.cat([clothes_mask,clothes_mask,clothes_mask],1)
            combine = torch.cat([a[0],d[0],b[0],c[0],rgb[0]], 2).squeeze()
            # combine=c[0].squeeze()
            cv_img=(combine.permute(1,2,0).detach().cpu().numpy()+1)/2

            # image_name = os.path.join(prediction_dir, data['name'][0])
            # ground_truth_image_name = os.path.join(ground_truth_dir, data['name'][0])
            # ground_truth_image_name = os.path.join(ground_truth_dir, data['name'][0])
            # util.save_tensor_as_image(warped_cloth, image_name)
            # util.save_tensor_as_image(img_fore, ground_truth_image_name)
            # util.save_tensor_as_image(img_fore, ground_truth_image_name)
            # save output
            # for j in range(opt.batchSize):
            #     print("Saving", data['name'][j])
            #     util.save_tensor_as_image(fake_image[j],
            #                             os.path.join(fake_image_dir, data['name'][j]))
            #     util.save_tensor_as_image(warped_cloth[j],
            #                             os.path.join(warped_cloth_dir, data['name'][j]))
            #     util.save_tensor_as_image(refined_cloth[j],
            #                             os.path.join(refined_cloth_dir, data['name'][j]))
            #     if epoch_iter >= dataset_size:
            #         break
        
        # end of epoch 
        iter_end_time = time.time()
        print('End of epoch %d / %d \t Time Taken: %d sec' %
            (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        break