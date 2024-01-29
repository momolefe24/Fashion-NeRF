#coding=utf-8
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from PIL import ImageDraw
from glob import glob

import os
import os.path as osp
import numpy as np
import json

class CPDataset(data.Dataset):
    """Dataset for CP-VTON.
    """
    def __init__(self, root_opt, opt):
        super(CPDataset, self).__init__()
        # base setting
        self.root_opt = root_opt
        self.opt = opt 
        self.root_dir = root_opt.root_dir
        if root_opt.dataset_name == "Rail":
            self.read_data_dir = osp.join(self.root_dir, root_opt.rail_dir)
        else:
            self.read_data_dir = osp.join(self.root_dir, root_opt.original_dir)
        self.datamode = opt.datamode # train or test or self-defined
        self.dataroot = os.path.join(self.root_dir,os.path.normpath(os.path.join(root_opt.original_dir, "../")))
        self.stage = opt.stage # GMM or TOM
        self.fine_height = root_opt.fine_height
        self.fine_width = root_opt.fine_width
        self.radius = opt.radius
        self.transform = transforms.Compose([  \
                transforms.ToTensor(),   \
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.transform_parse_shape  = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        self.model_dir = osp.join(os.getcwd(),opt.VITON_selection_dir)
        # load data list
        self.im_names, self.cloth_names = [], []
        if root_opt.dataset_name ==  'Rail':
            self.full_input_data_path = os.path.join(root_opt.root_dir, root_opt.rail_dir)
            if self.datamode == 'train':
                data_items = glob(f"{self.full_input_data_path}/train_img/*")
            else:
                data_items = glob(f"{self.full_input_data_path}/test_img/*")
            self.get_clothing_name = lambda path_to_image:"_".join(path_to_image.split("/")[-1].split("_")[:-1])
            self.im_names = [os.path.join(path.split("/")[-1]) for path in data_items]
            self.clothing_names = [f"{self.get_clothing_name(image)}.jpg" for image in data_items]
            self.cloth_names = self.unique_clothes = list(set(self.clothing_names))
            # self.cloth_names['unpaired'] = random.sample(self.c_names['paired'], len(self.c_names['paired']))
        else:
            text_mode = "train_pairs.txt" if self.datamode == 'train' else "test_pairs.txt"
            with open(os.path.join(self.dataroot,text_mode)) as f:
                for line in f.readlines():
                    img_name, c_name = line.strip().split()
                    self.im_names.append(img_name)
                    self.cloth_names.append(c_name)
            self.unique_clothes = list(set(self.cloth_names))

        self.c_names = self.cloth_names

    def name(self):
        return "CPDataset"

    def __getitem__(self, index):
        
        im_name = self.im_names[index]
        parse_name = im_name.replace('.jpg', '.png')
        if self.root_opt.dataset_name == 'Rail':
            clothing_name = self.get_clothing_name(im_name)
            c_name = f"{clothing_name}.jpg"
            c_path= osp.join(self.read_data_dir, 'train_color', c_name) if self.datamode == 'train' else osp.join(self.read_data_dir, 'test_color', c_name)
            cmask_path= osp.join(self.read_data_dir, 'train_edge', c_name) if self.datamode == 'train' else osp.join(self.read_data_dir, 'test_edge', c_name)
            image_path = osp.join(self.read_data_dir, 'train_img', im_name) if self.datamode == 'train' else osp.join(self.read_data_dir, 'test_img', im_name)
            parse_path = osp.join(self.read_data_dir, 'train_label', parse_name) if self.datamode == 'train' else osp.join(self.read_data_dir, 'test_label', parse_name)
        else:
            c_name = self.c_names[index]
            c_path= osp.join(self.read_data_dir, 'cloth', c_name)
            cmask_path= osp.join(self.read_data_dir, 'cloth-mask', c_name)
            image_path = osp.join(self.read_data_dir, 'image', im_name)
            parse_path = osp.join(self.read_data_dir, 'image-parse', parse_name)
            c_name = self.c_names[index]

        # cloth image & cloth mask
        c = Image.open(c_path) 
        cm = Image.open(cmask_path) if self.root_opt.dataset_name == "Original" else Image.open(cmask_path).convert("L")
     
        c = self.transform(c)  # [-1,1]
        cm_array = np.array(cm)
        cm_array = (cm_array >= 128).astype(np.float32)
        cm = torch.from_numpy(cm_array) # [0,1]
        cm.unsqueeze_(0)

        # person image 
        
        im = Image.open(image_path)
        im = self.transform(im) # [-1,1]

        # load parsing image
        
        im_parse = Image.open(parse_path)
        parse_array = np.array(im_parse)
        parse_shape = (parse_array > 0).astype(np.float32)
        parse_head = (parse_array == 1).astype(np.float32) + \
                (parse_array == 2).astype(np.float32) + \
                (parse_array == 4).astype(np.float32) + \
                (parse_array == 13).astype(np.float32)
        parse_cloth = (parse_array == 5).astype(np.float32) + \
                (parse_array == 6).astype(np.float32) + \
                (parse_array == 7).astype(np.float32)
       
        # shape downsample
        parse_shape = Image.fromarray((parse_shape*255).astype(np.uint8))
        parse_shape = parse_shape.resize((self.fine_width//16, self.fine_height//16), Image.BILINEAR)
        parse_shape = parse_shape.resize((self.fine_width, self.fine_height), Image.BILINEAR)
        shape = self.transform_parse_shape(parse_shape) # [-1,1]
        phead = torch.from_numpy(parse_head) # [0,1]
        pcm = torch.from_numpy(parse_cloth) # [0,1]

        # upper cloth
        im_c = im * pcm + (1 - pcm) # [-1,1], fill 1 for other parts
        im_h = im * phead - (1 - phead) # [-1,1], fill 0 for other parts

        # load pose points
        if self.root_opt.dataset_name == 'Original':
            pose_name = im_name.replace('.jpg', '_keypoints.json')
            pose_path = osp.join(self.read_data_dir, 'pose', pose_name)
        else:    
            pose_name = im_name.replace('.jpg', '.json')
            pose_path = osp.join(self.read_data_dir, 'train_pose', pose_name)
        if self.opt.datamode == 'test':
           pose_path = pose_path.replace('train','test') 
        with open(pose_path, 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1,3))

        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)
        r = self.radius
        im_pose = Image.new('L', (self.fine_width, self.fine_height))
        pose_draw = ImageDraw.Draw(im_pose)
        for i in range(point_num):
            one_map = Image.new('L', (self.fine_width, self.fine_height))
            draw = ImageDraw.Draw(one_map)
            pointx = pose_data[i,0]
            pointy = pose_data[i,1]
            if pointx > 1 and pointy > 1:
                draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
                pose_draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
            one_map = self.transform_parse_shape(one_map)
            pose_map[i] = one_map[0]

        # just for visualization
        im_pose = self.transform_parse_shape(im_pose)
        
        # cloth-agnostic representation
        agnostic = torch.cat([shape, im_h, pose_map], 0) 

        if self.stage == 'GMM':
            # im_g = Image.open('grid.png')
            im_g = Image.open(osp.join(self.model_dir, 'grid.png'))
            im_g = self.transform(im_g)
        else:
            im_g = ''

        result = {
            'c_name':   c_name,     # for visualization
            'im_name':  im_name,    # for visualization or ground truth
            'cloth':    c,          # for input
            'cloth_mask':     cm,   # for input
            'image':    im,         # for visualization
            'agnostic': agnostic,   # for input
            'parse_cloth': im_c,    # for ground truth
            'shape': shape,         # for visualization
            'head': im_h,           # for visualization
            'pose_image': im_pose,  # for visualization
            'grid_image': im_g,     # for visualization
            }

        return result

    def __len__(self):
        return len(self.im_names)

class CPDataLoader(object):
    def __init__(self, opt, dataset):
        super(CPDataLoader, self).__init__()

        if opt.shuffle :
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=opt.viton_batch_size, shuffle=(train_sampler is None),
                num_workers=opt.viton_workers, pin_memory=True, sampler=train_sampler)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()
       
    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch
class CPDataTestLoader(object):
    def __init__(self, opt, dataset):
        super(CPDataTestLoader, self).__init__()

        self.data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=1)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()
       
    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch

if __name__ == "__main__":
    print("Check the dataset for geometric matching module!")
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default = "data")
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--stage", default = "GMM")
    parser.add_argument("--data_list", default = "train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 3)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    parser.add_argument('-j', '--workers', type=int, default=1)
    
    opt = parser.parse_args()
    dataset = CPDataset(opt)
    data_loader = CPDataLoader(opt, dataset)

    print('Size of the dataset: %05d, dataloader: %04d' \
            % (len(dataset), len(data_loader.data_loader)))
    first_item = dataset.__getitem__(0)
    first_batch = data_loader.next_batch()

    from IPython import embed; embed()

