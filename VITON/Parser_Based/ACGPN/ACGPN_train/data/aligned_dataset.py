## Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
from VITON.Parser_Based.ACGPN.ACGPN_train.data.base_dataset import BaseDataset, get_params, get_transform, normalize
from VITON.Parser_Based.ACGPN.ACGPN_train.data.image_folder import make_dataset, make_dataset_test
from PIL import Image
import torch
import json
import numpy as np
import os.path as osp
from PIL import ImageDraw
import ipdb

class AlignedDataset(BaseDataset):
    def initialize(self, opt, root_opt):
        self.opt = opt
        self.root_dir = opt.root_dir
        if opt.dataset_name == "Rail":
            self.read_data_dir = osp.join(self.root_dir, root_opt.rail_dir)
        else:
            self.read_data_dir = osp.join(self.root_dir, root_opt.original_dir)
        self.diction={}
        self.data_path = self.read_data_dir
        

        self.fine_height=256
        self.fine_width=192
        self.radius=5
        
        ### input A test (label maps)
        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(self.data_path, opt.datamode + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        ### input B (real images)
        dir_B = '_B' if self.opt.label_nc == 0 else '_img'
        self.dir_B = os.path.join(self.data_path, opt.datamode + dir_B)
        self.B_paths = sorted(make_dataset(self.dir_B))
        self.BR_paths = sorted(make_dataset(self.dir_B))

        self.dataset_size = len(self.A_paths)
        self.build_index(self.B_paths)

        ### input E (edge_maps)
        dir_E = '_edge'
        self.dir_E = os.path.join(self.data_path, opt.datamode + dir_E)
        self.E_paths = sorted(make_dataset(self.dir_E))
        self.ER_paths = make_dataset(self.dir_E)

        ### input M (masks)
        if opt.dataset_name == 'Original':
            dir_M = '_mask'
        else:    
            dir_M = '_edge'
        self.dir_M = os.path.join(self.data_path, opt.datamode + dir_M)
        self.M_paths = sorted(make_dataset(self.dir_M))
        self.MR_paths = make_dataset(self.dir_M)

        ### input MC(color_masks)
        if opt.dataset_name == 'Original':
            dir_MC = '_colormask'
        else:    
            dir_MC = '_color'
        self.dir_MC = os.path.join(self.data_path, opt.datamode + dir_MC)
        self.MC_paths = sorted(make_dataset(self.dir_MC))
        self.MCR_paths = make_dataset(self.dir_MC)
        
        ### input C(color)
        dir_C = '_color'
        self.dir_C = os.path.join(self.data_path, opt.datamode + dir_C)
        self.C_paths = sorted(make_dataset(self.dir_C))
        self.CR_paths = make_dataset(self.dir_C)

            
    def random_sample(self,item):
        name = item.split('/')[-1]
        name = name.split('-')[0]
        lst=self.diction[name]
        new_lst=[]
        for dir in lst:
            if dir != item:
                new_lst.append(dir)
        return new_lst[np.random.randint(len(new_lst))]
    def build_index(self,dirs):
        #ipdb.set_trace()
        for k,dir in enumerate(dirs):
            name=dir.split('/')[-1]
            name=name.split('-')[0]

            # print(name)
            for k,d in enumerate(dirs[max(k-20,0):k+20]):
                if name in d:
                    if name not in self.diction.keys():
                        self.diction[name]=[]
                        self.diction[name].append(d)
                    else:
                        self.diction[name].append(d)


    def __getitem__(self, index):        
        train_mask=9600
        ### input A (label maps)
        # box=[]
        # for k,x in enumerate(self.A_paths):
        #     if '2372656' in x :
        #         box.append(k)
        # index=box[np.random.randint(len(box))]
        test=index#np.random.randint(10000)
        A_path = self.A_paths[index]
        A = Image.open(A_path).convert('L')


        params = get_params(self.opt, A.size)
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(A.convert('RGB'))
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0
        
        B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)
        B_path = self.B_paths[index]
        BR_path = self.BR_paths[index]
        B = Image.open(B_path).convert('RGB')
        BR = Image.open(BR_path).convert('RGB')
        transform_B = get_transform(self.opt, params)      
        B_tensor = transform_B(B)
        BR_tensor = transform_B(BR)

        
        ### input M (masks)
        if self.opt.dataset_name == 'Original':
            M_path = self.M_paths[np.random.randint(len(self.M_paths))]
            MR_path =self.MR_paths[np.random.randint(len(self.M_paths))]
        else:
            M_path = self.M_paths[np.random.randint(len(self.M_paths))]
            MR_path =self.MR_paths[np.random.randint(len(self.MR_paths))]
            
        M = Image.open(M_path).convert('L')
        MR = Image.open(MR_path).convert('L')
        M_tensor = transform_A(MR)

        ### input_MC (colorMasks)
        MC_path = B_path#self.MC_paths[1]
        MCR_path = B_path#self.MCR_paths[1]
        MCR = Image.open(MCR_path).convert('L')
        MC_tensor = transform_A(MCR)

        ### input_C (color)
        # print(self.C_paths)
        get_clothing_name = lambda path_to_image:"_".join(path_to_image.split("/")[-1].split("_")[:-1])
        clothing_name = get_clothing_name(self.A_paths[test])
        if self.opt.dataset_name == 'Original':
            C_path =  f"{os.path.join(self.dir_C, clothing_name)}_1.jpg"
        else:
            C_path =  f"{os.path.join(self.dir_C, clothing_name)}.jpg"
        # try:
        #     C_path = self.C_paths[test]
        # except:
        #     print(f"Trying to get index {test} at {self.C_paths}")
        C = Image.open(C_path).convert('RGB')
        C_tensor = transform_B(C)

        ##Edge
        # E_path = self.E_paths[test]
        if self.opt.dataset_name == 'Original':
            E_path =  f"{os.path.join(self.dir_E, clothing_name)}_1.jpg"
        else:
            E_path =  f"{os.path.join(self.dir_E, clothing_name)}.jpg"
        # print(E_path)
        E = Image.open(E_path).convert('L')
        E_tensor = transform_A(E)


        ##Pose
        if self.opt.dataset_name == 'Original':
            pose_name =B_path.replace('.png', '_keypoints.json').replace('.jpg','_keypoints.json').replace('train_img','train_pose')
        else:
            pose_name =B_path.replace('.png', '.json').replace('.jpg','.json').replace('train_img','train_pose')
        if not self.opt.isTrain:
            pose_name = pose_name.replace("train", "test")
        if self.opt.datamode == 'test':
            pose_name = pose_name.replace("test_img", "test_pose")
        with open(osp.join(pose_name), 'r') as f:
            pose_label = json.load(f)
            try:
                pose_data = pose_label['people'][0]['pose_keypoints']
            except IndexError:
                pose_data = [0 for i in range(54)]
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
            one_map = transform_B(one_map.convert('RGB'))
            pose_map[i] = one_map[0]
        P_tensor=pose_map
        # Define the mapping {0: 1, 3: 4, 7: 8}
        if self.opt.dataset_name == 'Rail':
            mapping = {# VITON: ACGPN
                    0: 0, # background
                    10: 0, # background
                    1:1, # hair
                    2:1, # hair
                    4:12, # face
                    13: 12, # face
                    5:4, # upper body
                    6:4, # upper body
                    7:4, # upper body
                    9:8, # bottom
                    12:8, # bottom 
                    14:11, # left arm
                    15:13, # right arm
                    16: 9, # left leg
                    17: 10, # righttleg
                    18: 5, # left shoe
                    19: 6, # right shoe
                    3: 7, # noise,
                    11: 7, # noise,
                    8: 0, # scoks
                    }
            A_tensor = self.map_labels(A_tensor, mapping)
        if self.opt.isTrain:
            input_dict = { 'name': A_path.split("/")[-1], 'label': A_tensor,'image': B_tensor, 'image_ref': BR_tensor, 'path': A_path,
                            'edge': E_tensor,'color': C_tensor, 'mask': M_tensor, 'colormask': MC_tensor,'pose':P_tensor
                          }
        else:
            input_dict = {'name': A_path.split("/")[-1], 'label': A_tensor, 'edge': E_tensor,'color': C_tensor, 'image': B_tensor, 'image_ref': BR_tensor, 'path': A_path, 'pose':P_tensor}

        return input_dict

    def __len__(self):
        return len(self.A_paths) 

    def name(self):
        return 'AlignedDataset'
    import torch


    def map_labels(self, image, mapping):
        # Convert the image to a tensor with float type for processing
        image_float = image.type(torch.float32)

        # Apply the mappings
        for src, dst in mapping.items():
            image_float = torch.where(image == src, torch.tensor(dst, dtype=torch.float32), image_float)

        # Convert back to the original data type
        mapped_image = image_float.type(torch.uint8)
        return mapped_image


