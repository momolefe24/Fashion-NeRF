import os.path
from VITON.Parser_Free.PF_AFN.PF_AFN_train.data.base_dataset import BaseDataset, get_params, get_transform
from VITON.Parser_Free.PF_AFN.PF_AFN_train.data.image_folder import make_dataset
from PIL import Image
from glob import glob
import torch
import json
import numpy as np
import os.path as osp
from PIL import ImageDraw


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

        
        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(self.data_path, opt.datamode + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        self.fine_height=256
        self.fine_width=192
        self.radius=5

        dir_B = '_B' if self.opt.label_nc == 0 else '_img'
        self.dir_B = os.path.join(self.data_path, opt.datamode + dir_B)  
        self.B_paths = sorted(make_dataset(self.dir_B))

        self.dataset_size = len(self.A_paths)

        
        dir_E = '_edge'
        self.dir_E = os.path.join(self.data_path, opt.datamode + dir_E)
        self.E_paths = sorted(make_dataset(self.dir_E))

        
        dir_C = '_color'
        self.dir_C = os.path.join(self.data_path, opt.datamode + dir_C)
        self.C_paths = sorted(make_dataset(self.dir_C))
    
    def map_labels(self, image, mapping):
        # Convert the image to a tensor with float type for processing
        image_float = image.type(torch.float32)

        # Apply the mappings
        for src, dst in mapping.items():
            image_float = torch.where(image == src, torch.tensor(dst, dtype=torch.float32), image_float)

        # Convert back to the original data type
        mapped_image = image_float.type(torch.uint8)
        return mapped_image
    
    def __getitem__(self, index):        

        A_path = self.A_paths[index]
        A = Image.open(A_path).convert('L')

        params = get_params(self.opt, A.size)
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(A.convert('RGB'))
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0

        B_path = self.B_paths[index]
        B = Image.open(B_path).convert('RGB')
        transform_B = get_transform(self.opt, params)      
        B_tensor = transform_B(B)

        get_clothing_name = lambda path_to_image:"_".join(path_to_image.split("/")[-1].split("_")[:-1])
        clothing_name = get_clothing_name(self.A_paths[index])
        if self.opt.dataset_name == 'Original':
            C_path =  f"{os.path.join(self.dir_C, clothing_name)}_1.jpg"
        else:
            C_path =  f"{os.path.join(self.dir_C, clothing_name)}.jpg"
        C = Image.open(C_path).convert('RGB')
        C_tensor = transform_B(C)

        if self.opt.dataset_name == 'Original':
            E_path =  f"{os.path.join(self.dir_E, clothing_name)}_1.jpg"
        else:
            E_path =  f"{os.path.join(self.dir_E, clothing_name)}.jpg"
        E = Image.open(E_path).convert('L')
        E_tensor = transform_A(E)

        index_un = np.random.randint(len(self.A_paths))
        C_un_path = self.C_paths[index_un]
        C_un = Image.open(C_un_path).convert('RGB')
        C_un_tensor = transform_B(C_un)

        E_un_path = self.E_paths[index_un]
        E_un = Image.open(E_un_path).convert('L')
        E_un_tensor = transform_A(E_un)

        ##Pose
        if self.opt.dataset_name == 'Original':
            pose_name =B_path.replace('.png', '.json').replace('.jpg','.json').replace('train_img','train_pose')
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

        densepose_name = B_path.replace('.png', '.npy').replace('.jpg','.npy').replace('train_img','train_densepose')
        dense_mask = np.load(densepose_name).astype(np.float32)
        dense_mask = transform_A(dense_mask)

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
            input_dict = { 'name': A_path.split("/")[-1],'label': A_tensor, 'image': B_tensor, 'path': A_path, 'img_path': B_path ,'color_path': C_path,'color_un_path': C_un_path,
                            'edge': E_tensor, 'color': C_tensor, 'edge_un': E_un_tensor, 'color_un': C_un_tensor, 'pose':P_tensor, 'densepose':dense_mask
                          }
        else:
            input_dict = { 'name': A_path.split("/")[-1],'label': A_tensor, 'image': B_tensor, 'path': A_path, 'img_path': B_path ,'color_path': C_path,'color_un_path': C_un_path,
                            'edge': E_tensor, 'color': C_tensor, 'edge_un': E_un_tensor, 'color_un': C_un_tensor, 'pose':P_tensor, 'densepose':dense_mask
                          }

        return input_dict

    def __len__(self):
        return len(self.A_paths) 

    def name(self):
        return 'AlignedDataset'
