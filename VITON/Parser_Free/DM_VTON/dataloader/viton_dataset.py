import json
import random
from pathlib import Path
from glob import glob
import numpy as np
import os
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


class LoadVITONDataset(Dataset):
    def __init__(
        self,
        root_opt,
        path: str,
        phase: str = 'train',
        size: tuple[int, int] = (256, 192),
        rail: bool = False
    ) -> None:
        super().__init__()
        self.dataroot = path
        self.phase = phase
        self.rail = rail
        self.root_opt = root_opt
        self.height, self.width = size[0], size[1]
        self.radius = 5
        self.transform_image = get_transform(train=(self.phase == 'train'))
        self.transform_parse = get_transform(
            train=(self.phase == 'train'), method=Image.NEAREST, normalize=False
        )
        if phase == 'train':
            self.img_names, self.cloth_names = [], []
            if root_opt.dataset_name ==  'Rail':
                self.full_input_data_path = os.path.join(root_opt.root_dir, root_opt.rail_dir)
                data_items = glob(f"{self.full_input_data_path}/train_img/*")
                self.get_clothing_name = lambda path_to_image:"_".join(path_to_image.split("/")[-1].split("_")[:-1])
                self.img_names = [os.path.join(path.split("/")[-1]) for path in data_items]
                self.clothing_names = [f"{self.get_clothing_name(image)}.jpg" for image in data_items]
                self.cloth_names = self.unique_clothes = list(set(self.clothing_names))
                # self.cloth_names['unpaired'] = random.sample(self.c_names['paired'], len(self.c_names['paired']))
            else:
                with open(os.path.join(self.dataroot,f"{phase}_pairs.txt")) as f:
                    for line in f.readlines():
                        img_name, c_name = line.strip().split()
                        self.img_names.append(img_name)
                        self.cloth_names.append(c_name)
                self.unique_clothes = list(set(self.cloth_names))
        else:
            self.img_names = glob(os.path.join(self.dataroot.replace("train","test"), "test_img","*") )
            self.cloth_names = glob(os.path.join(self.dataroot.replace("train","test"), "test_color","*") )
            self.edge_names = glob(os.path.join(self.dataroot.replace("train","test"), "test_edge","*") )
            
    def get_clothing(self, im_name):
        base_name = '_'.join(im_name.split('_')[:-1])
        base_name = f'{base_name}.jpg'
        if self.root_opt.datamode == 'test':
            base_name = base_name.replace('test_img','test_color')
        corresponding_index = self.cloth_names.index(base_name)
        return self.cloth_names[corresponding_index]

    def __getitem__(self, index: int) -> dict:
        if self.phase == 'train':
            if self.root_opt.dataset_name == 'Rail':
                im_name  = self.img_names[index]
                c_name = self.get_clothing(self.img_names[index])
            else:
                im_name, c_name = self.img_names[index], self.cloth_names[index]
            # Person image
            img = Image.open(os.path.join(self.dataroot,f'{self.phase}_img', im_name)).convert('RGB')
            cloth = Image.open(os.path.join(self.dataroot,f'{self.phase}_color', c_name)).convert('RGB')
                # Clothing edge
            cloth_edge = Image.open(os.path.join(self.dataroot,f'{self.phase}_edge', c_name)).convert('L')
        else:
            if self.root_opt.dataset_name == 'Rail':
                im_name  = self.img_names[index]
                c_name = self.get_clothing(self.img_names[index])
            else:
                im_name, c_name = self.img_names[index], self.cloth_names[index]
            edge_name = c_name.replace('color','edge')
            # Person image
            img = Image.open(im_name).convert('RGB')
            cloth = Image.open(c_name).convert('RGB')
            
            # Clothing edge
            cloth_edge = Image.open(edge_name).convert('L')
            im_name, c_name, edge_name = im_name.split("/")[-1], c_name.split("/")[-1],  edge_name.split("/")[-1]
            
        # img = img.resize((self.width, self.height))
        img_tensor = self.transform_image(img)  # [-1,1]

        # Clothing image
        
        # cloth = cloth.resize((self.width, self.height))
        cloth_tensor = self.transform_image(cloth)  # [-1,1]

        # cloth_edge = cloth_edge.resize((self.width, self.height))
        cloth_edge_tensor = self.transform_parse(cloth_edge)  # [-1,1]

        
        # Unpaired clothing image
        if self.phase == 'train':
            other_clothes = self.unique_clothes.copy()
            other_clothes.remove(c_name)  # remove the original cloth
            if len(other_clothes) > 0:
                un_c_name = random.choice(other_clothes)
            else:
                un_c_name = self.unique_clothes.copy()[0]
                
            un_cloth = Image.open(os.path.join(self.dataroot, f'{self.phase}_color', un_c_name)).convert(
                'RGB'
            )
            # un_cloth = un_cloth.resize((self.width, self.height))
            un_cloth_tensor = self.transform_image(un_cloth)  # [-1,1]

            # Unpaired Clothing edge
            
            un_cloth_edge = Image.open(os.path.join(self.dataroot, f'{self.phase}_edge', un_c_name)).convert(
                'L'
            )
            # un_cloth_edge = un_cloth_edge.resize((self.width, self.height))
            un_cloth_edge_tensor = self.transform_parse(un_cloth_edge)  # [-1,1]

        # Parse map
        parse_path1 = os.path.join(self.dataroot,f'{self.phase}_label', f'{Path(im_name).stem}.jpg')
        parse_path2 = os.path.join(self.dataroot,f'{self.phase}_label', f'{Path(im_name).stem}.png')
        parse_path = parse_path1 if os.path.isfile(parse_path1) else parse_path2
        parse = Image.open(parse_path).convert('L')
        # parse = parse.resize((self.width, self.height), Image.NEAREST)
        parse_tensor = self.transform_parse(parse) * 255.0

        # Pose: 18 keypoints [x0, y0, z0, x1, y1, z1, ...]
        with open(
            # Path(self.dataroot) / f'{self.phase}_pose' / f'{Path(im_name).stem}.json'
            os.path.join(self.dataroot,f'{self.phase}_pose', f'{Path(im_name).stem}.json')
        ) as f:
            pose_label = json.load(f)
            try:
                pose_data = pose_label['people'][0]['pose_keypoints']
            except IndexError:
                pose_data = [0 for i in range(54)]
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))[:18]

        point_num = pose_data.shape[0]
        pose_tensor = torch.zeros(point_num, self.height, self.width)
        r = self.radius
        im_pose = Image.new('L', (self.width, self.height))
        pose_draw = ImageDraw.Draw(im_pose)
        for i in range(point_num):
            one_map = Image.new('L', (self.width, self.height))
            draw = ImageDraw.Draw(one_map)
            pointx = pose_data[i, 0]
            pointy = pose_data[i, 1]
            if pointx > 1 and pointy > 1:
                draw.rectangle(
                    (pointx - r, pointy - r, pointx + r, pointy + r), 'white', 'white'
                )
                pose_draw.rectangle(
                    (pointx - r, pointy - r, pointx + r, pointy + r), 'white', 'white'
                )
            one_map = self.transform_image(one_map.convert('RGB'))
            pose_tensor[i] = one_map[0]

        # Densepose
        dense_mask = np.load(
            # Path(self.dataroot) / f'{self.phase}_densepose' / f'{Path(im_name).stem}.npy'
            os.path.join(self.dataroot, f'{self.phase}_densepose', f'{Path(im_name).stem}.npy')
        ).astype(np.float32)
        dense_tensor = self.transform_parse(dense_mask)

        if self.phase == 'train':
            return {
                'img_name': im_name,
                'color_name': c_name,
                'color_un_name': un_c_name,
                'parse_name': parse_path.split("/")[-1],
                'image': img_tensor,
                'color': cloth_tensor,
                'edge': cloth_edge_tensor,
                'color_un': un_cloth_tensor,
                'edge_un': un_cloth_edge_tensor,
                'label': parse_tensor,
                'pose': pose_tensor,
                'densepose': dense_tensor,
            }
        else:
            return {
                'img_name': im_name,
                'image': img_tensor,
                'color': cloth_tensor,
                'edge': cloth_edge_tensor,
                'p_name': im_name,
                'c_name': c_name,
                'label': parse_tensor,
            }

    def __len__(self) -> int:
        return len(self.img_names)


def get_transform(train, method=Image.BICUBIC, normalize=True):
    transform_list = []

    base = float(2**4)
    transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if train:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, 0)))

    transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def normalize():
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def __make_power_2(img, base, method=Image.BICUBIC):
    try:
        ow, oh = img.size  # PIL
    except Exception:
        oh, ow = img.shape  # numpy
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
