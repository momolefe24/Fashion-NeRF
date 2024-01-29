import torch.utils.data
# from base_data_loader import BaseDataLoader
import os.path
import os.path as osp
from VITON.Parser_Free.PF_AFN.PF_AFN_train.data.base_dataset import BaseDataset, get_params, get_transform
from VITON.Parser_Free.PF_AFN.PF_AFN_train.data.image_folder import make_dataset
from PIL import Image
import linecache
from glob import glob

class BaseDataLoader():
    def __init__(self):
        pass
    
    def initialize(self, opt):
        self.opt = opt
        pass

    def load_data(self):
        return None

def CreateDataset(opt, root_opt):
    dataset = None
    dataset = AlignedDataset()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt, root_opt)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt, root_opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt, root_opt)
        self.dataset.__getitem__(0)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.viton_batch_size,
            shuffle = False,
            num_workers=1)

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)

class CustomDatasetTestDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetTestDataLoader'

    def initialize(self, opt, root_opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt, root_opt)
        self.dataset.__getitem__(0)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=1,
            shuffle = False,
            num_workers=1)

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)

def CreateDataLoader(opt, root_opt):
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt, root_opt)
    return data_loader

def CreateDataTestLoader(opt, root_opt):
    data_loader = CustomDatasetTestDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt, root_opt)
    return data_loader

class AlignedDataset(BaseDataset):
    def initialize(self, opt, root_opt):
        self.opt = opt
        self.root_dir = opt.root_dir
        self.fine_height=256
        self.fine_width=192
        if opt.dataset_name == "Rail":
            self.read_data_dir = osp.join(self.root_dir, root_opt.rail_dir)
        else:
            self.read_data_dir = osp.join(self.root_dir, root_opt.original_dir)
        self.diction={}
        self.data_path = self.read_data_dir
        # self.dataset_size = len(open(os.path.isfile(os.path.join(os.getcwd(),"PF-AFN/PF-AFN_test","demo.txt"))).readlines())

        # self.dir_I = os.path.join(opt.dataroot, opt.phase + dir_I)
        self.dir_I = os.path.join(self.read_data_dir, opt.datamode + "_img")
        self.I_paths = sorted(make_dataset(self.dir_I))

        # self.dir_C = os.path.join(opt.datamode + dir_C)
        self.dir_C = os.path.join(self.read_data_dir, opt.datamode + "_color")
        self.C_paths = sorted(make_dataset(self.dir_C))
        # self.dir_E = os.path.join(opt.datamode + dir_E)
        self.dir_E = os.path.join(self.read_data_dir, opt.datamode + "_edge")
        self.E_paths = sorted(make_dataset(self.dir_E))
        self.dataset_size = len(self.I_paths)
        
        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(self.data_path, opt.datamode + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))
        
        
        self.get_clothing_name = lambda path_to_image:"_".join(path_to_image.split("/")[-1].split("_")[:-1])
        
    def __getitem__(self, index):
        if self.opt.dataset_name == "Rail":
            file_path ='demo.txt'
            im_name  = self.I_paths[index].split("/")[-1]
            c_name = f"{self.get_clothing_name(im_name)}.jpg"
            # c_name = self.get_clothing(self.img_names[index])

            A_path = self.A_paths[index]
            A = Image.open(A_path).convert('L')

            params = get_params(self.opt, A.size)
            if self.opt.label_nc == 0:
                transform_A = get_transform(self.opt, params)
                A_tensor = transform_A(A.convert('RGB'))
            else:
                transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
                A_tensor = transform_A(A) * 255.0
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
            
            I_path = os.path.join(self.dir_I,im_name)
            I = Image.open(I_path).convert('RGB')

            params = get_params(self.opt, I.size)
            transform = get_transform(self.opt, params)
            transform_E = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)

            I_tensor = transform(I)

            C_path = os.path.join(self.dir_C,c_name)
            C = Image.open(C_path).convert('RGB')
            C_tensor = transform(C)

            E_path = os.path.join(self.dir_E,c_name)
            E = Image.open(E_path).convert('L')
            E_tensor = transform_E(E)

            input_dict = { 'im_name':im_name,'image': I_tensor,'label': A_tensor,'clothes': C_tensor, 'edge': E_tensor}
            return input_dict
        else:
            im_name, c_name, e_name = self.I_paths[index], self.C_paths[index], self.E_paths[index]

            I = Image.open(im_name).convert('RGB')
            params = get_params(self.opt, I.size)
            transform = get_transform(self.opt, params)
            transform_E = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)

            I_tensor = transform(I)

            C = Image.open(c_name).convert('RGB')
            C_tensor = transform(C)

            E = Image.open(e_name).convert('L')
            E_tensor = transform_E(E)

            input_dict = {'image': I_tensor, 'clothes': C_tensor, 'edge': E_tensor}
            return input_dict

    def __len__(self):
        return self.dataset_size
    
    
    def map_labels(self, image, mapping):
        # Convert the image to a tensor with float type for processing
        image_float = image.type(torch.float32)

        # Apply the mappings
        for src, dst in mapping.items():
            image_float = torch.where(image == src, torch.tensor(dst, dtype=torch.float32), image_float)

        # Convert back to the original data type
        mapped_image = image_float.type(torch.uint8)
        return mapped_image
    def name(self):
        return 'AlignedDataset'
