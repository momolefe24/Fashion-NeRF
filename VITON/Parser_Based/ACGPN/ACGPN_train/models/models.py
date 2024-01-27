### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch
import ipdb

def create_model(opt, wandb_opt=None):
    if opt.model == 'pix2pixHD':
        from .pix2pixHD_model import Pix2PixHDModel, InferenceModel
        if opt.datamode == 'train':
            model = Pix2PixHDModel()
            #ipdb.set_trace()
        else:
            model = InferenceModel()

    model.initialize(opt, wandb_opt=wandb_opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))

    if opt.datamode == 'train':
        model = torch.nn.DataParallel(model, device_ids=[opt.device])

    return model
