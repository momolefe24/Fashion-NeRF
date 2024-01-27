import os
import torch
import sys

class BaseModel(torch.nn.Module):
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = [ opt.device ]
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = opt.g1_save_final_checkpoint_dir
        self.load_dir = opt.g1_load_final_checkpoint_dir

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, epoch_label,checkpoint,checkpoint_dir,cuda=True):
        if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
        if "step" in checkpoint:
            torch.save(network.state_dict(), checkpoint % epoch_label)
        else:
            torch.save(network.state_dict(), checkpoint )
        if cuda:
            network.cuda()

    # helper loading function that can be used by subclasses
    def load_network(self, network,checkpoint):        
        network.load_state_dict(torch.load(checkpoint))

    def update_learning_rate():
        pass
