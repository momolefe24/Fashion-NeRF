import argparse
from pathlib import Path
import os
from VITON.Parser_Free.DM_VTON.utils.general import increment_path, yaml_save

import yaml
class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def parse_opt(self, root_opt):
        

        with open(os.path.join(root_opt.experiment_run_yaml,'train_condition.yml'), 'r') as config_file:
            config = yaml.safe_load(config_file)

        return config

    def _parse_opt(self, known: bool = False) -> None:
        self._add_args()
        return self.parser.parse_known_args()[0] if known else self.parser.parse_args()
    
    def _add_args(self):
        # For experiment
        self.parser.add_argument('--project', default='runs/test', help='save to project/name')  
        self.parser.add_argument('--name', default='Inference Pipeline', help='save to project/name')
        self.parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')    
        # self.parser.add_argument('--dataroot', type=str, default='../data/VITON-Clean/VITON_test', help='train dataset path')
        # self.parser.add_argument('--valroot', type=str, default='../data/VITON-Clean/VITON_test', help='val/test dataset path')
        # self.parser.add_argument('--dataroot', type=str, default='../data/rail/pf', help='train dataset path')
        # self.parser.add_argument('--valroot', type=str, default='../data/rail/pf', help='val/test dataset path')
        self.parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        self.parser.add_argument('--workers', type=int, default=1, help='max dataloader workers (per RANK in DDP mode)')    
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')
