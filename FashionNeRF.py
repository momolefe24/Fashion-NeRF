from utils import set_seed, get_vton_opt, get_vton_sweep, get_opt, process_opt, override_arguments, parse_arguments
from train import *
from test import *
from evaluate import *
import debugpy


root_opt = get_opt()
python_arg = parse_arguments()

if python_arg.debug == 1:
    debugpy.listen(5678)
    print("Ready")
    debugpy.wait_for_client()
    debugpy.breakpoint()
    
root_opt = override_arguments(root_opt, python_arg)
root_opt = process_opt(root_opt)
opt = get_vton_opt(root_opt)
sweeps = get_vton_sweep(root_opt) if python_arg.sweeps == 1 else None
run_wandb = True if python_arg.run_wandb == 1 else False
set_seed(python_arg.seed)


VITON_Name = python_arg.VITON_Name
VITON_Model = python_arg.VITON_Model

if python_arg.datamode == 'train':
    train_viton(opt, root_opt, run_wandb, sweeps,VITON_Name, VITON_Model)
elif python_arg.datamode == 'test':
    test_viton(opt, root_opt,VITON_Name, VITON_Model)
elif python_arg.datamode == 'evaluate':
    evaluate_viton(opt, root_opt,VITON_Name)