from utils import get_opt, process_opt, override_arguments, parse_arguments
import debugpy
import yaml

def get_vton_opt(root_opt):
    yml_name =  f"yaml/{str(root_opt.VITON_Name).lower()}.yml"
    with open(yml_name, 'r') as config_file:
        config = yaml.safe_load(config_file)
    return config

def get_vton_sweep(root_opt):
    with open(root_opt.sweeps_yaml, 'r') as config_file:
        config = yaml.safe_load(config_file)
    return config

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
sweeps = None
if python_arg.sweeps == 1:
    sweeps = get_vton_sweep(root_opt)   
    
if python_arg.run_wandb == 1:
    run_wandb=True
else:
    run_wandb=False


""" ================== NeRF imports =================="""
# from NeRF.Vanilla_NeRF.test_helper import render, create_nerf, to8b
# from NeRF.Vanilla_NeRF.run_nerf import train
""" ================== PARSER-BASED VITON IMPORTS =================="""
# Low-Res
""" ------------------ CP-VTON imports ------------------"""
# from VITON.Parser_Based.CP_VTON.train import train_cpvton_
# from VITON.Parser_Based.CP_VTON.test import test_cpvton_
""" ------------------ CP-VTON-Plus imports ------------------"""
# from VITON.Parser_Based.CP_VTON_plus.train import train_cpvton_plus_
# from VITON.Parser_Based.CP_VTON_plus.test import test_cpvton_plus_

""" ------------------ ACGPN imports ------------------"""
# from VITON.Parser_Based.ACGPN.ACGPN_train.train import train_acgpn_
# from VITON.Parser_Based.ACGPN.ACGPN_inference.test import test_acgpn_

# High-Res
""" ------------------ HR-VITON imports ------------------ """
# from VITON.Parser_Based.HR_VITON.train_condition import train_hrviton_tocg_
# from VITON.Parser_Based.HR_VITON.train_generator import train_tryon_
from VITON.Parser_Based.HR_VITON.test_condition import test_hrviton_tocg_
from VITON.Parser_Based.HR_VITON.evaluate import evaluate_hrviton_tocg_
# from VITON.Parser_Based.HR_VITON.test_generator import test_hrviton_gen_
""" ------------------ LadiVTON imports ------------------"""
# from VITON.Parser_Based.Ladi_VTON.src.train_tps import train_ladi_vton_tps_
# from VITON.Parser_Based.Ladi_VTON.src.train_emasc import train_emasc_
# from VITON.Parser_Based.Ladi_VTON.src.train_tps import main

""" ================== PARSER-FREE VITON IMPORTS =================="""
# if root_opt.VITON_Model != 'EMASC':
#     from VITON.Parser_Free.DM_VTON.train_pb_warp import train_pb_warp_
#     from VITON.Parser_Free.DM_VTON.train_pb_e2e import train_pb_e2e_
#     from VITON.Parser_Free.DM_VTON.train_pf_warp import train_pf_warp_
#     from VITON.Parser_Free.DM_VTON.train_pf_e2e import train_pf_e2e_
#     from VITON.Parser_Free.DM_VTON.test import test_dm_vton
    
#     from VITON.Parser_Free.PF_AFN.PF_AFN_train.train_PBAFN_stage1 import train_pfafn_pb_warp_
#     from VITON.Parser_Free.PF_AFN.PF_AFN_train.train_PBAFN_e2e import train_pfafn_pb_gen_
#     from VITON.Parser_Free.PF_AFN.PF_AFN_train.train_PFAFN_stage1 import train_pfafn_pf_warp_
#     from VITON.Parser_Free.PF_AFN.PF_AFN_train.train_PFAFN_e2e import train_pfafn_e2e_
#     from VITON.Parser_Free.PF_AFN.PF_AFN_test.test import test_pfafn_

# def train_cp_vton(): # Warping
#     train_cpvton_(opt, root_opt, run_wandb=run_wandb, sweep=sweeps)
  
# def test_cp_vton():
#     test_cpvton_(opt, root_opt)
    
# def train_cp_vton_plus():
#     train_cpvton_plus_(opt, root_opt, run_wandb=run_wandb, sweep=sweeps)
    
# def test_cp_vton_plus():
#     test_cpvton_plus_(opt, root_opt)
    
# def test_acgpn():
#     test_acgpn_(opt, root_opt)
    
# def train_acgpn():
#     train_acgpn_(opt, root_opt, run_wandb=run_wandb, sweep=sweeps)
    
# def train_dmvton():
#     if python_arg.VITON_Model == 'PB_Warp':
#         train_pb_warp_(opt, root_opt, run_wandb=run_wandb, sweep=sweeps)
#     elif python_arg.VITON_Model == 'PB_Gen':
#         train_pb_e2e_(opt, root_opt, run_wandb=run_wandb, sweep=sweeps)
#     elif python_arg.VITON_Model == 'PF_Warp':
#         train_pf_warp_(opt, root_opt, run_wandb=run_wandb, sweep=sweeps)
#     elif python_arg.VITON_Model == 'PF_Gen':
#         train_pf_e2e_(opt, root_opt, run_wandb=run_wandb, sweep=sweeps)    

# def test_pfafn():
#     test_pfafn_(opt, root_opt)
    
# def train_pfafn():
#     if python_arg.VITON_Model == 'PB_Warp':
#         train_pfafn_pb_warp_(opt, root_opt, run_wandb=run_wandb, sweep=sweeps)
#     elif python_arg.VITON_Model == 'PB_Gen':
#         train_pfafn_pb_gen_(opt, root_opt, run_wandb=run_wandb, sweep=sweeps)
#     elif python_arg.VITON_Model == 'PF_Warp':
#         train_pfafn_pf_warp_(opt, root_opt, run_wandb=run_wandb, sweep=sweeps)
#     elif python_arg.VITON_Model == 'PF_Gen':
#         train_pfafn_e2e_(opt, root_opt, run_wandb=run_wandb, sweep=sweeps)   

# def test_dmvton():
#     test_dm_vton(opt, root_opt)        


# def train_hrviton():
#     if python_arg.VITON_Model == 'TOCG':
#         train_hrviton_tocg_(opt, root_opt, run_wandb=run_wandb, sweep=sweeps)    
#     elif python_arg.VITON_Model == 'GEN':
#         train_tryon_(opt, root_opt, run_wandb=run_wandb, sweep=sweeps)

def test_hrviton():
    if python_arg.VITON_Model == 'TOCG':
        test_hrviton_tocg_(opt, root_opt)   
    # elif python_arg.VITON_Model == 'GEN':
    #     test_hrviton_gen_(opt, root_opt)   
     
def evaluate_hrviton():
    if python_arg.VITON_Model == 'TOCG':
        evaluate_hrviton_tocg_(opt, root_opt)   
    # elif python_arg.VITON_Model == 'GEN':
    #     evaluate_hrviton_gen_(opt, root_opt)  
        
# def train_ladi_vton():
#     if python_arg.VITON_Model == 'TPS':
#         train_ladi_vton_tps_(opt, root_opt, run_wandb=run_wandb)
#     elif python_arg.VITON_Model == 'EMASC':
#         train_emasc_(opt, root_opt, run_wandb=run_wandb)


if python_arg.datamode == 'train':
    pass
    # if python_arg.VITON_Name == 'HR_VITON':
    #     train_hrviton()
    # elif python_arg.VITON_Name == 'ACGPN':
    #     train_acgpn()
    # elif python_arg.VITON_Name == 'DM_VTON':
    #     train_dmvton()
    # elif python_arg.VITON_Name == 'PF_AFN':
    #     train_pfafn()
    # elif python_arg.VITON_Name == 'CP_VTON':
    #     train_cp_vton()
    # elif python_arg.VITON_Name == 'CP_VTON_plus':
    #     train_cp_vton_plus()
elif python_arg.datamode == 'test':
    pass
    # if python_arg.VITON_Name == 'HR_VITON':
    #     test_hrviton()
    # elif python_arg.VITON_Name == 'ACGPN':
    #     test_acgpn()
    # elif python_arg.VITON_Name == 'DM_VTON':
    #     test_dmvton()
    # elif python_arg.VITON_Name == 'CP_VTON':
    #     test_cp_vton()
    # elif python_arg.VITON_Name == 'CP_VTON_plus':
    #     test_cp_vton_plus()    
    # elif python_arg.VITON_Name == 'PF_AFN':
    #     test_pfafn()
elif python_arg.datamode == 'evaluate':
    if python_arg.VITON_Name == 'HR_VITON':
        evaluate_hrviton()
    # elif python_arg.VITON_Name == 'ACGPN':
    #     evaluate_acgpn()
    # elif python_arg.VITON_Name == 'DM_VTON':
    #     evaluate_dmvton()
    # elif python_arg.VITON_Name == 'CP_VTON':
    #     evaluate_cp_vton()
    # elif python_arg.VITON_Name == 'CP_VTON_plus':
    #     evaluate_cp_vton_plus()    
    # elif python_arg.VITON_Name == 'PF_AFN':
    #     evaluate_pfafn()