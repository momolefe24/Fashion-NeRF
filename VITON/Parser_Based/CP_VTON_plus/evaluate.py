from VITON.Parser_Based.CP_VTON_plus.test import process_opt
from metrics import *

def evaluate_cp_vton_plus(opt, root_opt):
    opt,root_opt = process_opt(opt, root_opt)
    evaluate(opt)
    
