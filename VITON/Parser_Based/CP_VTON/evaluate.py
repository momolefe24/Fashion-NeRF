from VITON.Parser_Based.CP_VTON.test import process_opt
from metrics import *

def evaluate_cp_vton(opt, root_opt):
    opt,root_opt = process_opt(opt, root_opt)
    evaluate(opt)
    
