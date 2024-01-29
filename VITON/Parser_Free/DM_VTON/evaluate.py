from VITON.Parser_Free.DM_VTON.test import process_opt
from metrics import *

def evaluate_dm_vton(opt, root_opt):
    opt,root_opt = process_opt(opt, root_opt)
    evaluate(opt)
