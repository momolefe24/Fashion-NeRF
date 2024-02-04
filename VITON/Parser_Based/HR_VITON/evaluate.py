from VITON.Parser_Based.HR_VITON.test_condition import condition_process_opt
from metrics import *

def evaluate_hrviton(opt, root_opt):
    opt,root_opt = condition_process_opt(opt, root_opt)
    evaluate(opt)