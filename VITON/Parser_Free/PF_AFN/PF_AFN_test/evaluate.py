from VITON.Parser_Free.PF_AFN.PF_AFN_test.test import process_opt
from metrics import *

def evaluate_pfafn(opt, root_opt):
    opt,root_opt = process_opt(opt, root_opt)
    evaluate(opt)