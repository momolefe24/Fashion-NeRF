def evaluate_acgpn(opt, root_opt):
    pass

def evaluate_hrviton(opt, root_opt):
    from VITON.Parser_Based.HR_VITON.evaluate import evaluate_hrviton
    evaluate_hrviton(opt, root_opt)   
        
def evaluate_cp_vton(opt, root_opt):
    from VITON.Parser_Based.CP_VTON.evaluate import evaluate_cp_vton
    evaluate_cp_vton(opt, root_opt)

def evaluate_cp_vton_plus(opt, root_opt):
    from VITON.Parser_Based.CP_VTON_plus.evaluate import evaluate_cp_vton_plus
    evaluate_cp_vton_plus(opt, root_opt)

def evaluate_dmvton(opt, root_opt):
    from VITON.Parser_Free.DM_VTON.evaluate import evaluate_dm_vton
    evaluate_dm_vton(opt, root_opt)
    
def evaluate_pfafn(opt, root_opt):
    from VITON.Parser_Free.PF_AFN.PF_AFN_test.evaluate import evaluate_pfafn
    evaluate_pfafn(opt, root_opt)
    
def evaluate_viton(opt, root_opt, VITON_Name):
    if VITON_Name == 'HR_VITON':
        evaluate_hrviton(opt, root_opt)
    elif VITON_Name == 'ACGPN':
        evaluate_acgpn(opt, root_opt)
    elif VITON_Name == 'DM_VTON':
        evaluate_dmvton(opt, root_opt)
    elif VITON_Name == 'PF_AFN':
        evaluate_pfafn(opt, root_opt)
    elif VITON_Name == 'CP_VTON':
        evaluate_cp_vton(opt, root_opt)
    elif VITON_Name == 'CP_VTON_plus':
        evaluate_cp_vton_plus(opt, root_opt)