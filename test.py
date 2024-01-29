""" ------------------ ACGPN------------------"""
def test_acgpn(opt, root_opt):
    from VITON.Parser_Based.ACGPN.ACGPN_inference.test import test_acgpn_
    test_acgpn_(opt, root_opt)
    
    
""" ------------------ DM_VTON ------------------"""
def test_dmvton(opt, root_opt, VITON_Model):
    from VITON.Parser_Free.DM_VTON.test import test_dm_vton
    if VITON_Model == 'PB_Warp' or VITON_Model == 'PF_Warp':
        test_dm_vton(opt, root_opt)
    # elif VITON_Model == 'PB_Gen' or VITON_Model == 'PF_Gen'
    #     test_dm_vton_gen(opt, root_opt)
        
def test_cp_vton(opt, root_opt): # Warping
    from VITON.Parser_Based.CP_VTON.test import test_cpvton_
    test_cpvton_(opt, root_opt)
    
    
def test_cp_vton_plus(opt, root_opt):
    from VITON.Parser_Based.CP_VTON_plus.test import test_cpvton_plus_
    test_cpvton_plus_(opt, root_opt)
    
    
def test_pfafn(opt, root_opt, VITON_Model):
    from VITON.Parser_Free.PF_AFN.PF_AFN_test.test import test_pfafn_
    if VITON_Model == 'PB_Warp' or VITON_Model == 'PF_Warp':
        test_pfafn_(opt, root_opt)
    # elif VITON_Model == 'PB_Gen' or VITON_Model == 'PF_Gen':
    #     test_pfafn_(opt, root_opt)

def test_hrviton(opt, root_opt, VITON_Model):
    from VITON.Parser_Based.HR_VITON.test_condition import test_hrviton_tocg_
    from VITON.Parser_Based.HR_VITON.test_generator import test_tryon_
    if VITON_Model == 'TOCG':
        test_hrviton_tocg_(opt, root_opt)    
    elif VITON_Model == 'GEN':
        test_tryon_(opt, root_opt)

def test_ladi_vton(opt, root_opt, VITON_Model):
    pass
        
def test_viton(opt, root_opt,VITON_Name, VITON_Model):
    if VITON_Name == 'HR_VITON':
        test_hrviton(opt, root_opt,VITON_Model)
    elif VITON_Name == 'ACGPN':
        test_acgpn(opt, root_opt,VITON_Model)
    elif VITON_Name == 'DM_VTON':
        test_dmvton(opt, root_opt,VITON_Model)
    elif VITON_Name == 'PF_AFN':
        test_pfafn(opt, root_opt,VITON_Model)
    elif VITON_Name == 'CP_VTON':
        test_cp_vton(opt, root_opt,VITON_Model)
    elif VITON_Name == 'CP_VTON_plus':
        test_cp_vton_plus(opt, root_opt,VITON_Model)
             
        
        
