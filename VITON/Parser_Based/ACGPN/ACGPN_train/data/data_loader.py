
def CreateDataLoader(opt, root_opt):
    from VITON.Parser_Based.ACGPN.ACGPN_train.data.custom_dataset_data_loader import CustomDatasetDataLoader
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt, root_opt)
    return data_loader
