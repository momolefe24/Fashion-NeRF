import torch.utils.data
from VITON.Parser_Based.ACGPN.ACGPN_train.data.base_data_loader import BaseDataLoader

def CreateDataset(opt, root_opt):
    dataset = None
    from VITON.Parser_Based.ACGPN.ACGPN_train.data.aligned_dataset import AlignedDataset
    dataset = AlignedDataset()
    

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt, root_opt)
    dataset.__getitem__(0)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt, root_opt):
        BaseDataLoader.initialize(self, opt, root_opt)
        self.dataset = CreateDataset(opt, root_opt)
        
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))
        
        self.test_dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))
        
    def load_data(self):
        return self.dataloader

    def load_test_data(self):
        return self.test_dataloader
    
    def __len__(self):
        return len(self.dataset)
