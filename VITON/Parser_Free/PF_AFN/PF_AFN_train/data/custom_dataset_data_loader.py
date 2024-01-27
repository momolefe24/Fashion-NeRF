import torch.utils.data
from VITON.Parser_Free.PF_AFN.PF_AFN_train.data.base_data_loader import BaseDataLoader


def CreateDataset(opt, root_opt):
    dataset = None
    from VITON.Parser_Free.PF_AFN.PF_AFN_train.data.aligned_dataset import AlignedDataset
    dataset = AlignedDataset()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt, root_opt)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt, root_opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt, root_opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.viton_batch_size,
            shuffle=not opt.shuffle,
            num_workers=1)

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
