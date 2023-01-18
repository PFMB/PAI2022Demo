import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.datasets as dset
import torchvision.transforms as transforms
from BNSeg.BNN_seg_src.data.data_camvid import CamVidDataset

def get_dataloader(config):
    data_dir = config.data_dir
    batch_size = config.batch_size
    test_batch_size = 1

    if config.is_train:
        if config.data_name == 'CamVid':
            train_dataset = CamVidDataset(data_dir,usage='train',scale=0.5) # downsample the image size for faster training
            test_dataset = CamVidDataset(data_dir,usage='test',scale=0.5)

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                num_workers=config.num_work, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size,
                                num_workers=config.num_work, shuffle=False)

        print('==>>> total trainning batch number: {}'.format(len(train_loader)))
        print('==>>> total testing batch number: {}'.format(len(test_loader)))

        data_loader = {'train': train_loader, 'test': test_loader}
    else:
        if config.data_name == 'CamVid':
            test_dataset = CamVidDataset(data_dir,usage='demo') # test
        test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size,
                                num_workers=config.num_work, shuffle=False)

        print('==>>> total testing batch number: {}'.format(len(test_loader)))

        data_loader = {'train': None, 'test': test_loader}        

    return data_loader
