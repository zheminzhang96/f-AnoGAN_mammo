import logging
import numpy as np
import csv

from .RSNAbreastCancer import *
#from .RSNAtestDataset import *


from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler

# cifar10_tsfms = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

def build_fanogan_dataset(dataset_name, cvae_batch_size):
    #logger = logging.getLogger()
    print("Build f-AnoGAN dataset for {}".format(dataset_name))
    
    assert dataset_name in ['breast', 'breast1', 'breast2', 'try']
        
    if dataset_name == "breast":
        train_df_mlo, test_df_mlo = get_breast_data('MLO')
        train_df_cc, test_df_cc = get_breast_data('CC')
        train_df_mlo.to_csv("train_df_mlo.csv")
        #val_df_mlo.to_csv("val_df_mlo.csv")
        test_df_mlo.to_csv("test_df_mlo.csv")
        train_df_cc.to_csv("train_df_cc.csv")
        #val_df_cc.to_csv("val_df_cc.csv")
        test_df_cc.to_csv("test_df_cc.csv")

        print("training size MLO: ", int(len(train_df_mlo)/2))
        #print("validation size MLO: ", int(len(val_df_mlo)/2))
        print("test size MLO: ", int(len(test_df_mlo)/2))
        
        print("training size CC: ", int(len(train_df_cc)/2))
        #print("validation size CC: ", int(len(val_df_cc)/2))
        print("test size CC: ", int(len(test_df_cc)/2))
        
        train_set_mlo = RSNAbreastCancer_Dataset(train_df_mlo)
        train_set_cc = RSNAbreastCancer_Dataset(train_df_cc)
        train_set = train_set_mlo+train_set_cc
        print("Total training size: ", len(train_set))

        # validate_set_mlo = RSNAbreastCancer_Dataset(val_df_mlo)
        # validate_set_cc = RSNAbreastCancer_Dataset(val_df_cc)
        # validate_set = validate_set_mlo+validate_set_cc
        # print("Total val size: ", len(validate_set))


    ## add test set later: 'test': DataLoader(test_set, batch_size = cvae_batch_size, num_workers = 1)
    ## , 'test':len(test_set)
    fanogan_dataloaders = {'train': DataLoader(train_set, batch_size = cvae_batch_size, shuffle = True, num_workers = 1)}
    fanogan_dataset_sizes = {'train': len(train_set)}
        
    return fanogan_dataloaders, fanogan_dataset_sizes
