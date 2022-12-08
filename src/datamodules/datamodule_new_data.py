from typing import Optional
import os
import pickle

import pandas as pd
import numpy as np

import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import T_co

class TransactionDatasetNewData(Dataset):

    def __init__(self, data_dir: str,
                 drop_time: bool = True,
                 time_column: Optional[str] = 'time',
                 drop_user_name: bool = True,
                 user_name_column: Optional[str] = 'user_id') -> None:
        super().__init__()

        self.data_dir = data_dir
        self.drop_time = drop_time
        self.time_column = time_column
        self.drop_user_name = drop_user_name
        self.user_name_column = user_name_column

        self.compare = dict(pickle.load(open('src\\datamodules\\compare_table', 'rb')))

    def __getitem__(self, index: int) -> T_co:
        sample = pd.read_csv(os.path.join(self.data_dir, f'{index}.csv'))
        if self.drop_time:
            sample.drop(columns=self.time_column, inplace=True)
        if self.drop_user_name:
            sample.drop(columns=self.user_name_column, inplace=True)
        
        sample = sample.values
        sample = np.apply_along_axis(lambda x: [self.compare[x[0]], x[1]], 1, sample)

        return torch.tensor(sample).float()

    def __len__(self):
        return len(os.listdir(self.data_dir))


class TransactionDataModuleNewData(LightningDataModule):

    def __init__(self,
                 data_train_folder: str = '.\\',
                 train_val_split: float = .8,
                 batch_size: int = 64,
                 **params) -> None:
        super().__init__()

        self.data_train_folder = data_train_folder
        self.train_val_split = train_val_split
        self.batch_size = batch_size
        self.params_dataset = params

        self.train = None
        self.val = None

    def setup(self, stage: str) -> None:
        if stage in (None, 'fit'):
            train_val = TransactionDatasetNewData(self.data_train_folder,
                                                  **self.params_dataset)

            l = len(train_val)
            self.train, self.val = random_split(train_val,
                                               (int(l * self.train_val_split),
                                                l - int(l * self.train_val_split)))
        
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          shuffle=True,
                          collate_fn=self._collate_fn)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val,
                          batch_size=self.batch_size,
                          collate_fn=self._collate_fn)

    def _collate_fn(self, data):
        data = pad_sequence(data, batch_first=True)
        return data
