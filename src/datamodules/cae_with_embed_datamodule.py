from typing import Optional
import math

import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import T_co
from sklearn.preprocessing import StandardScaler


class TransactionsDatasetWithEmbed(Dataset):

    def __init__(self, data_file, max_length, drop_time, with_anomalies):
        super(TransactionsDatasetWithEmbed, self).__init__()
        self.max_length = max_length
        self.data_file = data_file
        self.drop_time = drop_time
        all_data = pd.read_csv(data_file, index_col=[0])
        print('Dataset saved in RAM')

        self.scaled_data = all_data[
            ['client_id', 'trans_date', 'amount_rur'] if not drop_time else ['client_id', 'amount_rur']
        ]
        self.embed_small_group = all_data[['small_group']]

        print('Starting to fit scaler for data')
        self.sc = StandardScaler().fit(self.scaled_data)
        print('Scaler fitted')

    def __getitem__(self, index) -> T_co:
        sample = self.sc.transform(self.scaled_data.iloc[index * self.max_length:(index + 1) * self.max_length]).T
        embed = self.embed_small_group.iloc[index * self.max_length:(index + 1) * self.max_length].values
        # Scaling time column
        if not self.drop_time:
            sample[1, :] -= sample[1, 0]

        sample = torch.tensor(sample).float()
        embed = torch.tensor(embed).squeeze_().int()

        return sample, embed

    def __len__(self):
        return self.scaled_data.shape[0] // self.max_length


class TransactionDataModuleWithEmbed(LightningDataModule):

    def __init__(self,
                 data_train_file='.\\',
                 data_test_file='.\\',
                 batch_size=64,
                 max_length=40,
                 drop_time=False,
                 with_anomalies=False):
        super(TransactionDataModuleWithEmbed, self).__init__()
        self.data_train_file = data_train_file
        self.data_test_file = data_test_file
        self.batch_size = batch_size
        self.max_length = max_length
        self.drop_time = drop_time
        self.with_anomalies = with_anomalies
        self.test = None
        self.train = None
        self.val = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in [None, 'fit']:
            train_val = TransactionsDatasetWithEmbed(self.data_train_file, self.max_length, self.drop_time, self.with_anomalies)
            l = len(train_val)
            self.train, self.val = random_split(train_val, (int(l * 0.8), l - int(l * 0.8)))

        if stage in [None, 'test']:
            self.test = TransactionsDatasetWithEmbed(self.data_test_file, self.max_length, self.drop_time, self.with_anomalies)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          shuffle=True)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test,
                          batch_size=self.batch_size)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val,
                          batch_size=self.batch_size)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def collate_fn(self, data):
        data = pad_sequence(data, batch_first=True).reshape(self.batch_size, 4, -1).float()

        return data
