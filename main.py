from pytorch_lightning import Trainer, loggers
from torchsummary import summary
import torch
import os

from src import Conv1dAutoEncoder, LSTMAutoEncoder, TransactionDataModuleNewData, LSTMAutoEncoderEmbed


def test_lstm_network(train_dataset):
    model = LSTMAutoEncoder(40, 3)
    logger = loggers.TensorBoardLogger('lightning_logs_new', 'lstm')
    trainer = Trainer(gpus=0, max_epochs=20, logger=logger)
    #dm = TransactionDataModule(train_dataset, test_dataset, drop_time=
    dm = TransactionDataModuleNewData(train_dataset)
    trainer.fit(model, dm)
    trainer.test(model, dm)

def test_lstm_network_embed(train_dataset):
    model = LSTMAutoEncoderEmbed(17, 4)
    logger = loggers.TensorBoardLogger('lightning_logs_new', 'lstm')
    trainer = Trainer(gpus=0, max_epochs=20, logger=logger)
    #dm = TransactionDataModule(train_dataset, test_dataset, drop_time=
    dm = TransactionDataModuleNewData(train_dataset)
    trainer.fit(model, dm)
    trainer.test(model, dm)

def test_cae_network(train_dataset):
    model = Conv1dAutoEncoder(1, 8)
    logger = loggers.TensorBoardLogger('lightning_logs', 'cae')
    trainer = Trainer(gpus=1, max_epochs=4, logger=logger)
    dm = TransactionDataModuleNewData(train_dataset)

    trainer.fit(model, dm)
    # trainer.test(model, dm)


def test_cae_with_embed_network(train_dataset, test_dataset):
    model = Conv1dEmbedAutoEncoder(4, 8)
    logger = loggers.TensorBoardLogger('lightning_logs_new', 'cae_with_embed')
    trainer = Trainer(gpus=0, max_epochs=20, logger=logger)
    dm = TransactionDataModule(train_dataset, test_dataset)

    trainer.fit(model, dm)
    trainer.test(model, dm)


def get_summary(model, device):
    model = model.to(device)
    return summary(model, (3, 40), batch_size=1)

def test_lstm_freeze(train_dataset):
    model = LSTMAutoEncoderEmbed(17, 4)
    checkpoint = torch.load('.\\lightning_logs_new\\lstm\\version_4\\checkpoints\\epoch=19-step=87780.ckpt')
    model.load_state_dict(checkpoint['state_dict'])
    logger = loggers.TensorBoardLogger('lightning_logs_new', 'lstm')
    trainer = Trainer(gpus=0, max_epochs=20, logger=logger)
    #dm = TransactionDataModule(train_dataset, test_dataset, drop_time=
    dm = TransactionDataModuleNewData(train_dataset)
    #trainer.fit(model, dm)
    trainer.test(model, dm)


if __name__ == '__main__':
    # test_lstm_network()
    #print(os.listdir(os.path.join('C:\\','Users')))
    #print(os.listdir(os.path.join('src','networks')))
    #print(os.path.join('src','networks'))
    #print(os.listdir('C:\\Users\\User\\Desktop\\MSD\\project\\data\\normal'))
    #data_folder = 'C:\\Users\\User\\Desktop\\MSD\\project\\data\\normal'
    data_folder = '.\\data\\normal\\'
    #test_cae_network(data_folder)
    test_lstm_freeze(data_folder)
