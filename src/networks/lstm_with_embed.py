from typing import Any, Optional

from pytorch_lightning import LightningModule
import time
import torch
from torch import Tensor
import torch.nn as nn
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT


def init_weights(m):
    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
    m.bias.data.fill_(0.01)


class LSTMAutoEncoderEmbed(LightningModule):

    def __init__(self, n_features, embedding_dim=4, num_layers: int = 10, n_embedding: int = 16, n_vocab_size: int = 386):
        super(LSTMAutoEncoderEmbed, self).__init__()

        self.embed = nn.Embedding(n_vocab_size + 1, n_embedding, 0)
        self.encoder1 = nn.LSTM(input_size=n_features,
                                hidden_size=embedding_dim * 2,
                                num_layers=1,
                                batch_first=True)
        self.encoder2 = nn.LSTM(input_size=embedding_dim * 2,
                                hidden_size=embedding_dim,
                                num_layers=1,
                                batch_first=True)

        self.decoder1 = nn.LSTM(input_size=embedding_dim,
                                hidden_size=embedding_dim * 2,
                                num_layers=1,
                                batch_first=True)
        self.decoder2 = nn.LSTM(input_size=embedding_dim * 2,
                                hidden_size=embedding_dim * 2,
                                num_layers=1,
                                batch_first=True)

        self.output_layer = nn.Linear(embedding_dim * 2, n_features)

        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.time_start = time.time()

    def predict_step(self, x: Any) -> Any:
        seq_len = x.shape[1] # bs x L x 2
        x_out, x_embed, latent = self(x)

        return {'loss': torch.nn.MSELoss()(x_embed, x_out), 'latent': torch.mean(latent,1)}

    def forward(self, x: Tensor, *args, **kwargs) -> Any:
        #print(x.shape, 'Very beginning')
        x_without_mcc = x[:, :, 1:] #Bs x L x 2 ==mcc+value
        if x_without_mcc.dim() < 3:
            x_without_mcc = x_without_mcc.unsqueeze(-1)
        mcc = x[:, :, 0].long() #true value of mcc
        mcc = self.embed(mcc) #embedding of mcc len = 16
        x_embed = torch.concat((x_without_mcc, mcc), -1) #x_embed: Bs x L x 16+1
        #print(x_embed.shape, 'mcc embedding + transaction')

        x_ = x_embed # Bs x L x 17
        x_, (hidden_n, _) = self.encoder1(x_) #Bs x L x emb_dim*2
        #print(x_.shape, 'output lstm 1')
        x_, (hidden_n, _) = self.encoder2(x_) #bs x L x emb_dim
        #print(x_.shape, 'output lstsm 2')
        latent = x_

        out, (h, c) = self.decoder1(x_) #bs x L x emb_dim*2
        #print(out.shape, 'decoder 1')
        out, (h, c) = self.decoder2(out) #bs x L x emb_dim*2
        #print(out.shape, 'decoder 2')
        x_out = self.output_layer(out) #bs x L x n_features (17)

        return x_out, x_embed, latent

    def training_step(self, batch: Tensor, *args, **kwargs) -> STEP_OUTPUT:
        x_out, x_embed, latent = self(batch)

        loss = torch.nn.MSELoss()(x_embed, x_out)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss}

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.log('train_time', time.time() - self.time_start, prog_bar=True)

    def validation_step(self, batch: Tensor, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        x_out, x_embed, latent = self(batch)
        loss = torch.nn.MSELoss()(x_embed, x_out)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss}

    def test_step(self, batch: Tensor, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        x_out, x_embed, latent = self(batch)
        loss = torch.nn.MSELoss()(x_embed, x_out)

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.003)

        def adjust_lr(epoch):
            if epoch < 20:
                return 0.003
            if 20 <= epoch < 50:
                return 0.001
            if 50 <= epoch < 80:
                return 0.0003
            if 80 <= epoch < 120:
                return 0.00003
            else:
                return 0.000003

        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=adjust_lr
            ),
            "name": "lr schedule",
        }
        return [optimizer], [lr_scheduler]
