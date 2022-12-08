__all__ = ["Conv1dAutoEncoder"]

import time

import pytorch_lightning as pl
import torch
import torch.nn as nn


def init_weights(m):
    """
    Simple weight initialization
    """
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
        m.bias.data.fill_(0.01)


class Conv1dAutoEncoder(pl.LightningModule):

    def __init__(
        self,
        in_channels: int, # without mcc
        n_latent_features: int,
        n_embedding: int = 16,
        n_vocab_size: int = 386
    ):
        super().__init__()
        self.out = n_latent_features

        self.embed = nn.Embedding(n_vocab_size + 1, n_embedding, 0)

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=in_channels + n_embedding, out_channels=512, kernel_size=3),
            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3),
            nn.BatchNorm1d(256),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.Conv1d(in_channels=32, out_channels=self.out, kernel_size=3),
        )
        self.encoder.apply(init_weights)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=self.out, out_channels=32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ConvTranspose1d(in_channels=64, out_channels=128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ConvTranspose1d(in_channels=128, out_channels=256, kernel_size=3),
            nn.BatchNorm1d(256),
            nn.ConvTranspose1d(in_channels=256, out_channels=512, kernel_size=3),
            nn.BatchNorm1d(512),
            nn.ConvTranspose1d(
                in_channels=512, out_channels=in_channels + n_embedding, kernel_size=3
            ),
        )
        self.decoder.apply(init_weights)

        self.train_index = 0
        self.val_index = 0
        self.final_labels = None
        self.time_start = time.time()

    def forward(self, x):
        """
        Returns embeddings
        """
        x_without_mcc = x[:, :, 1:]
        if x_without_mcc.dim() < 3:
            x_without_mcc = x_without_mcc.unsqueeze(-1)
        mcc = x[:, :, 0].long()
        mcc = self.embed(mcc)
        x_embed = torch.concat((x_without_mcc, mcc), -1)
        x_embed = x_embed.permute((0, 2, 1))

        latent = self.encoder(x_embed)
        return x_embed, latent

    # TODO разобраться, нужен ли вообще этот метод
    def predict_step(self, x):
        x_embed, latent = self(x)
        loss = torch.nn.MSELoss()(self.decoder(latent), x_embed)
        return {'latent': latent, 'loss': loss}

    def training_step(self, batch, batch_idx):
        x = batch
        x_embed, latent = self(x)
        loss = torch.nn.MSELoss()(self.decoder(latent), x_embed)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        self.log("train_time", time.time() - self.time_start, prog_bar=False)

    def validation_step(self, batch, batch_idx):
        x = batch
        x_embed, latent = self(x)
        loss = torch.nn.MSELoss()(self.decoder(latent), x_embed)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        pass

    def test_step(self, batch, batch_idx: int):
        x = batch
        x_embed, latent = self(x)
        loss = torch.nn.MSELoss()(self.decoder(latent), x_embed)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "latent": latent}

    def test_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=.003)

        def adjust_lr(epoch):
            if epoch < 10:
                return 3e-3
            if 10 <= epoch < 20:
                return 1e-3
            if 20 <= epoch < 50:
                return 3e-4
            if 50 <= epoch < 100:
                return 3e-5
            else:
                return 3e-6

        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=adjust_lr
            ),
            "name": "lr schedule",
        }
        return [optimizer], [lr_scheduler]
