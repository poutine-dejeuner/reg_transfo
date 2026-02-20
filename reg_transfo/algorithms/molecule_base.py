import lightning.pytorch as pl
import torch
import torch.nn as nn


class MoleculeRegressor(pl.LightningModule):
    def __init__(self, lr: float = 1e-3, weight_decay: float = 0.0):
        super().__init__()
        # self.save_hyperparameters() # Subclasses will handle save_hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay

    def training_step(self, batch, batch_idx):
        preds = self(batch)
        loss = nn.MSELoss()(preds, batch.y)
        self.log('train/loss', loss, batch_size=batch.y.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        preds = self(batch)
        mse = nn.MSELoss()(preds, batch.y)
        mae = nn.L1Loss()(preds, batch.y)
        self.log('val/loss', mse, batch_size=batch.y.size(0))
        self.log('val/mae', mae, batch_size=batch.y.size(0))
        return mse

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
