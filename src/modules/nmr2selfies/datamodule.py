import pytorch_lightning as pl
from torch.utils.data import DataLoader

from ..config import Config


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.data_dir = cfg.app.data_path
        self.batch_size = cfg.train.batch_size

    def prepare_data(self):
        # (how to download, tokenize, etc…)
        return super().prepare_data()

    def setup(self, stage=None):
        # (how to split, define dataset, etc…)
        raise NotImplementedError("MNISTDataModule setup not implemented yet")

    def train_dataloader(self):
        # TODO: implement this method to return a DataLoader for the training set
        return DataLoader()

    def val_dataloader(self):
        raise NotImplementedError(
            "MNISTDataModule val_dataloader not implemented yet")

    def test_dataloader(self):
        raise NotImplementedError(
            "MNISTDataModule test_dataloader not implemented yet")

    def predict_dataloader(self):
        raise NotImplementedError(
            "MNISTDataModule predict_dataloader not implemented yet")
