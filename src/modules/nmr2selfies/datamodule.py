from ..config import Config
import pytorch_lightning as pl


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Config):
        super().__init__()
        
