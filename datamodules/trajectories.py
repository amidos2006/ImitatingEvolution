import pytorch_lightning as pl
from datasets.trajectories import Trajectories as TDataset
from torch.utils.data import random_split, DataLoader

class Trajectories(pl.LightningDataModule):
    def __init__(self, train_batch_size, chromosomes, size, channels, type, portion, bins, repeats, early_threshold):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.chromosomes = chromosomes
        self.size = size
        self.channels = channels
        self.type = type
        self.portion = portion
        self.bins = bins
        self.repeats = repeats
        self.early_threshold = early_threshold

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.data_train = TDataset(
                chromosomes=self.chromosomes, 
                size=self.size, 
                channels=self.channels, 
                type=self.type, 
                portion=self.portion, 
                bins=self.bins, 
                repeats=self.repeats, 
                early_threshold=self.early_threshold
            )
            self.data_train.setup()
        elif stage == "infer":
            print("nothing here yet")


    
    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.train_batch_size, shuffle=True)
    
    # def val_dataloader(self):
    #     return DataLoader(self.data_val, batch_size=self.train_batch_size)
    
    # def test_dataloader(self):
    #     return DataLoader(self.data_test, batch_size=self.test_batch_size)