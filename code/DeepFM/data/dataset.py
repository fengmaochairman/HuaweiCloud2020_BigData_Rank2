import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

class CriteoDataset(Dataset):
    """
    Custom dataset class for Criteo dataset in order to use efficient 
    dataloader tool provided by PyTorch.
    """ 
    def __init__(self, df, train = True):
        """
        Initialize file path and train/test mode.

        Inputs:
        - root: Path where the processed data file stored.
        - train: Train or test. Required.
        """
        self.train = train

        if not self._check_exists:
            raise RuntimeError('Dataset not found.')

        if self.train:
            self.train_data = df.iloc[:, :-1].values
            self.target = df.iloc[:, -1].values
        else:
            self.test_data = df.iloc[:, :-1].values
    
    def __getitem__(self, idx):
        if self.train:
            dataI, targetI = self.train_data[idx, :], self.target[idx]
            Xi = torch.from_numpy(dataI.astype(np.int32)).unsqueeze(-1)
            Xv = torch.from_numpy(np.ones_like(dataI))
            return Xi, Xv, targetI
        else:
            dataI = self.test_data.iloc[idx, :]
            Xi = torch.from_numpy(dataI.astype(np.int32)).unsqueeze(-1)
            Xv = torch.from_numpy(np.ones_like(dataI))
            return Xi, Xv

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(self.root)