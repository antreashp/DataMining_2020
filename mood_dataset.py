import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
class MOOD(Dataset):
    def __init__(self, X, y=None, model_type=None,data_type='train',debug_mode=False):
        self.X = X
        # print(X.shape)
        if debug_mode:
            if type == 'train': # debug_mode
                self.paths = self.paths[:10]
        self.y = y
        self.model_type = model_type
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        datapoint = torch.from_numpy(self.X[index])
        if self.model_type == 'reg':
            return datapoint.float(), torch.from_numpy(np.array([self.y[index]])).float() /10
        elif self.model_type == 'cls':
            return datapoint.float(), torch.from_numpy(np.array([self.y[index]])).float()
            
