import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torchvision.transforms as transforms
import scipy.misc as misc
class MOOD(Dataset):
    def __init__(self, X, y=None, model_type=None,data_type='train',debug_mode=False):
        self.X = X
        # print(X.shape)
        if debug_mode:
            if type == 'train': # debug_mode
                self.paths = self.paths[:10]
        self.y = y
        # print(y[y==None])
        self.model_type = model_type
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        datapoint = torch.from_numpy(self.X[index])
        # print([self.y[index]])
        # print(np.array([self.y[index]]))
        # print(datapoint)
        # print(torch.from_numpy(np.array([self.y[index]])).float() /10)
        if self.model_type == 'reg':
            return datapoint.float(), torch.from_numpy(np.array([self.y[index]])).float() /9
        elif self.model_type == 'cls':
            return datapoint.float(), torch.from_numpy(np.array([self.y[index]])).float()
            
class HIL_MOOD(Dataset):
    def __init__(self, X, model_type=None,data_type='train',debug_mode=False):
        self.X = X
        # print(X.shape)
        filepath = 'hilbert_data'
        if debug_mode:
            if type == 'train': # debug_mode
                self.paths = self.paths[:10]
        self.transform = transforms.Compose([
                        # transforms.CenterCrop(10),
                        # transforms.Normalize(0.5, std, inplace=False)
                        transforms.ToPILImage(),
                        transforms.Grayscale(),
                        transforms.Resize((64,64)),
                        transforms.ToTensor(),
                            ])

        # print(y[y==None])
        self.model_type = model_type
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        datapoint = misc.imread('hilbert_data/'+self.X[index])
        y = np.array([float(self.X[index].split('.jpg')[0].split('mood')[1])])
        # print([self.y[index]])
        # print(datapoint.shape)
        # print(y.shape)
        datapoint = self.transform(datapoint)
        
        # print(np.array([self.y[index]]))
        # print(datapoint.shape)
        # print(y)
        # print(torch.from_numpy(np.array([self.y[index]])).float() /10)
        if self.model_type == 'reg':
            return datapoint.float(), torch.from_numpy(y).float() /9
        elif self.model_type == 'cls':
            return datapoint.float(), torch.from_numpy(y).float()
            
