import torch.nn as nn
import torch.nn.functional as F
import numpy as np # linear algebra
import torch


class MLP(nn.Module):
    def __init__(self,needed_dim=None,model_type='reg',n_classes=None):
        super(MLP, self).__init__()
        if  needed_dim is None:
            print('needing information about the input dim')
            exit()
        if model_type == 'reg':
            last_layer = nn.Linear(1000, 1)
        elif model_type =='cls':
            last_layer = nn.Linear(1000, n_classes)
            
        self.layers = nn.Sequential(
            nn.Linear(needed_dim, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            last_layer
        )
        
    def forward(self, x):
        # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x     


if __name__ == '__main__':
    from mood_dataset import MOOD
    from torch.utils.data import Dataset, DataLoader
    X = np.load('bined_x.npy')
    y = np.load('bined_y.npy')
    print(X.shape,y.shape)
    dataset_reg = MOOD(X, y, model_type='reg',data_type='train',debug_mode=True)
    reg_loader = DataLoader(dataset=dataset_reg, batch_size=1, shuffle=True)
    dataset_cls = MOOD(X, y, model_type='cls',data_type='train',debug_mode=True)
    cls_loader = DataLoader(dataset=dataset_reg, batch_size=1, shuffle=True)
    print('testing models...')
    print('mlp regression')

    reg_mlp = MLP(needed_dim=27,model_type='reg',n_classes=None)
    cls_mlp = MLP(needed_dim=27,model_type='cls',n_classes=10)
    for inp,trg in reg_loader:
        out = reg_mlp(inp)
        print(out)
        break
    for inp,trg in cls_loader:
        out = cls_mlp(inp)
        print(out)
        break