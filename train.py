import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np # linear algebra
import matplotlib.pyplot as plt
import torch

from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import os,sys
from shutil import copyfile
import shutil
import time
import numpy as np



exp_name = 'runs/L1_normal'
batch_size =128
if os.path.exists(exp_name):
    shutil.rmtree(exp_name)

time.sleep(1)
writer = SummaryWriter(exp_name,flush_secs=1)


class MOOD(Dataset):
    def __init__(self, X, y=None, transform=None,type='train'):
        self.X = X
        # if type == 'train':
        #     self.paths = self.paths[:10]
        self.y = y
        self.transform = transforms.Compose([

                        transforms.ToPILImage(),
                        # transforms.Resize((1600,1)),
                        transforms.RandomCrop((512,1)),
                        transforms.ToTensor(),
                            ])
        self.type = type
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        
        # image = np.load(os.path.join(os.getcwd(),'data','audio_1d',self.paths[index])).reshape(-1,1)
        datapoint = torch.from_numpy(self.X[index])
        # if self.transform is not None:
        #     image = self.transform(image)

        if self.y is not None:
            return datapoint.float(), torch.from_numpy(np.array([self.y[index]])).float()
        else:
            label = self.y[index]
            # label = int(self.paths[index].split('_')[1])
            # y_onehot = label
            
            return datapoint.float(),label.float()
class MLP(nn.Module):
    def __init__(self,needed_dim=None):
        super(MLP, self).__init__()
        if  needed_dim is None:
            print('needing information about the input dim')
            sys.exit()
        self.layers = nn.Sequential(
            nn.Linear(needed_dim, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1)
        )
        
    def forward(self, x):
        # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x     




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if 'L1' in exp_name:
    data_path = 'L1_data.npy' 
elif 'var' in exp_name:
    data_path = 'var_thresh_data.npy' 
elif 'Raw' in exp_name:
    data_path = 'bined_x.npy' 
elif 'tree' in exp_name:
    data_path = 'tree_data.npy' 
    

X = np.load(data_path)
needed_dim = X.shape[1]
y = np.load('bined_y.npy')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dataset_train = MOOD(X_train,y_train,type='train')
dataset_val = MOOD(X_test,y_test,type='val')
train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False)


model =  MLP(needed_dim=needed_dim)
model.to(device)


optimizer = torch.optim.Adam(   model.parameters(), lr=0.03)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',verbose=True,threshold=0.0001,patience = 10)
loss_fn = torch.nn.MSELoss()


mean_train_losses = []
mean_valid_losses = []
valid_acc_list = []
epochs = 50
best = 0


for epoch in range(epochs):
    model.train()
    
    train_losses = []
    valid_losses = []
    for i, (images, labels) in enumerate(tqdm(train_loader)):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        # print(images.shape)
        outputs = model(images)

        loss =loss_fn(outputs,labels)
        # print('loss: ',loss.item())
        writer.add_scalar('Loss/train', loss.item(), len(train_loader)*epoch+i)

        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        del outputs
        if (i * batch_size) % (batch_size * 100) == 0:
            print(f'{i * batch_size} / 50000')
            
    model.eval()
    correct = 0
    total_loss = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(valid_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss =  loss_fn(outputs, labels)
            
            # _,outputs = torch.max(outputs, 1)
            # correct += (outputs == labels).sum().item()
            # total_loss += loss.item()
            # total += labels.size(0)
            
            
            valid_losses.append(loss.item())


            
    mean_train_losses.append(np.mean(train_losses))
    mean_valid_losses.append(np.mean(valid_losses))
    # scheduler.step(np.mean(valid_losses))
    # accuracy = 100*correct/total
    if np.mean(valid_losses) < best:
        best = np.mean(valid_losses)
        torch.save(model.state_dict(),os.path.join(os.getcwd(),'models','meh.pth'))
    # writer.add_scalar('Acc/val', accuracy, epoch)
    writer.add_scalar('Loss/val', np.mean(valid_losses), epoch)
    # valid_acc_list.append(accuracy)
    print('epoch : {}, train loss : {:.4f}, valid loss : {:.4f}'\
         .format(epoch+1, np.mean(train_losses), np.mean(valid_losses)))

