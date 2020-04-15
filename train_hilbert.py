import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np # linear algebra
import matplotlib.pyplot as plt
import torch

from sklearn.preprocessing import StandardScaler
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
from sklearn.decomposition import PCA

from mood_dataset import HIL_MOOD
from models import Net
def accat(out,trg,thresh=0.05 ):
        out = out.detach().cpu().numpy().squeeze()
        trg = trg.detach().cpu().numpy().squeeze()
        diff = np.abs(out - trg)
        # print(diff)
        diff[diff > thresh] = 1
        diff[diff <= thresh] = 0
        diff = (diff * -1) + 1
        correct = np.sum(diff)
        # print(correct)
        return correct
def train(options):
    exp_name = options['exp_name']
    batch_size = options['batch_size']
    use_pca = options['use_pca']
    model_type = options['model_type']
    loss_fn = options['loss_fn']
    optim = options['optim']
    use_scheduler = options['use_scheduler']
    lr = options['lr']
    epochs = options['epochs']
    pca_var_hold = options['pca_var_hold']
    debug_mode = options['debug_mode']
    
    if os.path.exists(exp_name):
        shutil.rmtree(exp_name)

    time.sleep(1)
    writer = SummaryWriter(exp_name,flush_secs=1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = os.listdir('hilbert_data')
    X_train = X[:int(0.8*len(X))]
    X_test = X[int(0.8*len(X)):]
    # X = np.load('bined_x.npy')
    # y = np.load('bined_y.npy')
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # if use_pca and 'Raw' in exp_name:
    #     scaler = PCA(pca_var_hold)
    #     scaler.fit(X_train)
    #     X_train = scaler.transform(X_train)
    #     X_test = scaler.transform(X_test)

    # needed_dim = X_train.shape[1]

    dataset_train = HIL_MOOD(X_train, model_type=model_type,data_type='train',debug_mode=debug_mode)
    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    
    dataset_val = HIL_MOOD(X_test, model_type=model_type,data_type='val')
    valid_loader = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False)
    
    model = Net()
    model.to(device)
    if optim == None:
        print('you need to specify an optimizer')
        exit()
    elif optim == 'adam':
        optimizer = torch.optim.Adam(   model.parameters(), lr=lr)
    elif optim == 'sgd':
        optimizer = torch.optim.SGD(   model.parameters(), lr=lr,momentum=0.9)
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',verbose=True,threshold=0.0001,patience = 10)
    if loss_fn == None:
        print('you need to specify an optimizer')
        exit()
    else:

        if loss_fn == 'mse':

            loss_fn = torch.nn.MSELoss()
        elif loss_fn == 'cross_entropy':
            loss_fn = torch.nn.CrossEntropyLoss()
    
    
    
    mean_train_losses = []
    mean_valid_losses = []
    valid_acc_list = []
    best = 0  #small number for acc big number for loss to save a model
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        valid_losses = []
        for i, (images, labels) in enumerate(train_loader):
            if images.shape[0] != batch_size:
                continue
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            # print(images.shape)
            outputs = model(images)
            # print(images.shape)
            # print(outputs.shape)
            # print(labels.shape)
            # print(i)
            loss =loss_fn(outputs,labels)
            # print('loss: ',loss.item())
            writer.add_scalar('Loss/train', loss.item(), len(train_loader)*epoch+i)

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            del outputs
            # if (i * batch_size) % (batch_size * 100) == 0:
            #     print(f'{i * batch_size} / 50000')
                
        model.eval()
        correct_5_2 = 0
        correct_5_1 = 0
        
        total_loss = 0
        total = 0
        accsat =[0.5,0.05,0.005]
        accs = np.zeros(len(accsat))
        # corrs = np.zeros(len(accsat))
        correct_array = np.zeros(len(accsat))
        with torch.no_grad():
            for i, (images, labels) in enumerate(valid_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss =  loss_fn(outputs, labels)

                
                for i in range(len(accsat)):

                    correct_array[i] += accat(outputs,labels,thresh=accsat[i])

                # total_loss += loss.item()
                total += labels.size(0)
                
                
                valid_losses.append(loss.item())


                
        mean_train_losses.append(np.mean(train_losses))
        mean_valid_losses.append(np.mean(valid_losses))
        # scheduler.step(np.mean(valid_losses))
        for i in range(len(accsat)):
            accs[i] = 100*correct_array[i]/total
            writer.add_scalar('Acc/val_@'+str(accsat[i]), accs[i], epoch)
        
        if np.mean(valid_losses) < best:
            best = np.mean(valid_losses)
            torch.save(model.state_dict(),os.path.join(os.getcwd(),'models','meh.pth'))
        
        writer.add_scalar('Loss/val', np.mean(valid_losses), epoch)
        # valid_acc_list.append(accuracy)
        if epoch ==epochs-1:
            print('epoch : {}, train loss : {:.4f}, valid loss : {:.4f}, acc@0.05 : {:.4f}'\
                .format(epoch+1, np.mean(train_losses), np.mean(valid_losses), accsat[1]))
if __name__ == '__main__':
    options ={'exp_name'      : 'runs/Hilbert_reg_pca_128_0003_win5',
              'batch_size'    : 128,
              'epochs'        : 50,
              'lr'            : 0.0003,
              'use_pca'       : True,
              'pca_var_hold'  : 0.995,
              'model_type'    : 'reg', #'cls'
              'loss_fn'       : 'mse', #cross-entropy
              'optim'         : 'adam',#sgd
              'use_scheduler' : False, #true decreaseing  
              'debug_mode'    : False  #makes training set smaller 
    }
    train(options)