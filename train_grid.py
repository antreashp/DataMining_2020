import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np # linear algebra
import matplotlib.pyplot as plt
import torch

from skopt import dump, load
from dim_reduction import L1_based_selection,tree_selection,remove_with_var_thresh
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
import skopt
SPACE =[skopt.space.Real(0.001, 0.1, name='lr', prior='log-uniform'),
            skopt.space.Integer(5, 9, name='batch_size'),
            skopt.space.Integer(7, 11, name='hid_layer1'),
            skopt.space.Integer(7, 11, name='hid_layer2'),
            skopt.space.Integer(0, 2, name='dim_redu_params'),
            skopt.space.Integer(0, 3, name='name'),
            ]
create_dim_redu = True
experiments = ['Raw','var','L1','tree']
@skopt.utils.use_named_args(SPACE)
def train(**params):
    exp_params = [None, [(.01 * (1 - .01)),(.05 * (1 - .05)),(.1 * (1 - .1))],[0.1,0.05,0.01],[50,100,250]]

    all_params = {**params}
    
    exp_name = experiments[all_params['name']]
    exp_param = exp_params[all_params['name']][all_params['dim_redu_params'] ] if all_params['name'] != 0 else 0 
    lr = all_params['lr']
    batch_size =2**int(all_params['batch_size'])
    exp_name = 'runs/'+exp_name+ '_lr'+str(round(lr,4)) +'_'+ 'bs'+str(batch_size) +'_'+ 'layers'+str(all_params['hid_layer1'])+'_'+str(all_params['hid_layer2'])+'_'+str(round(exp_param,4))

    print(exp_name)
    if os.path.exists(exp_name):
        shutil.rmtree(exp_name)

    time.sleep(1)
    writer = SummaryWriter(exp_name,flush_secs=1)


    class MOOD(Dataset):
        def __init__(self, X, y=None, transform=None,type='train'):
            self.X = X[:500]
            # if type == 'train':
            #     self.paths = self.paths[:10]
            self.y = y
            self.transform = transforms.Compose([

                            transforms.ToPILImage(),
                            # transforms.Resize((1600,1)),
                            # transforms.RandomCrop((512,1)),
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
                nn.Linear(needed_dim, 2**all_params['hid_layer1']),
                nn.ReLU(),
                nn.Linear( 2**all_params['hid_layer1'] , 2**all_params['hid_layer2'] ),
                nn.ReLU(),
                nn.Linear( 2**all_params['hid_layer2'] , 1)
            )
            
        def forward(self, x):
            # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
            x = x.view(x.size(0), -1)
            x = self.layers(x)
            return x     




    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = 'bined_x.npy'
    Raw_x = np.load(data_path)
    # print(Raw_x.shape)
    # with open('test.txt','w') as f:
    #     for line in range(len(Raw_x)):
    #         f.write(str(Raw_x[line]).replace('\n','')+'\n')
    #     # f.writelines(Raw_x)
    # exit()
    y = np.load('bined_y.npy')

    if create_dim_redu:
        try:
            if 'L1' in exp_name:
                X = L1_based_selection(Raw_x,y,exp_param)    
            elif 'var' in exp_name:
                X = remove_with_var_thresh(Raw_x,exp_param)     
            elif 'Raw' in exp_name:
                X = Raw_x
            elif 'tree' in exp_name:
                X = tree_selection(Raw_x,y,exp_param)    
        except:
            if 'L1' in exp_name:
                data_path = 'L1_data.npy' 
            elif 'var' in exp_name:
                data_path = 'var_thresh_data.npy' 
            elif 'Raw' in exp_name:
                data_path = 'bined_x.npy' 
            elif 'tree' in exp_name:
                data_path = 'tree_data.npy'
            X = np.load(data_path)
    else:
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    dataset_train = MOOD(X_train,y_train,type='train')
    dataset_val = MOOD(X_test,y_test,type='val')
    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False)

    try:
    
        model =  MLP(needed_dim=needed_dim)
    except:
        print(X.shape)
        print(needed_dim)
        return 10
    model.to(device)


    optimizer = torch.optim.Adam(   model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',verbose=True,threshold=0.0001,patience = 10)
    loss_fn = torch.nn.MSELoss()


    mean_train_losses = []
    mean_valid_losses = []
    valid_acc_list = []
    epochs = 7
    best =10  #small number for acc big number for loss to save a model
    def accat(out,trg,thresh=0.05 ):

        out = out.detach().cpu().numpy().squeeze()
        trg = trg.detach().cpu().numpy().squeeze()
        diff = np.abs(out - trg)

        diff[diff > thresh] = 1
        diff[diff <= thresh] = 0
        diff = (diff * -1) + 1
        correct = np.sum(diff)

        return correct
        # exit()
    for epoch in range(epochs):
        model.train()
        
        train_losses = []
        valid_losses = []
        for i, (images, labels) in enumerate(train_loader):
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
            # if (i * batch_size) % (batch_size * 100) == 0:
            #     print(f'{i * batch_size} / 50000')
                
        model.eval()
        correct_5_2 = 0
        correct_5_1 = 0
        
        total_loss = 0
        total = 0
        accsat =[0.1,0.05,0.01]
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
            # torch.save(model.state_dict(),os.path.join(os.getcwd(),'models',exp_name+'.pth'))
        
        writer.add_scalar('Loss/val', np.mean(valid_losses), epoch)
        # valid_acc_list.append(accuracy)
        if epoch ==epochs-1:
            print('epoch : {}, train loss : {:.4f}, valid loss : {:.4f}'\
                .format(epoch+1, np.mean(train_losses), np.mean(valid_losses)))
    return best
def main():
    
    res = skopt.forest_minimize(train, SPACE, n_calls=100)
    print(res)
    # import pickle
    dump(res, 'result.gz')
if __name__ == "__main__":
    # experiments = ['Raw','var','L1','tree']
    # exp_params = [None, [(.01 * (1 - .01))],[0.01],[50]]
    # hid_layers = [1000,1000]
    # batch_size = 128
    # lr = 0.03
    main()
    # for name,params in zip(experiments,exp_params):
    
    
    # train(batch_size=batch_size,exp_name=name,lr=lr,hid_layers=hid_layers,dim_redu_params=params,create_dim_redu=True)