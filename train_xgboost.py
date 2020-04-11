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
import xgboost as xgb
from sklearn.metrics import mean_squared_error
# import pandas as pd
import numpy as np
def train():
    exp_name = 'runs/Raw_normal_realseason_pca'
    batch_size =128
    use_pca = True
    if os.path.exists(exp_name):
        shutil.rmtree(exp_name)

    time.sleep(1)
    # writer = SummaryWriter(exp_name,flush_secs=1)


   




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

    y = np.load('bined_y.npy')
    print(np.max(y))
    print(np.min(y))
    print(X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(X_test.shape)
    # if use_pca and 'Raw' in exp_name:
    # scaler = PCA(.995)
    # # scaler = StandardScaler()
    # scaler.fit(X_train)
    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)
    # print(np.max(y_train))
    # print(np.max(1+y_train*9))
    
    # exit()
    # data_dmatrix = xgb.DMatrix(data=X_train,label=X_test)
    
    xg_reg = xgb.XGBClassifier(max_depth =3, learning_rate = 0.01)
    
    # xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                # max_depth = 5, alpha = 10, n_estimators = 5,verbosity=1,gamma=  1)
    # print(X_train[:10])
    xg_reg.fit(X_train,y_train)

    preds = xg_reg.predict(X_test)
    print(preds[-10:],y_test[-10:])
    diff = preds- y_test 
    accuracy = (len(preds)-np.count_nonzero(diff) )/len(preds)
    # accuracy = (np.abs(preds - y_test) < 0.0001 ).all().mean()
    print(accuracy)
    # rmse = np.sqrt(mean_squared_error(y_test, preds))
    # print("RMSE: %f" % (rmse))
    # plt.figure(1)
    # xgb.plot_tree(xg_reg,num_trees=0, rankdir='LR')
    # xgb.plot_tree(xg_reg,num_trees=1, rankdir='LR')
    # xgb.plot_tree(xg_reg,num_trees=2, rankdir='LR')
    # xgb.plot_tree(xg_reg,num_trees=3, rankdir='LR')
    # xgb.plot_tree(xg_reg,num_trees=4, rankdir='LR')
    # xgb.plot_tree(xg_reg,num_trees=5, rankdir='LR')
    # xgb.plot_tree(xg_reg,num_trees=6, rankdir='LR')
    # xgb.plot_tree(xg_reg,num_trees=7, rankdir='LR')
    # xgb.plot_tree(xg_reg,num_trees=8, rankdir='LR')
    # xgb.plot_tree(xg_reg,num_trees=9, rankdir='LR')
    # xgb.plot_tree(xg_reg,num_trees=10, rankdir='LR')
    # xgb.plot_tree(xg_reg,num_trees=11, rankdir='LR')
    # xgb.plot_tree(xg_reg,num_trees=12, rankdir='LR')
    # xgb.plot_tree(xg_reg,num_trees=13, rankdir='LR')
    # plt.rcParams['figure.figsize'] = [50, 10]
    # plt.figure(2)
    xgb.plot_importance(xg_reg)
    plt.rcParams['figure.figsize'] = [5, 5]
    plt.show()
train()