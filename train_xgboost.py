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
    feature_names = [ 'arousal', 'valence', 'activity', 'screen', 'call', 'sms', 'builtin',
           'communication', 'entertainment', 'finance', 'game', 'office',
           'other', 'social', 'travel', 'unknown', 'utilities', 'weather',
           'morning', 'noon', 'afternoon', 'night', 'winter', 'spring', 'spring2', 'spring3','mood']
    print(len(feature_names))
    exp_name = 'runs/Raw_normal_realseason_pca'
    batch_size =128
    
    use_pca = True
    if os.path.exists(exp_name):
        shutil.rmtree(exp_name)

    time.sleep(1)
    # writer = SummaryWriter(exp_name,flush_secs=1)


   




    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
    
    X = np.load('bined_x_win2.npy')

    y = np.load('bined_y_win2.npy')
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
    
    # xg_reg = xgb.XGBClassifier(max_depth =3, learning_rate = 0.01)
    
    xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.03, learning_rate = 0.03,
                max_depth =10, alpha = 200, n_estimators = 500,verbosity=1,feature_names=feature_names,booster ='dart',gamma = 0.01,max_delta_step =2000)
    # print(X_train[:10])
    xg_reg.fit(X_train,y_train)

    preds = xg_reg.predict(X_test)
    print(preds.shape,y_test.shape)
    print(preds[-10:],y_test[-10:])
    diff = abs(preds- y_test )
    # print(diff)
    accuracy = (len(diff[diff<0.5]) )/preds.shape[0]
    # accuracy01 = (len(diff[diff>0.5]) )/len(preds)
    # accuracy = (len(diff[diff>0.5]) )/len(preds)
    # accuracy = (np.abs(preds - y_test) < 0.0001 ).all().mean()
    print('accuracy,',accuracy)
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
    # plt.figure(1)
    xgb.plot_importance(xg_reg,show_values=True)
    # xgb.plot_importance(xg_reg.get_booster())
    
    plt.rcParams['figure.figsize'] = [5, 5]
    
    plt.show()
train()