import numpy as np
import os,sys
from collections import defaultdict
from tqdm import tqdm
# from datetime import datetime
from datetime import date as date_lib
import ast
import matplotlib.pyplot as plt
import pickle
# #--------------------------------------
# #                PICKLE LOAD
# with open('RAW_Data.pickle') as f:
#     loaded_obj = pickle.load(f)

# print 'loaded_obj is', loaded_obj
# #--------------------------------------


# >>> x = u'[ "A","B","C" , " D"]'
var_ids =[ 'mood'  , 'circumplex.arousal','circumplex.valence' ,'activity','screen' ,'call','sms' ,'appCat.builtin' ,'appCat.communication', 'appCat.entertainment','appCat.finance' ,'appCat.game','appCat.office','appCat.other','appCat.social' ,'appCat.travel' ,'appCat.unknown' ,'appCat.utilities', 'appCat.weather','morning','noon','afternoon','night','winter','spring','summer','fall' ]
# idx = var_ids.index(var)
# data = [None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None]
    
filename = 'text.txt'
with open (filename,'r') as f: 
    data = f.readlines()

summ = np.zeros(27)
for line in range(len(data)):
    date = data[line].split('_')[0]
    rest_of_line = data[line].split('_')[1]
    # print(data)
    id = rest_of_line.split('[[')[0]
    x = ast.literal_eval('[['+rest_of_line.split('[[')[1])  
    mood = x[0]
    x= x[1:]
    print(date,id,mood, x)
    x = np.array(x)
    x[x==None] = 0
    print(np.max(x,axis=0))
    exit()
    x = np.array(x)
    x[x==None] = 0
    
    x[x!=0] = 1
    summ =summ+ np.sum(x,axis=1)
    # print(x)
# plt.scatter(np.arange(27),summ)
# plt.show()
    # exit()
# np.zeros(user,days,26 vars)