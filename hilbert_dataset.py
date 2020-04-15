import numpy as np
import os, sys
from collections import defaultdict
from tqdm import tqdm
# from datetime import datetime
from datetime import date as date_lib
import ast
import matplotlib.pyplot as plt
import pickle
import shutil
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from preprosses import preprocess,dict_to_numpy
import pandas as pd 

def rot ( n, x, y, rx, ry ):

#*****************************************************************************80
#
## ROT rotates and flips a quadrant appropriately.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license. 
#
#  Modified:
#
#    03 January 2016
#
#  Parameters:
#
#    Input, integer N, the length of a side of the square.  
#    N must be a power of 2.
#
#    Input/output, integer X, Y, the coordinates of a point.
#
#    Input, integer RX, RY, ???
#
    if ( ry ==   0 ):
#
#  Reflect.
#
        if ( rx == 1 ):
            x = n - 1 - x
            y = n - 1 - y
    #
    #  Flip.
    #
        t = x
        x = y
        y = t

    return x, y

def d2xy ( m, d ):

#*****************************************************************************80
#
## D2XY converts a 1D Hilbert coordinate to a 2D Cartesian coordinate.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license. 
#
#  Modified:
#
#    03 January 2016
#
#  Parameters:
#
#    Input, integer M, the index of the Hilbert curve.
#    The number of cells is N=2^M.
#    0 < M.
#
#    Input, integer D, the Hilbert coordinate of the cell.
#    0 <= D < N * N.
#
#    Output, integer X, Y, the Cartesian coordinates of the cell.
#    0 <= X, Y < N.
#
    n = 2 ** m

    x = 0
    y = 0
    t = d
    s = 1

    while ( s < n ):

        rx = ( ( t // 2 ) % 2 )
        if ( rx == 0 ):
            ry = ( t % 2 )
        else:
            ry = ( ( t ^ rx ) % 2 )
        x, y = rot ( s, x, y, rx, ry )
        x = x + s * rx
        y = y + s * ry
        t = ( t // 4 )

        s = s * 2

    return x, y

filename = 'RAW_Data.pickle'
win_size = 1
preprocess_instance = preprocess(filename, window_size=win_size)
preprocess_instance.normalize()
def dict_to_user_numpy(my_dict):
    x = []
    y = []
    for i,id in enumerate(my_dict):
        x.append([])
        y.append([])
        for date in  my_dict[id]:
            x[i].append(my_dict[id][date][1:])
            y[i].append(my_dict[id][date][0])
    x = np.array(x)
    y = np.array(y)
    return x,y
# preprocess_instance.bin()
none_days = 0
total_days = 0
for user, user_data in preprocess_instance.data.items():
    for day, day_data in preprocess_instance.data[user].items():
        total_days += 1

        if all(data[0] is None for data in day_data):
            # print(day)
            none_days += 1
        # for data in day_data:
        #     if data[0] is None:
        #         print(day, day_data[0])
print(none_days, total_days)
preprocess_instance.bin(include_remainder=False)
for user, user_data in preprocess_instance.processed_data.items():
    for day, day_data in preprocess_instance.processed_data[user].items():
        try:
            preprocess_instance.processed_data[user][day][0] = preprocess_instance.processed_data[user][day][0]/10
        except:
            print('fart')
            preprocess_instance.processed_data[user][day][0] =0            
x,y = dict_to_user_numpy(preprocess_instance.processed_data)
print(x.shape)
print(y.shape)
for i,(user, user_data )in enumerate(tqdm(preprocess_instance.processed_data.items())):

    print(len(x[i]))
    for k in range(len(x[i])-9):
        image = np.array(x[i][k:k+9])
        image = image.reshape(-1,1)
        for meh in range(13):
            image = np.insert(image, 0, 0, axis=0)
        # print(image.shape)
        # hilxy = d2xy ( 5, d )
        final = np.zeros((16,16))
        for meh in range (len(image)):
            x_cor,y_cor = d2xy (4, meh )
            final[x_cor][y_cor] = image[meh]
        # np.save('hilbert_data/user'+str(i)+'_day'+str(k)+'_mood'+str(y[i][k+9]), final)
        # plt.figure(1)
        # plt.imshow(final)
        fig = plt.figure(figsize=[6,6])
        ax = fig.add_subplot(111)
        ax.imshow(final)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)
        fname ='hilbert_data/user'+str(i)+'_day'+str(k)+'_mood'+str(round(y[i][k+9],2))+'.jpg'
        plt.savefig(fname, bbox_inches='tight',pad_inches=0)
        # plt.savefig(, bbox_inches = 'tight',
    # pad_inches = 0)
        # plt.show()
        # exit()
    # for j,day in enumerate(x[i]):

    # hilxy = d2xy ( 5, d )
# print(x[0])
