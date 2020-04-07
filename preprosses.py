import numpy as np
import os, sys
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
var_ids = ['mood', 'circumplex.arousal', 'circumplex.valence', 'activity', 'screen', 'call', 'sms', 'appCat.builtin',
           'appCat.communication', 'appCat.entertainment', 'appCat.finance', 'appCat.game', 'appCat.office',
           'appCat.other', 'appCat.social', 'appCat.travel', 'appCat.unknown', 'appCat.utilities', 'appCat.weather',
           'morning', 'noon', 'afternoon', 'night', 'winter', 'spring', 'summer', 'fall']


# idx = var_ids.index(var)
# data = [None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None]

class preprocess:
    def __init__(self, filename, window_size=1, step_size=1):
        """

        :param filename: str
        :param window_size: int
        :param step_size: int
        """
        with open(filename, 'rb') as f:
            self.data = pickle.load(f)
        self.window = window_size if window_size >= 1 and type(window_size) is int else 1
        self.step = step_size if step_size is step_size <= window_size and step_size >= 1 and type(
            step_size) is int else 1

    def average(self, record):
        return [sum([x[i] for x in record]) / len(record) for i in range(len(record[0]))]

    def normalize(self):
        for user in self.data.keys():
            user_data = self.data[user]
            date_keys = [x for x in user_data.keys()]
            max_values = [0 for x in user_data[date_keys[0]][0]]
            for date in date_keys:
                date_data = user_data[date]
                for record in date_data:
                    for i in range(3, 18):
                        if record[i] is None:
                            record[i] = 0
                        if max_values[i] < record[i]:
                            max_values[i] = record[i]
            for date in date_keys:
                date_data = user_data[date]
                for i in range(len(date_data)):
                    record = date_data[i]
                    for j in range(3, 18):
                        record[j] = record[j] / max_values[j] if max_values[j] != 0 else 0
                    self.data[user][date][i] = record
            print(self.data[user])

    def bin(self, include_remainder=False):
        for user in self.data.keys():
            user_data = self.data[user]
            current_index = 0
            processed_user_data = {}
            date_keys = [x for x in user_data.keys()]
            while (current_index + + self.window < len(user_data) and not include_remainder) \
                    or (current_index + self.step < len(user_data) and include_remainder):
                record = []
                if current_index + self.window >= len(user_data) and include_remainder:
                    for i in range(len(user_data) - current_index):
                        record.append(user_data[date_keys[current_index + i]])
                else:
                    for i in range(self.window):
                        record.append(user_data[date_keys[current_index + i]])
                current_index += self.step

filename = 'RAW_Data.pickle'
print(type(1))

if __name__ == '__main__':
    preprocess_instance = preprocess(filename)
    preprocess_instance.normalize()
    # preprocess_instance.bin()

# summ = np.zeros(27)
# for line in range(len(data)):
#     date = data[line].split('_')[0]
#     rest_of_line = data[line].split('_')[1]
#     # print(data)
#     id = rest_of_line.split('[[')[0]
#     x = ast.literal_eval('[['+rest_of_line.split('[[')[1])
#     mood = x[0]
#     x= x[1:]
#     print(date,id,mood, x)
#     x = np.array(x)
#     x[x==None] = 0
#     print(np.max(x,axis=0))
#     exit()
#     x = np.array(x)
#     x[x==None] = 0
#
#     x[x!=0] = 1
#     summ =summ+ np.sum(x,axis=1)
