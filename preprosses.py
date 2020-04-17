import numpy as np
import pandas as pd
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

var_ids = ['mood', 'circumplex.arousal', 'circumplex.valence', 'activity', 'screen', 'call', 'sms', 'appCat.builtin',
           'appCat.communication', 'appCat.entertainment', 'appCat.finance', 'appCat.game', 'appCat.office',
           'appCat.other', 'appCat.social', 'appCat.travel', 'appCat.unknown', 'appCat.utilities', 'appCat.weather',
           'morning', 'noon', 'afternoon', 'night', 'winter', 'spring', 'spring2', 'spring3','summer','mood']

class preprocess:
    def __init__(self, filename, window_size=1, methods=None,writer = None):
        """
        :param filename: str
        :param window_size: int
        :param step_size: int
        """

        self.step = 1 # step size always 1
        self.methods = methods
        with open(filename, 'rb') as f:
            self.data = pickle.load(f)
        self.window = window_size if window_size >= 1 and type(window_size) is int else 1
                
        self.writer = writer

        # self.step = step_size if step_size is step_size <= window_size and step_size >= 1 and type(
        #     step_size) is int else 1
        self.processed_data = {}
        self.encoded = False
        self.indexes = {
            'target': [0, 0],
            'valence': [1, 1],
            'arousal': [2, 2],
            'appcat': [3, 18],
            'time': [19, 22],
            'season': [23, 27],
            'mood': [28, 28]
        }
        self.variable_names = ['mood', 'circumplex.arousal', 'circumplex.valence', 'activity', 'screen', 'call', 'sms', 'appCat.builtin',
           'appCat.communication', 'appCat.entertainment', 'appCat.finance', 'appCat.game', 'appCat.office',
           'appCat.other', 'appCat.social', 'appCat.travel', 'appCat.unknown', 'appCat.utilities', 'appCat.weather',
           'morning', 'noon', 'afternoon', 'night', 'winter', 'spring', 'spring2', 'spring3','summer','average_mood']
        # inclusive
        
    def set_index(self, indexes):
        """
        Input a dictionary, with variable name as keys, indexes (list of 2)

        :param indexes: dict
        :return:
        """
        for key, value in indexes.items():
            self.indexes[key] = value

    def average(self, record):
        """
        :param record: list
        """
        methods = ['average'] * len(record[0]) if self.methods is None else self.methods

        return [(sum([x[i] for x in record if x[i] is not None]) / len([x for x in record if x[i] is not None])) if
                methods[i] == 'average' else max([x[i] for x in record if x[i] is not None]) if methods[i] == 'max' else
                min([x[i] for x in record if x[i] is not None]) for i in range(self.indexes['appcat'][0],
                                                                               self.indexes['appcat'][1] + 1)]

    def average_mood(self, record):
        """
        :param record: list
        """
        method = self.methods[0] is self.methods is not None
        moods = []
        for data_point in record:
            if data_point[self.indexes['target'][0]] is not None:
                moods.append(data_point[self.indexes['target'][0]])
        return -1 if not len(moods) else sum(moods) / len(moods) if method == 'average' else max(moods) if method == \
                                                                                             'max' else min(moods)

    def average_circumplex(self, record):
        """
        :param record: list
        """
        method = self.methods[0] is self.methods is not None
        arousal = []
        valence = []
        for data_point in record:
            if data_point[self.indexes['arousal'][0]] is not None:
                arousal.append(data_point[self.indexes['arousal'][0]])
        for data_point in record:
            if data_point[self.indexes['valence'][0]] is not None:
                valence.append(data_point[self.indexes['valence'][0]])
        return 0.5 if not len(arousal) else sum(arousal) / len(arousal) if method == 'average' else max(arousal) if \
            method == 'max' else min(arousal), \
               0.5 if not len(valence) else sum(valence) / len(valence) if method == 'average' else max(valence) if \
                   method == 'max' else min(valence)

    def average_time_and_season(self, record):
        """
        :param record: list
        """
        return [sum([x[i] for x in record]) / len(record) for i in range(self.indexes['time'][0], self.indexes['season'][
            1] + 1)]

    def normalize(self):
        for user in self.data.keys():
            user_data = self.data[user]
            date_keys = [x for x in user_data.keys()]
            max_values = [0 for x in user_data[date_keys[0]][0]]
            for date in date_keys:
                date_data = user_data[date]
                for record in date_data:
                    for i in range(self.indexes['appcat'][0], self.indexes['appcat'][1] + 1):
                        if record[i] is None:
                            record[i] = 0
                        if max_values[i] < record[i]:
                            max_values[i] = record[i]
            for date in date_keys:
                date_data = user_data[date]
                for i in range(len(date_data)):
                    record = date_data[i]
                    for j in range(self.indexes['appcat'][0], self.indexes['appcat'][1] + 1):
                        record[j] = record[j] / max_values[j] if max_values[j] != 0 else 0
                    self.data[user][date][i] = record

    def bin(self, include_remainder=False):
        for user in self.data.keys():
            user_data = self.data[user]
            current_index = 0
            processed_user_data = {}
            date_keys = [x for x in user_data.keys()]
            while (current_index + self.window < len(user_data.items()) and not include_remainder) \
                    or (current_index + self.step < len(user_data) and include_remainder):
                record = []
                if current_index + self.window >= len(user_data) and include_remainder:
                    for i in range(len(user_data) - current_index):
                        for j in range(len(user_data[date_keys[current_index + i]])):
                            record.append(user_data[date_keys[current_index + i]][j])
                else:
                    for i in range(self.window):
                        for j in range(len(user_data[date_keys[current_index + i]])):
                            record.append(user_data[date_keys[current_index + i]][j])
                target_mood = self.average_mood(record)
                previous_mood = self.average_mood([record[0]])
                data_point = [None] + list(self.average_circumplex(record)) + self.average(
                    record) + self.average_time_and_season(record) + [target_mood / 9]
                if target_mood == -1 or previous_mood == -1:
                    # print(user, date_keys[current_index], 'removed')
                    current_index += self.step
                    del user_data[date_keys[current_index]]
                    date_keys.pop(current_index)
                    continue
                processed_user_data[date_keys[current_index]] = data_point
                if current_index > 0:
                    processed_user_data[date_keys[current_index - 1]][0] = previous_mood
                    # print(processed_user_data[date_keys[current_index - 1]])
                # print(target_mood)
                current_index += self.step
            self.processed_data[user] = processed_user_data

    def create_dataframe(self):
        data_matrix = []
        days_without_mood = set()
        for user, user_data in self.data.items():
            for day, day_data in user_data.items():
                has_mood = False
                for record in day_data:
                    if record[self.indexes['target'][0]] is not None:
                        has_mood = True
                    row = [user, day]
                    for i, value in enumerate(record):
                        if i >= self.indexes['appcat'][0] and i < self.indexes['appcat'][1]:
                            row.append(0 if value is None else value)
                        else:
                            row.append(value)
                    data_matrix.append(row)
                if not has_mood:
                    days_without_mood.add(day)
        data_matrix_clear = []
        for row in data_matrix:
            if row[1] not in days_without_mood:
                data_matrix_clear.append(row)
        dataframe = pd.DataFrame(data_matrix_clear, columns=['user_id', 'date'] + self.variable_names[:-1])
        return dataframe

    def create_dataframe_pros(self):
        data_matrix = []
        days_without_mood = set()
        for user, user_data in self.processed_data.items():
            for day, record in user_data.items():
                has_mood = False
                if record[self.indexes['target'][0]] is not None:
                    has_mood = True
                row = [user, day]
                for i, value in enumerate(record):
                    if i >= self.indexes['appcat'][0] and i < self.indexes['appcat'][1]:
                        row.append(0 if value is None else value)
                    else:
                        row.append(value)
                data_matrix.append(row)
                if not has_mood:
                    days_without_mood.add(day)
        data_matrix_clear = []
        for row in data_matrix:
            if row[1] not in days_without_mood:
                data_matrix_clear.append(row)
        dataframe = pd.DataFrame(data_matrix_clear, columns=['user_id', 'date'] + self.variable_names)
        return dataframe
    def bin_nonorm(self, include_remainder=False):
        for user in self.data.keys():
            user_data = self.data[user]
            current_index = 0
            processed_user_data = {}
            date_keys = [x for x in user_data.keys()]
            while (current_index + self.window < len(user_data.items()) and not include_remainder) \
                    or (current_index + self.step < len(user_data) and include_remainder):
                record = []
                if current_index + self.window >= len(user_data) and include_remainder:
                    for i in range(len(user_data) - current_index):
                        for j in range(len(user_data[date_keys[current_index + i]])):
                            record.append(user_data[date_keys[current_index + i]][j])
                else:
                    for i in range(self.window):
                        for j in range(len(user_data[date_keys[current_index + i]])):
                            record.append(user_data[date_keys[current_index + i]][j])
                target_mood = self.average_mood(record)
                previous_mood = self.average_mood([record[0]])
                data_point = [None] + list(record) + record + record + [target_mood / 9]
                if target_mood == -1 or previous_mood == -1:
                    # print(user, date_keys[current_index], 'removed')
                    current_index += self.step
                    del user_data[date_keys[current_index]]
                    date_keys.pop(current_index)
                    continue
                processed_user_data[date_keys[current_index]] = data_point
                if current_index > 0:
                    processed_user_data[date_keys[current_index - 1]][0] = previous_mood
                    # print(processed_user_data[date_keys[current_index - 1]])
                # print(target_mood)
                current_index += self.step
            self.processed_data[user] = processed_user_data

    def bench_mark(self):
        count_accurate = 0
        count_total = 0
        for i,user_data in enumerate(self.processed_data.values()):
            for day_data in user_data.values():
                if self.encoded:
                    mood = self.decode_target(day_data[0])
                else:
                    mood = day_data[0]
                # print('meh',day_data[0])
                # print('blah',day_data[-1]*9)
                # print(mood)
                if mood is not None and abs(mood-day_data[-1]*9) <0.5:
                # exit()
                # if mood is not None and day_data[0] <= day_data[-1] * 9 + 0.5 and mood > day_data[-1] * 9\
                        # - 0.5:
                    count_accurate += 1
                count_total += 1
            temp_acc = count_accurate / count_total

            self.writer.add_scalar('Acc/val_@'+str(0.05),float(temp_acc*100), i)

        accuracy = count_accurate / count_total
        return accuracy

    def transform_target(self):
        self.encoded = True
        counts = [0 for i in range(9)]
        for user, user_data in self.processed_data.items():
            for data_point in user_data.values():
                mood = data_point[0]
                if mood is None:
                    continue
                if mood == 9:
                    counts[8] += 1
                else:
                    counts[round(mood)] += 1
        total_count = sum(counts)
        K = [9 * counts[i] / total_count for i in range(9)]
        offsets = [sum([K[j] for j in range(i)]) for i in range(9)]
        self.offsets = offsets.copy()
        self.K = K.copy()
        for user, user_data in self.processed_data.items():
            for day, data_point in user_data.items():
                mood = data_point[0]
                if mood is None:
                    continue
                transformed_mood = offsets[round(mood) if mood < 9 else 8] + K[round(mood) if mood < 9 else 8] * (
                    mood - round(mood))
                self.processed_data[user][day][0] = transformed_mood

    def decode_target(self, value):
        if value == 9:
            return 9
        if value == 0:
            return 0
        if value is None:
            return None
        k = 0
        for offset in self.offsets:
            if value < offset:
                break
            k += 1
        k -= 1
        original_value = k + (value - self.offsets[k]) / self.K[k]
        return original_value
    
    def decode_targets(self, arr):
        d_arr = np.zeros(arr.shape)
        for i,value in enumerate(arr):

            if value == 9:
                return 9
            if value == 0:
                return 0
            if value is None:
                return None
            k = 0
            for offset in self.offsets:
                if value < offset:
                    break
                k += 1
            k -= 1
            d_arr[i] = k + (value - self.offsets[k]) / self.K[k]
        return d_arr

filename = 'RAW_Data.pickle'
def dict_to_numpy(my_dict):
    x = []
    y = []
    for id in my_dict:
        for date in  my_dict[id]:
            x.append(my_dict[id][date][1:])
            y.append(my_dict[id][date][0])
    x = np.array(x)
    y = np.array(y)
    return x,y


def save_numpy(arr,filename):
    np.save(filename, arr)


if __name__ == '__main__':
    
    for win_size in range(1,2):
        '''change methods here'''
        methods = ['average','max','max','max','max','max','max','max','max','max',
        'max','max','max','max','max','max','max','max','average','average',
        'average','average','average','average','average','average','average','average']
        writer = SummaryWriter(os.path.join('runs','benchmark_win'+str(win_size)),flush_secs=1)
        preprocess_instance = preprocess(filename, window_size=win_size, methods=methods,writer=writer)
        preprocess_instance.normalize()
        none_days = 0
        total_days = 0
        for user, user_data in preprocess_instance.data.items():
            for day, day_data in preprocess_instance.data[user].items():
                total_days += 1
                if all(data[0] is None for data in day_data):
                    # print(day)
                    none_days += 1
                    
        print(none_days, total_days)
        preprocess_instance.bin(include_remainder=False)
        preprocess_instance.transform_target()
        exp_name = 'runs/benchmark_win'+str(win_size)
        if os.path.exists(exp_name):
            shutil.rmtree(exp_name)

        xaxis = np.ones((50)) *preprocess_instance.bench_mark()

        print('benchmark accuracy: ',preprocess_instance.bench_mark())
        '''Save preprocess_instance.processed_data:'''
        # print(preprocess_instance.processed_data['AS14.01'][735327])
        # print(len(preprocess_instance.processed_data.keys()))
        # print(len(preprocess_instance.processed_data['AS14.02'].keys()))
        # preprocess_instance.processed_data
        
        clean_records =[]
        for user, user_data in preprocess_instance.processed_data.items():
            for day, day_data in preprocess_instance.processed_data[user].items():
                # total_days += 1
                # print(preprocess_instance.processed_data [user][day])

                if  day_data[0] is None :
                    pass
                else:
                    # print(preprocess_instance.processed_data [user][day])
                    # exit()
                    clean_records.append(preprocess_instance.processed_data [user][day])
        
        clean_records = np.array(clean_records)
        print(clean_records.shape)
        x = clean_records[:,1:]
        y = clean_records[:,0]
        # x,y = dict_to_numpy(preprocess_instance.processed_data)
        print(x.shape)
        print(y.shape)
        save_numpy(x,'bined_x_win'+str(win_size))
        save_numpy(y,'bined_y_win'+str(win_size))
        '''
        {
            user_id: {
                date: {
                    [variables]
                }, ...
            }, ...
        }
        '''