import numpy as np
import os, sys
from collections import defaultdict
from tqdm import tqdm
# from datetime import datetime
from datetime import date as date_lib
import ast
import matplotlib.pyplot as plt
import pickle


var_ids = ['mood', 'circumplex.arousal', 'circumplex.valence', 'activity', 'screen', 'call', 'sms', 'appCat.builtin',
           'appCat.communication', 'appCat.entertainment', 'appCat.finance', 'appCat.game', 'appCat.office',
           'appCat.other', 'appCat.social', 'appCat.travel', 'appCat.unknown', 'appCat.utilities', 'appCat.weather',
           'morning', 'noon', 'afternoon', 'night', 'winter', 'spring', 'summer', 'fall']


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
        self.processed_data = {}

    def average(self, record):
        """
        :param record: list
        """
        return [sum([x[i] for x in record if x[i] is not None]) / len([x for x in record if x[i] is not None])
                for i in range(3, 19)]

    def average_mood(self, record):
        """
        :param record: list
        """
        moods = []
        for data_point in record:
            if data_point[0] is not None:
                moods.append(data_point[0])
        return sum(moods) / len(moods) if len(moods) else -1

    def average_circumplex(self, record):
        """
        :param record: list
        """
        arousal = []
        valence = []
        for data_point in record:
            if data_point[1] is not None:
                arousal.append(data_point[1])
        for data_point in record:
            if data_point[2] is not None:
                valence.append(data_point[2])
        return sum(arousal) / len(arousal) if len(arousal) else 0.5, sum(valence) / len(valence) if len(valence) else\
            0.5

    def average_time_and_season(self, record):
        """
        :param record: list
        """
        return [sum([x[i] for x in record]) / len(record) for i in range(19, 27)]

    def normalize(self):
        for user in self.data.keys():
            user_data = self.data[user]
            date_keys = [x for x in user_data.keys()]
            max_values = [0 for x in user_data[date_keys[0]][0]]
            for date in date_keys:
                date_data = user_data[date]
                for record in date_data:
                    for i in range(3, 19):
                        if record[i] is None:
                            record[i] = 0
                        if max_values[i] < record[i]:
                            max_values[i] = record[i]
            for date in date_keys:
                date_data = user_data[date]
                for i in range(len(date_data)):
                    record = date_data[i]
                    for j in range(3, 19):
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
                data_point = [self.average_mood(record)] + list(self.average_circumplex(record)) + self.average(
                    record) + self.average_time_and_season(record)
                if data_point[0] == -1:
                    current_index += self.step
                    continue
                processed_user_data[date_keys[current_index]] = data_point
                current_index += self.step
            self.processed_data[user] = processed_user_data


filename = 'RAW_Data.pickle'

if __name__ == '__main__':
    preprocess_instance = preprocess(filename)
    preprocess_instance.normalize()
    preprocess_instance.bin()

    '''Save preprocess_instance.processed_data:'''
    print(preprocess_instance.processed_data)


    '''
    {
        user_id: {
            date: {
                [variables]
            }, ...
        }, ...
    }
    '''
