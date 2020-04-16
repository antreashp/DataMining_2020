import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pickle
import pandas as pd
import seaborn as sns
class Plotter():
    def __init__(self):

        filename = 'RAW_Data.pickle'
        with open(filename, 'rb') as f:
            self.data = pickle.load(f)
            # print(self.data.keys())
            self.users = list(self.data.keys())
            self.save_dir = 'plots/'
            self.var_names =  ['user','day','mood', 'circumplex.arousal', 'circumplex.valence', 'activity', 'screen', 'call', 'sms', 'appCat.builtin',
           'appCat.communication', 'appCat.entertainment', 'appCat.finance', 'appCat.game', 'appCat.office',
           'appCat.other', 'appCat.social', 'appCat.travel', 'appCat.unknown', 'appCat.utilities', 'appCat.weather',
           'morning', 'noon', 'afternoon', 'night', 'winter', 'spring', 'spring2', 'spring3','summer']
            # print(len(self.var_names))
            # print(list(self.data['AS14.01'].keys())[:10])
            # print(len(self.data['AS14.01'][735290]))
            # print(len(self.data['AS14.01'][735290][0]))
            my_arr = []
            for i,user in enumerate(self.users):
                # my_arr.append([])
                for j,day in enumerate(list(self.data[user].keys())):
                    # my_arr[i].append([])    
                    for k,record in enumerate(list(self.data[user][day])):
                        my_arr.append([user]+[j]+self.data[user][day][k])
            numpy_data = np.array(my_arr)

            self.df = pd.DataFrame(data=numpy_data, columns=self.var_names)
            group_by_day = self.df.groupby(by=['day','user'])
            mood_count = group_by_day.max()
            # print(mood_count)
            mood_count = mood_count.dropna(subset=['mood'])
            print(mood_count)
            d = mood_count.index
            print(d)
            exit()
            print(self.df.head(1))
            # filtered_class = self.df
            
            sns.pairplot(self.df.loc[:])
            # sns.lmplot('user',"day", data= self.df, hue='mood', fit_reg=False, col="circumplex.arousal", col_wrap=3)
            plt.show()
    def scatter_variable_for_user(self,var=None,show=True,save=False):
        if var is None:
            print('you have to select a variable to plot...')
            return None
        
        


if __name__ == "__main__":
    plotter = Plotter()


# meh = Counter(y[:]) 
# for i,val in meh.items():
#     print(i,val)
# exit()
# print(x.shape)
# plt.figure(1)
# plt.ylabel('valence_counts')
# plt.xlabel('value*100')
# meh = Counter(x[:,0])
# counts = np.zeros((100))
# for i in list(meh.keys()):
#     print(i)
#     counts[int(float(i)*100)-1] = meh[i]
# plt.bar(range(100),counts)

# plt.figure(2)
# plt.ylabel('valence')
# plt.ylabel('records')
# plt.scatter(range(x.shape[0]),x[:,0])
# plt.show()

