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


            self.bins =  {'user':27,'day':80,'mood':10, 'circumplex.arousal':5, 'circumplex.valence':5, 'activity':100, 'screen':100, 'call':100, 'sms':100, 'appCat.builtin':100,
           'appCat.communication':100, 'appCat.entertainment':100, 'appCat.finance':100, 'appCat.game':100, 'appCat.office':100,
           'appCat.other':100, 'appCat.social':100, 'appCat.travel':100, 'appCat.unknown':100, 'appCat.utilities':100, 'appCat.weather':100,
           'morning':10, 'noon':10, 'afternoon':10, 'night':10, 'winter':100, 'spring':100, 'spring2':100, 'spring3':100,'summer':100}
            # print(len(self.var_names))
            # print(list(self.data['AS14.01'].keys())[:10])
            # print(len(self.data['AS14.01'][735290]))
            # print(len(self.data['AS14.01'][735290][0]))
            my_arr = []
            for i,user in enumerate(self.users):
                # my_arr.append([])
                for j,day in enumerate(list(self.data[user].keys())):
                    # my_arr[i].append([])    
                    if all(data[0] is None for data in self.data[user][day]):
                        pass
                    else:

                        for k,record in enumerate(list(self.data[user][day])):
                            # if self.data[user][day][k][0] is None:
                            #     continue
                            # print(record)
                            my_arr.append([i]+[j]+self.data[user][day][k])
            numpy_data = np.array(my_arr)

            self.df = pd.DataFrame(data=numpy_data, columns=self.var_names)
            
            # print(self.df)
            # exit()
            # print(self.df['screen'].max())
            # self.df = self.df.groupby(['day','user'])
            # for j in self.df['mood' ].max():

            #     print(j)
            # exit()
            self.df.fillna(value=np.nan, inplace=True)
            # self.df = self.df.fillna(0)

            # print(self.df.head(10))
            # group_by_day = self.df.groupby(by=['day','user'])
            # self.df = group_by_day.mean()         # filtered_class = self.df
            # corr = self.df
            # print(corr)
            # sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap=sns.diverging_palette(220, 10, as_cmap=True))
            # sns.pairplot(self.df.head(30).loc[:])
            # sns.lmplot('user',"day", data= self.df, hue='mood', fit_reg=False, col="circumplex.arousal", col_wrap=3)
            # plt.show()
    def scatter_variable_for_user(self,var=None,show=True,save=False):
        if var is None:
            print('you have to select a variable to plot...')
            return None
        
    def histogram_ofvar(self,var=None,show=True,save=False):
        if var is None:
            print('you have to select a variable to plot...')
            return None
        sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})
        meh = self.df.dropna(subset=[var])
        print(meh)
        temp_plot = sns.distplot(meh[var], norm_hist=False, kde=False if  var != 'user' and var != 'day' else False ,
             hist_kws={"alpha": 1}).set(xlabel=var, ylabel='Count',title=var+' Histogram')
        if save:
            plt.savefig('plots/'+var+"_histogram.png")
        if show:
            plt.show()
        return   temp_plot  


if __name__ == "__main__":
      
    plotter = Plotter()
    for var in plotter.var_names:
        ans = plotter.histogram_ofvar(var=var,show=True,save=True)

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

