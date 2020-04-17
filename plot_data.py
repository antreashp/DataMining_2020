import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pickle
import pandas as pd
import seaborn as sns
from preprosses import  preprocess

class Plotter():
    def __init__(self):

        filename = 'RAW_Data.pickle'
        methods = ['average','max','max','max','max','max','max','max','max','max',
        'max','max','max','max','max','max','max','max','average','average',
        'average','average','average','average','average','average','average','average']
        preprocess_instance = preprocess(filename, window_size=1, methods=methods)
        preprocess_instance.normalize()
        preprocess_instance.bin()
        self.df = preprocess_instance.create_dataframe()
        self.df_pros = preprocess_instance.create_dataframe_pros()

        self.save_dir = 'plots/'
        self.var_names =  ['user_id','date','mood', 'circumplex.arousal', 'circumplex.valence', 'activity', 'screen', 'call', 'sms', 'appCat.builtin',
        'appCat.communication', 'appCat.entertainment', 'appCat.finance', 'appCat.game', 'appCat.office',
        'appCat.other', 'appCat.social', 'appCat.travel', 'appCat.unknown', 'appCat.utilities', 'appCat.weather',
        'morning', 'noon', 'afternoon', 'night', 'winter', 'spring', 'spring2', 'spring3','summer']


        self.bins =  {'num':50,'times':10,'months':30}
        self.categorical =['user_id','date']
        self.numerical   =['mood', 'circumplex.arousal', 'circumplex.valence', 'activity', 'screen', 'call', 'sms', 'appCat.builtin',
        'appCat.communication', 'appCat.entertainment', 'appCat.finance', 'appCat.game', 'appCat.office',
        'appCat.other', 'appCat.social', 'appCat.travel', 'appCat.unknown', 'appCat.utilities', 'appCat.weather']
        self.times =['morning', 'noon', 'afternoon', 'night']
        self.months =[ 'winter', 'spring', 'spring2', 'spring3','summer']
    
        self.df.fillna(value=np.nan, inplace=True)
        self.df = self.df.fillna(0)



        
    
    def correlation_matrix(self,show=True,save=False):
        
        # sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})
        meh  = self.df_pros[self.numerical +['average_mood']][(self.df_pros[self.numerical +['average_mood']].T != 0).any()]
        corr = meh.corr(method ='spearman') 

        # print(corr)
        # mask = np.triu(np.ones_like(corr, dtype=np.bool))
        # print(mask)
        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))

        # Generate a custom diverging colormap
        # cmap = sns.diverging_palette(1, 0, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, vmax=1,
                    square=True, linewidths=.5)
        # sns.pairplot()
        if save:
            plt.savefig("plots/correlation_heatmap.png")
        if show:
            plt.show()
    def histogram_ofvars(self,var=None,show=True,save=False,pros=False):
        df = self.df_pros if pros else self.df
        if var is None:
            print('you have to select a type of variable to plot...')
            return None
        sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})
        # meh = self.df.dropna(subset=[var])
        cols = self.numerical if var == 'num' else self.categorical if var == 'cat' else self.times if var == 'times' else self.months
        if pros and var=='num':
            cols+=['average_mood']
            # print(df.head(1))
        if var != 'cat':

            df[cols].hist(bins=self.bins[var], figsize=(20, 9), layout=(5, 4))
            if save:
                
                name ='plots/'+var+"_histogram.png" if not pros else 'plots/'+var+"_pros_histogram.png"
                plt.savefig(name)
            if show:
                plt.show()
        else:
            for i,col in enumerate(cols):
                plt.figure(i)
                sns.countplot(df[col])
                if save:
                    name ='plots/'+var+str(i)+"_histogram.png" if not pros else 'plots/'+var+str(i)+"_pros_histogram.png"
                    plt.savefig(name)
                if show:
                    plt.show()

if __name__ == "__main__":
      
    plotter = Plotter()
    
    for var in ['cat','num','times','months']:
        ans = plotter.histogram_ofvars(var=var,show=False,save=True,pros=True)
        ans = plotter.histogram_ofvars(var=var,show=False,save=True,pros=False)
    # print(plotter.df_pros.head(2))
    
    # print(plotter.df.head(2))
    plotter.correlation_matrix(show=False,save=True)
    # g = sns.pairplot(plotter.df_pros,vars=['mood','average_mood'])
    # plt.show()