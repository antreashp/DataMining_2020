import pandas as pd
from preprosses import  preprocess

filename = 'RAW_Data.pickle'
methods = ['average','max','max','max','max','max','max','max','max','max',
'max','max','max','max','max','max','max','max','average','average',
'average','average','average','average','average','average','average','average']
preprocess_instance = preprocess(filename, window_size=1, methods=methods)
preprocess_instance.normalize()
preprocess_instance.bin()
df = preprocess_instance.create_dataframe()
processed_df = preprocess_instance.create_dataframe_pros()
print(df.head(1))
print(processed_df.head(1))
