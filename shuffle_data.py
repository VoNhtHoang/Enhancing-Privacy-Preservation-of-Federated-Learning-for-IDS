import numpy as np
import pandas as pd
import dask.dataframe as dk
import tensorflow as tf

import os

#### Thay đổi hiển thị ####
pd.reset_option('display.max_columns')
pd.reset_option('display.max_rows')
pd.set_option('display.max_row', None)
pd.set_option('display.max_columns', None)
#### Change display ####

file_path = ["/mnt/c/Users/hoang/FileCSV_DACN_2025/IoT23.csv", "C:\\Users\\hoang\\FileCSV_DACN_2025\\IoT23.csv"]
out_path =  ["/mnt/c/Users/hoang/FileCSV_DACN_2025/shuffle_IoT23.csv", "C:\\Users\\hoang\\FileCSV_DACN_2025\\shuffle_IoT23.csv"]

if os.name == 'nt':
    file_path = file_path[1]
    out_path = out_path[1]
else:
    file_path = file_path[0]
    out_path = out_path[0]

print(file_path)

dictTypes = {}
df = dk.read_csv(file_path)
for col in df.columns:
    if col.startswith('proto') == True:
        dictTypes[col] = 'int32'
    elif col.startswith('service_') == True:
        dictTypes[col] = 'int32'
    elif col == 'label':
        dictTypes[col]= 'int32'
    elif col.startswith('detailed-label'):
        dictTypes[col] = 'str'
    else:
        dictTypes[col]='float32'
del df

df =dk.read_csv(file_path, dtype = dictTypes)
# df = df.drop(columns=['Variance', 'Weight'])
# print(df.npartitions)
print(df.dtypes)
print(df.tail(10))

from dask_ml.model_selection import train_test_split 
from sklearn.utils import shuffle

def shuffle_dask_dataframe(df, random_state=None):
    return (
        df.map_partitions(
            lambda part: part.assign(__shuffle_key=np.random.RandomState(random_state).random(len(part)))
        )
        .shuffle('__shuffle_key')
        .drop(columns='__shuffle_key')
    )
    
df = shuffle_dask_dataframe(df)

print(df.dtypes)
print(df.npartitions)
print(df.head(20))

header = True
for i in range(df.npartitions):
    df.get_partition(i).compute().to_csv(out_path, mode='a', header = header, index=False)
    header = False