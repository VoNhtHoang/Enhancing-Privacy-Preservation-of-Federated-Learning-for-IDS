import numpy as np
import pandas as pd
import dask.dataframe as dk
import tensorflow as tf
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout


input_files = ["file1.csv", "file2.csv", "file3.csv"]

temp_dir = "C:/Users/hoang/FileCSV_DACN_2025/"  # Thư mục lưu file tạm

input_files = [temp_dir + output_file for output_file in input_files]
print(input_files)

df = [dk.read_csv(input_file) for input_file in input_files]

batch_size = 512
ratio_test_all = 0.15

# from dask_ml.model_selection import train_test_split 
# # chia train test ratio 0.8:0.2 & random 
# train_df, test_df = train_test_split(df, test_size=ratio_test_all, random_state=42)

def dask_to_tf_dataset(dask_df, batch_size=128, num_classes=10): 
    def generator():
        for batch in dask_df.to_delayed():
            batch=batch.compute()  
            if batch.empty:
                continue

            X = batch.drop(columns='label').values.astype(np.float32)
            y = batch['label'].values
            y_onehot = to_categorical(y, num_classes=num_classes)  

            num_splits = max(1, len(X) // batch_size)  # Đảm bảo không chia nhỏ quá mức
            X_batches = np.array_split(X, num_splits)
            y_batches = np.array_split(y_onehot, num_splits)

            for X_batch, y_batch in zip(X_batches, y_batches):
                yield X_batch, y_batch
                
    output_signature = ( 
        tf.TensorSpec(shape=(None, 46), dtype=tf.float32), 
        tf.TensorSpec(shape=(None, 10), dtype=tf.int32),
    )
    
    return tf.data.Dataset.from_generator(generator, output_signature=output_signature).prefetch(tf.data.AUTOTUNE)


# train_df1, test_df1 = df1.random_split([1 - ratio_test_all, ratio_test_all])
# train_df2, test_df2 = df2.random_split([1 - ratio_test_all, ratio_test_all])
# train_df3, test_df3 = df3.random_split([1 - ratio_test_all, ratio_test_all])
train_dfs = []
test_dfs = []
for dff in df:
    train_df, test_df =dff.random_split([1 - ratio_test_all, ratio_test_all])
    train_dfs.append(train_df)
    test_dfs.append(test_df)
   

# train_gen1 = dask_to_tf_dataset(train_df1, 512, 10).repeat()
# train_gen2 = dask_to_tf_dataset(train_df2, 512, 10).repeat()
# train_gen3 = dask_to_tf_dataset(train_df3, 512, 10).repeat()
train_gens = [dask_to_tf_dataset(train_df, 512, 10).repeat() for train_df in train_dfs]

# test_gen1 = dask_to_tf_dataset(test_df1, 512, 10).repeat()
# test_gen2 = dask_to_tf_dataset(test_df2, 512, 10).repeat()
# test_gen3 = dask_to_tf_dataset(test_df3, 512, 10).repeat()
test_gens = [dask_to_tf_dataset(test_df , 512, 10).repeat() for test_df in test_dfs]

import datetime
import numpy as np

#
from server import Server
from client import Client
num_servers = 1
num_clients = 3

active_servers_list  = ['server_'+str(i)\
                        for i in range(num_servers)]
active_clients_list  = ['client_'+str(i)\
                        for i in range(num_clients)]

print(active_servers_list)
print(active_clients_list)

agents_dict= {}
serverObjects={}
clientObjects={}
serverObjects = {server_name: Server(server_name=server_name, \
                        active_clients_list=active_clients_list) \
                        for server_name in active_servers_list}

clientObjects = {client_name: Client(client_name, train_gens[clientID], test_gens[clientID], \
                        active_clients_list = active_clients_list) \
                        for clientID, client_name in enumerate(active_clients_list)}

# lưu dict
agents_dict['server'] = serverObjects
agents_dict['client'] = clientObjects

# init agents_dict vừa tạo vào client, server
for agent_name, agent in serverObjects.items():
    agent.set_agentsDict(agents_dict=agents_dict)
for agent_name, agent in clientObjects.items():
    agent.set_agentsDict(agents_dict=agents_dict)

client_name = 'client_1'
print("Agent_Dict: ", agents_dict['client'][client_name])

server = agents_dict['server']['server_0']

if __name__ == '__main__':
    server.InitLoop()
    server.final_statistics()