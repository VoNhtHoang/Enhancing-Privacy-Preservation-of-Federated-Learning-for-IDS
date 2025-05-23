import copy
import sys, os
import random
import threading
from warnings import simplefilter
from datetime import datetime, timedelta
from sklearn import metrics

import numpy as np
import tenseal as ts
import tensorflow as tf
import pandas as pd

######
from tensorflow import keras
from keras.models import load_model
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.models import load_model

#####
from dp_mechanisms import laplace

##### CODE SECTION
LATENCY_DICT = {}


class Message:
    def __init__(self, sender_name, recipient_name, body):
        self.sender = sender_name
        self.recipient = recipient_name
        self.body = body

    def __str__(self):
        return "Message from {self.sender} to {self.recipient}.\n Body is : {self.body} \n \n"

class Client():
    def __init__(self, client_name, data_train, data_val, data_test, steps_per_epoch, val_steps, test_steps, active_clients_list, he_context):
        self.client_name = client_name
        self.active_clients_list = active_clients_list
        self.data_train = data_train
        self.data_test = data_test
        self.data_val = data_val
        self.agent_dict = {}
        self.temp_dir = "federated_learning_log/"+ datetime.now().strftime("Month %m -- Day %d___%Hh %Mp")
        os.makedirs(self.temp_dir, exist_ok=True)
        self.temp_dir = self.temp_dir +"/"+ client_name + "_log"
        os.makedirs(self.temp_dir, exist_ok=True)
        
        
        ## global
        self.global_weights = {}
        self.global_biases = {}
        self.global_accuracy = {}
        self.global_loss = {}
        self.global_test_acc ={}
        self.global_test_loss= {}
        
        ## local
        self.model = self.init_model()
        self.local_weights = {}
        self.local_weights_shape=[]
        self.local_biases_shape= []
        self.local_biases = {}
        self.local_accuracy = {}
        self.local_loss = {}
        self.compute_times = {} # proc weight
        self.he_context = he_context
        self.convergence = 0 #số lần hội tụ qua nhiều iteration
        self.unconvergence =0 
        
        # dp parameter
        self.alpha = 1.0
        self.epsilon = 0.1
        self.mean = 0
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = val_steps
        self.test_steps = test_steps
        self.local_weights_noise = {}
        self.local_biases_noise = {}
        
        for name in active_clients_list:
            if name not in LATENCY_DICT.keys():
                LATENCY_DICT[name]={}
        if 'server_0' not in LATENCY_DICT.keys():
            LATENCY_DICT['server_0']={}
                    
        LATENCY_DICT['server_0']={client_name: timedelta(seconds=0.1) for client_name in active_clients_list}
        for client_name in active_clients_list:
            LATENCY_DICT[client_name]['server_0'] = timedelta(seconds= np.random.random())
            
    def get_clientID(self):
        return self.clientID
    
    def set_agentsDict(self, agents_dict):
        self.agents_dict = agents_dict
    
    def set_steps_per_epoch(self, steps_per_epoch=50):
        self.steps_per_epoch = steps_per_epoch
        
    def get_steps_per_epoch(self):
        print("Train steps: ", self.steps_per_epoch)
        
    def set_validation_steps(self, validation_steps):
        self.validation_steps = validation_steps
        
    def get_validation_steps(self):
        print("Val steps: ", self.validation_steps)

    def set_test_steps(self, test_steps):
        self.test_steps = test_steps
        
    def get_test_steps(self):
        print("Test steps: ", self.test_steps)
        
############################## INIT MODEL ########################################
    def init_model(self):
        features, labels = next(iter(self.data_train))
        input_shape = (features.shape[1], 1)
        output_shape = labels.shape[1]
        
        """====================== Classification ====================="""
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv1D(filters=128, kernel_size=3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv1D(filters=128, kernel_size=3,  padding="same",activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(output_shape, activation='softmax')
        ])
        
        # =============  binary =====================
        # model = keras.Sequential([
        #     layers.Input(shape=input_shape),
        #     layers.Conv1D(filters=128, kernel_size=3, padding="same", strides=1, activation="relu"),
        #     layers.BatchNormalization(),
        #     layers.MaxPooling1D(pool_size=2),
        #     layers.Conv1D(filters=128, kernel_size=3, padding="same", strides=1, activation="relu"),
        #     layers.BatchNormalization(),
        #     layers.MaxPooling1D(pool_size=2),
        #     layers.Flatten(),
        #     layers.Dropout(0.5),
        #     layers.Dense(128, activation='relu'),
        #     layers.Dense(64, activation='relu'),  
        #     layers.Dropout(0.3),                  
        #     layers.Dense(1, activation='sigmoid')
        # ])
                # free
        del input_shape, features, labels
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

############################  INIT MODEL ######################################

##########################################     HE CONTEXT     #################################################
    def init_he_context(self):
        """Thiết lập context mã hóa đồng hình"""
        context = ts.context(
            ts.SCHEME_TYPE.CKKS, # ckks cho số thực, bfv cho int
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        context.generate_galois_keys()
        context.global_scale = 2**40
        return context
    
    def he_params_encryption(self, flattened_weights, flattened_biases):
        encrypted_weights = ts.ckks_vector(self.he_context, flattened_weights)
        encrypted_biases = ts.ckks_vector(self.he_context, flattened_biases)
        # print("$$$$$flatten_bias$$$$$", flattened_biases.shape)
        return encrypted_weights.serialize() , encrypted_biases.serialize()
    
    def he_params_decryption(self, encrypted_weights, encrypted_biases):
        decrypted_weights = np.array(encrypted_weights.decrypt())
        decrypted_biases = np.array(encrypted_biases.decrypt())
        # print("======Decrypted Biases=======", decrypted_biases.shape)
        
        return decrypted_weights , decrypted_biases
##########################################     HE CONTEXT     #################################################


##########################################     FLATTEN       ##############################################
    def save_shape(self, weights, biases, iteration):
        if iteration <2:
            for index, weights_layer in enumerate(weights):
                weights_shape = weights_layer.shape
                self.local_weights_shape.append(weights_shape)
                
            for index, bias in enumerate(biases):
                bias_shape = bias.shape
                self.local_biases_shape.append(bias_shape)
        return None
    def flatten_weights(self, weights, biases):
        
        # print("Biases shapes: ",self.local_biases_shape)
        arr_1 = [weight.flatten() for weight in weights]
        arr_2 = [bias.flatten() for bias in biases]
        return np.concatenate(arr_1).ravel(), np.concatenate(arr_2).ravel()

    def de_flatten_weights(self, flattened_weights, flattened_biases):
        # print("+++ weight Length +++", len(flattened_weights))
        # print("=== Biases Shape ===", len(flattened_biases))
        
        weights=[]
        right_pointer=0
        for shape in self.local_weights_shape:
            delta = 1
            for i in shape:
                delta*=i
            weights.append(np.array(flattened_weights[right_pointer*1:right_pointer+delta].reshape(shape))) 
            right_pointer +=delta
            
        biases = []
        right_pointer=0
        for shape in self.local_biases_shape:
            delta = shape[0]
            # print(self.client_name+ f"{shape}")
            biases.append(np.array(flattened_biases[right_pointer:right_pointer+delta].reshape(shape)))
            right_pointer +=delta
        return weights, biases
##########################################     FLATTEN     ###############################################


################################################ ADD NOISE #####################################################
    def add_gamma_noise(self, local_weights, local_biases, iteration):
        weights_dp_noise=[]
        biases_dp_noise=[]
        sensitivity =  2 / (len(self.active_clients_list)
                          *self.steps_per_epoch*self.alpha)
        for weight in local_weights:
            if abs(weight) > 1e-15:
                weights_dp_noise.append(laplace(mean=self.mean, 
                                    sensitivity=sensitivity,
                                    epsilon=self.epsilon))
            else:
                weights_dp_noise.append(0)
                
        for weight in local_biases:
            if abs(weight) > 1e-15:
                biases_dp_noise.append(laplace(mean=self.mean, 
                                    sensitivity=sensitivity,
                                    epsilon=self.epsilon))
            else:
                biases_dp_noise.append(0)
        
        self.local_weights_noise[iteration] = weights_dp_noise
        self.local_biases_noise[iteration] = biases_dp_noise
        
        weights_with_noise = local_weights + weights_dp_noise
        biases_with_noise = local_biases + biases_dp_noise

        return np.array(weights_with_noise), np.array(biases_with_noise)
################################################ ADD NOISE #####################################################


################################################ MODEL FIT ###################################################
    def model_fit(self, iteration):
        file_path = self.temp_dir +"/Iteration_"+str(iteration)+".csv"
        file_path_model = self.temp_dir+"/model_"+str(iteration)+".keras"
         
        csv_logger = CSVLogger(file_path, append=True)
        
        if iteration > 1:
            print(f"{iteration} {self.client_name} Model update params!")
            # Set Weights và Biases tất cả các layer
            index =0
            for layer in self.model.layers:
                if layer.name.startswith('conv1d') or layer.name.startswith('dense'):
                    layer.set_weights([self.local_weights[iteration-1][index], self.local_biases[iteration-1][index]])
                    index+=1
        
        #steps = int(np.ceil(self.steps_per_epoch / 100))
        self.model.fit(self.data_train, epochs= 4,
                  validation_data= self.data_val, validation_steps= self.validation_steps , 
                  steps_per_epoch= self.steps_per_epoch, verbose = 1, callbacks=[csv_logger])
        
        self.model.save(file_path_model)
        
        print("Come done model fit")
        weights = []
        biases = []
        for layer in self.model.layers:
            if layer.name.startswith('conv1d') or layer.name.startswith('dense'):
                weights.append(layer.get_weights()[0])
                biases.append(layer.get_weights()[1])
        return weights, biases
################################################ MODEL FIT #####################################################


############################################## PRODUCE WEIGHTS #############################################################   
    def proc_weights(self, message):
        start_time = datetime.now()
        body = message.body
        iteration, lock, simulated_time = body['iteration'], body['lock'], body['simulated_time']
        
        weights, biases = self.model_fit(iteration) #,biases
        
        # Cập nhật local weights theo iteration
        self.local_weights[iteration] = weights
        self.local_biases[iteration] = biases
        # print("Local weight Length: ", len(self.local_weights[iteration]))
        
        # Flatten weights
        self.save_shape(weights, biases, iteration)
        weights, biases = self.flatten_weights(weights, biases)
        
        # add noise - lock để đảm bảo không xung đột tài nguyên máy khi HE
        lock.acquire()  # for random seed
        weights, biases = self.add_gamma_noise(local_weights=weights,  local_biases= biases,  iteration=iteration)   #, biases
        weights, biases = self.he_params_encryption(weights, biases)   #, final_encrypted_biases
        lock.release()
          
        #end
        end_time = datetime.now()
        compute_time = end_time - start_time
        self.compute_times[iteration] = compute_time

        # Giả sử không độ trễ 
        simulated_time += compute_time + LATENCY_DICT[self.client_name]['server_0']
        print(type(compute_time))
        body = {'context' : self.he_context.serialize(save_secret_key=False),
                # 'weights_original_shape': self.local_weights_shape,
                'encrypted_weights': weights,
                'encrypted_biases': biases,
                'iter': iteration,
                'compute_time': compute_time,
                'simulated_time': simulated_time}  # generate body

        print(self.client_name + "End Produce Weights")
        msg = Message(sender_name=self.client_name, recipient_name=self.agents_dict['server']['server_0'], body=body)
        return msg

############################################## RECEIVE WEIGHTS #########################################################  
    def recv_weights(self, message):
        body = message.body
        iteration, simulated_time = body['iteration'], body['simulated_time']
        
        # Giải mã thông số nhận được từ server
        encrypted_weights = ts.lazy_ckks_vector_from(body['encrypted_weights'])
        encrypted_weights.link_context(self.he_context)
        
        encrypted_biases = ts.lazy_ckks_vector_from(body['encrypted_biases'])
        encrypted_biases.link_context(self.he_context)
        
        # weights_original_shape = body['weights_original_shape']
        return_weights, return_biases = self.he_params_decryption(
            encrypted_weights, encrypted_biases
        )

        ## free
        del encrypted_weights , encrypted_biases
        
        ## remove dp
        return_weights -= self.local_weights_noise[iteration]
        return_biases -= self.local_biases_noise[iteration]
        
        self.global_weights[iteration], self.global_biases[iteration] = self.de_flatten_weights(return_weights, return_biases)
        
        # Tính độ hội tụ
        # check whether weights have converged
        
        self.local_accuracy[iteration], self.local_loss[iteration] = self.evaluate_accuracy(self.local_weights[iteration], self.local_biases[iteration])
        self.global_accuracy[iteration], self.global_loss[iteration] = self.evaluate_accuracy(self.global_weights[iteration], self.global_biases[iteration])
        
        ## Lưu global acc, loss, # precision, recall tùy ý
        history = {
            "global_acc": [],
            "global_loss": []
        }
        history['global_acc'].append(self.global_accuracy[iteration])
        history['global_loss'].append(self.global_loss[iteration])
        
        file_his = self.temp_dir+"/global_val.csv"
        if iteration ==1:
            pd.DataFrame(history).to_csv(file_his, index=False, header= True, mode='a')
        else:
            pd.DataFrame(history).to_csv(file_his, index=False, header= False, mode='a')
            
        
        # Kiểm tra hội tụ Có biến self.convergence ở trên
        converged = self.check_convergence(iteration)

        args = [self.client_name, iteration, self.local_accuracy[iteration], self.local_loss[iteration], self.global_accuracy[iteration], self.global_loss[iteration]]
        iteration_report = 'Performance Metrics for {} on iteration {} \n' \
                           '------------------------------------------- \n' \
                           'local accuracy: {} \n' \
                            'local loss: {} \n'\
                           'global accuracy: {} \n' \
                            'global_loss: {} \n' \
        

        args.append(self.compute_times[iteration])
        iteration_report += 'local compute time: {} \n'

        args.append(simulated_time)
        iteration_report += 'Simulated time to receive global weights: {} \n \n'
        
        print("Arguments: ",iteration_report.format(*args))

        msg = Message(sender_name=self.client_name,
                      recipient_name='server_0',
                      body={'converged': converged,
                            'simulated_time': simulated_time + LATENCY_DICT[self.client_name]['server_0']})
        return msg

############################################## PREDICT + EVALUATE #########################################################          
    def evaluate_accuracy(self,weights, biases):
        index =0 
        for layer in self.model.layers:
            if layer.name.startswith('conv1d') or layer.name.startswith('dense'):
                layer.set_weights([weights[index], biases[index]])
                index+=1
        loss, accuracy= self.model.evaluate(self.data_test, steps = self.test_steps)
        return accuracy, loss
############################################## PREDICT + EVALUATE #########################################################   


############################################## CHECK HỘI TỤ #########################################################       
    def check_convergence(self, iteration):
        #
        tolerance_left_edge = 0.2
        tolerance_right_edge=2.0
        
        if iteration > 1:
            if self.global_loss[iteration]>self.global_loss[iteration-1]:
                self.unconvergence +=1
            else:
                self.unconvergence -=1
                if self.unconvergence < 0:
                    self.unconvergence=0
            if self.global_accuracy[iteration] <= self.global_accuracy[iteration-1]:
                self.unconvergence +=1
            else:
                self.unconvergence -=1
                if self.unconvergence < 0:
                    self.unconvergence=0
        
        if np.std(self.global_loss[iteration]) < 0.05:
            self.convergence += 1
            
        flattened_global_weights, flattened_global_bias = self.flatten_weights(self.global_weights[iteration],self.local_biases[iteration])
        flattened_local_weights, flattened_local_bias = self.flatten_weights(self.local_weights[iteration],self.local_biases[iteration])

        weights_differences = np.abs(flattened_global_weights, flattened_local_weights)
        biases_differences = np.abs(flattened_global_bias, flattened_local_bias)

        # print("weights dif", weights_differences)
        # print("biases diff", biases_differences)
        if (weights_differences < tolerance_left_edge).all() and (biases_differences <tolerance_left_edge).all():
            self.convergence+=1
        elif (weights_differences > tolerance_right_edge).all() and (biases_differences > tolerance_right_edge).all():
            self.convergence+=1
        else:
            self.convergence -=1
            if self.convergence <0:
                self.convergence=0
        
        #4 điểm liên tiếp ok
        if(self.convergence > 3 and self.unconvergence<3):
            return True
        elif self.unconvergence>3:
            return True
                    
        return False
############################################## CHECK HỘI TỤ ######################################################### 



############################################## REMOVE CLIENTS #############################################################
    def remove_active_clients(self, message):
        body = message.body
        removing_clients, simulated_time, iteration \
        = body['removing_clients'], body['simulated_time'], body['iteration']
        
        print(f'[{self.client_name}] :Simulated time for client {removing_clients} to finish iteration {iteration}: {simulated_time}\n')

        self.active_clients_list = [active_client for active_client in self.active_clients_list if active_client not in removing_clients]
        return None
    
############################################### REMOVE CLIENTS ############################################################