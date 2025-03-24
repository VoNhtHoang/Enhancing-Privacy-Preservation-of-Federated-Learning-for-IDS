import copy
import numpy as np
import sys
import random
import threading
from warnings import simplefilter
from datetime import datetime, timedelta
from sklearn import metrics
from keras.models import load_model

#####
from dp_mechanisms import laplace


##### CODE SECTION
LATENCY_DICT = {}

tolerance = 10.0

class Message:
    def __init__(self, sender_name, recipient_name, body):
        self.sender = sender_name
        self.recipient = recipient_name
        self.body = body

    def __str__(self):
        return "Message from {self.sender} to {self.recipient}.\n Body is : {self.body} \n \n"

class Client():
    def __init__(self, client_name, data_train, data_test, \
        active_clients_list):
        self.client_name = client_name
        self.active_clients_list = active_clients_list
        self.data_train = data_train
        self.data_test = data_test
        self.agent_dict = {}
        
        ## global
        self.global_weights = {}
        self.global_biases = {}
        self.global_accuracy = 0.0
        
        ## local
        self.local_weights = {}
        self.local_biases = {}
        self.local_accuracy = 0.0
        self.compute_time = {} # proc weight
        
        # dp parameter
        self.alpha = 1.0
        self.espilon = 1.0
        self.mean = 0
        self.local_weights_noise ={}
        self.local_biases_noise = {}
        
        if 'server_0' not in LATENCY_DICT.keys():
            LATENCY_DICT['server_0'] = {}

        for client_name in self.active_clients_list:
            if client_name not in LATENCY_DICT.keys():
                LATENCY_DICT[client_name] = {client_name2: timedelta(seconds=0.1) for client_name2 in active_clients_list}
            LATENCY_DICT[client_name]['server_0'] = timedelta(seconds=0.1)
            LATENCY_DICT['server_0'][client_name] = timedelta(seconds=0.1)

        LATENCY_DICT['client_1'] = {client_name: timedelta(seconds=2.0) for client_name in active_clients_list}
        LATENCY_DICT['client_1']['server_0'] = timedelta(seconds=2.0)
        LATENCY_DICT['server_0']['client_1'] = timedelta(seconds=2.0)

        LATENCY_DICT['client_0']['server_0'] = timedelta(seconds=0.3)
        LATENCY_DICT['server_0']['client_0'] = timedelta(seconds=0.3)
        
    def get_clientID(self):
        return self.clientID
    
    def set_agentsDict(self, agents_dict):
        self.agents_dict = agents_dict
    
    def add_gamma_noise(self, local_weights, local_biases, iteration):
        weights_shape = local_weights.shape
        weights_dp_noise = np.zeros(weights_shape)

        biases_shape = local_biases.shape
        biases_dp_noise = np.zeros(biases_shape)
        
        len_per_iteration = 50 # data_train / iteration
        
        sensitivity =  2 / (len(self.active_clients_list)
                          *len_per_iteration*self.alpha)
        
        for i in range(weights_shape[0]):  # weights_modified is 2-D
            for j in range(weights_shape[1]):
                dp_noise = laplace(mean=self.mean, 
                                sensitivity=sensitivity,
                                epsilon=self.epsilon)
                weights_dp_noise [i][j] = dp_noise
                
        
        for i in range(biases_shape[0]):
            dp_noise = laplace(mean=self.mean,
                               sensitivity=sensitivity,
                               epsilon=self.epsilon)
            biases_dp_noise [i] = dp_noise
        
        weights_with_noise = copy.deepcopy(local_weights)  # make a copy to not mutate weights
        biases_with_noise = copy.deepcopy(local_biases)

        self.local_weights_noise[iteration] = weights_dp_noise
        weights_with_noise += weights_dp_noise
        self.local_biases_noise[iteration] = biases_dp_noise
        biases_with_noise += biases_dp_noise
        return weights_with_noise, biases_with_noise
    
    def model_fit(iteration):
        model = load_model("cnn_model_2-0_batch512_test015.h5")

        import numpy as np

        weights_list = []
        biases_list = []

        for layer in model.layers:
            if len(layer.get_weights()) == 2:  # Chỉ lấy các layer có weights & biaseses
                weights, biases = layer.get_weights()
                weights_list.append(weights)
                biases_list.append(biases)

        return weights, biases
    
    def proc_weights(self, message):
        start_time = datetime.now()
        body = message.body
        iteration, lock, simulated_time = body['iteration'], body['lock'], body['simulated_time']

        # if iteration - 1 > len(self.train_datasets):  # iteration is indexed starting from 1
        #     raise (ValueError(
        #         'Not enough data to support a {}th iteration. Either change iteration data length in config.py or decrease amount of iterations.'.format(
        #             iteration)))
        
        weights, biases = self.model_fit(iteration)

        self.local_weights[iteration] = weights
        self.local_biases[iteration] = biases

        final_weights, final_biases = copy.deepcopy(weights), copy.deepcopy(biases)

        # add noise lock để đảm bảo không xung đột
        lock.acquire()  # for random seed
        final_weights, final_biases = \
            self.add_noise(weights=weights, biases=biases, iteration=iteration)
        lock.release()
        
        #end
        end_time = datetime.now()
        compute_time = end_time - start_time
        self.compute_times[iteration] = compute_time
        # multiply latency by two: first the server has to request the value, then the client has to return it

        simulated_time += compute_time + LATENCY_DICT[self.client_name]['server_0']

        body = {'weights': final_weights, 'biases': final_biases, 'iter': iteration,
                'compute_time': compute_time, 'simulated_time': simulated_time}  # generate body

        msg = Message(sender_name=self.client_name, recipient_name=self.agents_dict['server']['server_0'], body=body)
        return msg

    def recv_weights(self, message):
        body = message.body
        iteration, return_weights, return_biases, simulated_time \
        = body['iteration'], body['return_weights'],
        body['return_biases'], body['simulated_time']
        
        return_weights = copy.deepcopy(return_weights)
        return_biases = copy.deepcopy(return_biases)
        
        ## remove dp
        return_weights -= self.weights_dp_noise[iteration] / len(self.active_clients_list)
        return_biases -= self.biases_dp_noise[iteration] / len(self.active_clients_list)
        
        self.global_weights[iteration] = return_weights
        self.global_biases[iteration]  = return_biases
        
        local_weights = self.local_weights[iteration]
        local_biases = self.local_biases[iteration]

        # Tính độ hội tụ
        converged = self.check_convergence((local_weights, local_biases), (
            return_weights, return_biases))  # check whether weights have converged
        
        local_accuracy = self.evaluator.accuracy(local_weights, local_biases)
        global_accuracy = self.evaluator.accuracy(return_weights, return_biases)
        
        
        self.local_accuracy[iteration] = local_accuracy
        self.global_accuracy[iteration] = global_accuracy

        args = [self.name, iteration, local_accuracy, global_accuracy]
        iteration_report = 'Performance Metrics for {} on iteration {} \n' \
                           '------------------------------------------- \n' \
                           'local accuracy: {} \n' \
                           'global accuracy: {} \n' \
        
        #latency - độ trễ giữa các client
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
        
    def evaluate_accuracy(local_weights, local_biases):
        # self.logisticRegr.coef_ = weights  # override weights and coefficients
        # self.logisticRegr.intercept_ = biases
        # return self.logisticRegr.score(self.X_test, self.Y_test)
        return 0.9
    
    def check_convergence(self, local_params, global_params):
        local_weights, local_biases = local_params
        global_weights, global_biases = global_params

        weights_differences = np.abs(global_weights - local_weights)
        biases_differences = np.abs(global_biases - local_biases)
        return (weights_differences < tolerance).all() and (
                biases_differences < tolerance).all()  # check all weights are close enough

    def remove_active_clients(self, message):
        body = message.body
        clients_to_remove, simulated_time, iteration \
        = body['clients_to_remove'], body['simulated_time'], body['iteration']
        
        print(f'[{self.client_name}] :Simulated time for client {clients_to_remove} to finish iteration {iteration}: {simulated_time}\n')

        self.active_clients_list -= clients_to_remove
        return None