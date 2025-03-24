import copy
import numpy as np
import sys
import random
import config
import threading
from warnings import simplefilter
from datetime import datetime
from sklearn import metrics

from agent_class import Agent
from message import Message
from utils.dp_mechanisms import laplace
import utils.diffie_hellman as dh

from sklearn.linear_model import SGDClassifier

class ClientAgent(Agent):
    def __init__(self, agent_number, train_datasets, evaluator, active_clients):

        super(ClientAgent, self).__init__(agent_number=agent_number, agent_type="client_agent")

        self.train_datasets = train_datasets
        self.evaluator = evaluator
        self.active_clients = active_clients

        assert (self.directory is not None)
        clients = self.directory.clients
        num_clients = len(clients)
        
        pubkeyList, seckeyList = dh.keygeneration(num_clients, self.agent_number)

        # note this works because dicts are ordered in Python 3.6+
        self.pubkeyList = dict(zip(clients.keys(), pubkeyList))
        self.seckeyList = dict(zip(clients.keys(), seckeyList))
        
        self.directory = None
        self.pubkeyList = None
        self.seckeyList = None
        
        self.otherkeyList = {agent_name: None for agent_name, __ in clients.items()}
        self.otherkeyList[self.name] = 0

        self.commonkeyList = {agent_name: None for agent_name, __ in clients.items()}
        self.commonkeyList[self.name] = 0

        self.seeds = {agent_name: None for agent_name, __ in clients.items()}
        self.seeds[self.name] = 0

        self.deltas = {agent_name: None for agent_name, __ in clients.items()}
        self.deltas[self.name] = 0

        self.computation_times = {}

        self.local_weights = {}  # local weights. Maps iteration (int) to weights (numpy array)
        self.local_bias = {}

        self.weights_dp_noise = {}  # keyed by iteration; noise added at each iteration
        self.bias_dp_noise = {}

        self.global_weights = {}  # averaged weights
        self.global_bias = {}

        self.local_accuracy = {}
        self.global_accuracy = {}
        
    def add_noise(self, weights, bias, iteration):
        # :return: weights, bias
        # :rtype: numpy arrays
        
        # Lấy shape weights
        weights_shape = weights.shape
        weights_dp_noise = np.zeros(weights_shape)

        bias_shape = bias.shape
        bias_dp_noise = np.zeros(bias_shape)

        # generate DP parameters
        active_clients_lens = [config.LENS_PER_ITERATION[client_name] for client_name in self.active_clients]


        smallest_dataset = min(active_clients_lens)
        if config.USING_CUMULATIVE:
            smallest_dataset *= iteration

        sensitivity = 2 / (len(self.active_clients) * smallest_dataset * config.alpha)
        epsilon = config.EPSILONS[self.name]

        random.seed(config.RANDOM_SEEDS[self.name][iteration])
        # adding differentially private noise
        for i in range(weights_shape[0]):
            for j in range(weights_shape[1]):
                if config.DP_ALGORITHM == 'Laplace':
                    dp_noise = laplace(mean=config.mean, sensitivity=sensitivity, epsilon=epsilon)
                weights_dp_noise[i][j] = dp_noise

        if config.bias_DP_NOISE:
            for i in range(bias_shape[0]):
                if config.DP_ALGORITHM == 'Laplace':
                    dp_noise = laplace(mean=config.mean, sensitivity=sensitivity, epsilon=epsilon)
                bias_dp_noise[i] = dp_noise


        weights_with_noise = copy.deepcopy(weights)  # make a copy to not mutate weights
        bias_with_noise = copy.deepcopy(bias)

        self.weights_dp_noise[iteration] = weights_dp_noise # dict mỗi iteration
        weights_with_noise += weights_dp_noise
        self.bias_dp_noise[iteration] = bias_dp_noise
        bias_with_noise += bias_dp_noise
        
        return weights_with_noise, bias_with_noise
    
    def produce_weights(self, message):
        # tính tg
        start_time = datetime.now()
        body = message.body
        iteration, lock, simulated_time = body['iteration'], body['lock'], body['simulated_time']

        if iteration - 1 > len(self.train_datasets):  # iteration is indexed starting from 1
            raise (ValueError(
                'Not enough data to support a {}th iteration. Either change iteration data length in config.py or decrease amount of iterations.'.format(
                    iteration)))
            
        # train/test -> weights & biases
        weights, biases = self.compute_weights_noncumulative(iteration)

        self.local_weights[iteration] = weights
        self.local_biases[iteration] = biases

        # create copies of weights and biases since we may be adding to them
        final_weights, final_biases = copy.deepcopy(weights), copy.deepcopy(biases)

        if config.USE_DP_PRIVACY: # nếu nhét nhiễu vào
            lock.acquire()  # for random seed
            final_weights, final_biases = self.add_noise(weights=weights, biases=biases, iteration=iteration)
            lock.release()

        # if config.USE_SECURITY:  # thêm offset diffle Hellman
        #     final_weights, final_biases = self.add_security_offsets(weights=final_weights, biases=final_biases)
                
        end_time = datetime.now()
        computation_time = end_time - start_time
        self.computation_times[iteration] = computation_time
        # multiply latency by two: first the server has to request the value, then the client has to return it
        
        simulated_time += computation_time + config.LATENCY_DICT[self.name]['server_agent']
        
        body = {'weights': final_weights, 'biases': final_biases, 'iter': iteration,
                'computation_time': computation_time, 'simulated_time': simulated_time}  # generate body

        return Message(sender_name=self.name, recipient_name=self.directory.server_agents, body=body)
    
    def compute_weights_noncumulative(self, iteration):
        X, y = self.train_datasets[iteration]

        lr = SGDClassifier(alpha=0.0001, loss="log", random_state=config.RANDOM_SEEDS[self.name][iteration])

        # Assign prev round coefficients
        if iteration > 1:
            global_weights = copy.deepcopy(self.global_weights[iteration - 1])
            global_biases = copy.deepcopy(self.global_biases[iteration - 1])
        else:
            global_weights = None
            global_biases = None

        lr.fit(X, y, coef_init=global_weights, intercept_init=global_biases)
        local_weights = lr.coef_
        local_biases = lr.intercept_

        return local_weights, local_biases
    
    def update_deltas(self):
        """
        Updates commonkeyList. Called after each iteration to update values.
        """
        if None not in self.commonkeyList.values():  # if first time calling this function
            agents_and_seeds = self.commonkeyList.items()
            self.commonkeyList = self.commonkeyList.fromkeys(self.commonkeyList.keys(), None)
        else:
            # use exisitng seed to generate new seeds and offsets
            agents_and_seeds = self.seeds.items()

        for agent, seed in agents_and_seeds:
            # uses current seeds to generate new deltas and new seeds
            if agent != self.name:
                seed_b = bin(seed)  # cast to binary
                delta_b = seed_b[:20]
                delta = int(delta_b, 2)  # convert back to decimal from base 2

                seed_b = seed_b[20:]
                seed = int(seed_b, 2)
                random.seed(seed) # generate new seed
                seed = random.randint(-sys.maxsize, sys.maxsize)
                self.seeds[agent] = seed
                self.deltas[agent] = delta
    
    def receive_weights(self, message):
        """
        Called by server agent to return global weights.
        :param message: message containing return weights and other necessary information
        :type message: Message
        :return: Message indicating whether client has converged in training this iteration, which only
        matters if config.CLIENT_DROPOUT is True.
        :rtype: Message
        """
        body = message.body
        iteration, return_weights, return_biases, simulated_time = body['iteration'], body['return_weights'], body[
            'return_biases'], body['simulated_time']

        return_weights = copy.deepcopy(return_weights)
        return_biases = copy.deepcopy(return_biases)

        if config.USE_DP_PRIVACY and config.SUBTRACT_DP_NOISE:
            # subtract your own DP noise
            return_weights -= self.weights_dp_noise[iteration] / len(self.active_clients)
            return_biases -= self.biases_dp_noise[iteration] / len(self.active_clients)

        self.global_weights[iteration] = return_weights
        self.global_biases[iteration] = return_biases

        local_weights = self.local_weights[iteration]
        local_biases = self.local_biases[iteration]

        converged = self.satisfactory_weights((local_weights, local_biases), (
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

        if config.SIMULATE_LATENCIES:
            args.append(self.computation_times[iteration])
            iteration_report += 'local computation time: {} \n'

            args.append(simulated_time)
            iteration_report += 'Simulated time to receive global weights: {} \n \n'

        if config.VERBOSITY:
            print(iteration_report.format(*args))

        msg = Message(sender_name=self.name, recipient_name=self.directory.server_agents,
                      body={'converged': converged,
                            'simulated_time': simulated_time + config.LATENCY_DICT[self.name]['server_agent0']})
        return msg
    
    def check_covergence(self, local, global_params): # kiểm tra độ hội tụ
        local_weights, local_biases = local
        global_weights, global_biases = global_params

        weights_differences = np.abs(global_weights - local_weights)
        biases_differences = np.abs(global_biases - local_biases)
        return (weights_differences < config.tolerance).all() and (
                biases_differences < config.tolerance).all()  # check all weights are close enough
        
    def remove_active_clients(self, message):
        """
        Nếu check độ hội tụ mà trả về close thì drop client
        :return: None
        """
        body = message.body
        clients_to_remove, simulated_time, iteration = body['clients_to_remove'], body['simulated_time'], body[
            'iteration']

        print('Simulated time for client {} to finish iteration {}: {}\n'.format(self.name, iteration, simulated_time))

        self.active_clients -= clients_to_remove
        return None