import sys
sys.path.append('..')

import numpy as np
from datetime import datetime, timedelta
import multiprocessing
from multiprocessing.pool import ThreadPool


###




###
def find_slowest_time(messages):
    simulated_communication_times = {message.sender: message.body['simulated_time'] for message in messages}
    slowest_client = max(simulated_communication_times, key=simulated_communication_times.get)
    simulated_time = simulated_communication_times[slowest_client]  # simulated time it would take for server to receive all values
    return simulated_time

num_iterations = 3
LATENCY_DICT = {}

class Message:
    def __init__(self, sender_name, recipient_name, body):
        self.sender = sender_name
        self.recipient = recipient_name
        self.body = body
        
    def __str__(self):
        return "Message from {self.sender} to {self.recipient}.\n Body is : {self.body} \n \n"
    
class Server():
    def __init__(self,server_name, active_clients_list):
        self.server_name = server_name
        self.global_weights = {}
        self.global_biases = {}
        self.active_clients_list = active_clients_list
        self.agents_dict = {}
        
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
        
    def set_agentsDict(self, agents_dict):
        self.agents_dict = agents_dict
    
    def get_av(self):
        return self.active_clients_list
    
    def get_agentsDict(self):
        return self.agents_dict
    
    def initIterations():
        return None
    
    def client_compute_caller(input_tuple):
        clientObject, message = input_tuple
        return_message = clientObject.proc_weights(message=message)
        return return_message


    def client_weights_returner(input_tuple):
        clientObject, message = input_tuple
        converged = clientObject.recv_weights(message)
        return converged


    def client_agent_dropout_caller(input_tuple):
        clientObject, message = input_tuple
        __ = clientObject.remove_active_clients(message)
        return None


    def InitLoop(self):
        converged_clients = {}
        active_clients_list = self.active_clients_list
        
        for iteration in range(1, num_iterations+1):
            weights = {}
            biases = {}
            
            m = multiprocessing.Manager()
            
            lock = m.lock()
            
            with ThreadPool(len(active_clients_list)) as calling_init_pool:
                arguments = []
                
                for client_name in active_clients_list:
                    clientObject = self.agents_dict['cliennt'][client_name]
                    
                    body = {'iteration': iteration, 'lock': lock, 'simulated_time': LATENCY_DICT[self.server_name][client_name]}
                    #message from server to client
                    msg = Message(sender_name=self.server_name, recipient_name=client_name, body = body)
                    
                    arguments.append((clientObject, msg))
                calling_returned_messages = calling_init_pool.map(self.client_compute_caller, arguments)
            
            
            start_call_time = datetime.now()
            simulated_time = find_slowest_time(calling_returned_messages)
            
            temp_sum_weights = sum(message.body['weights'] for message in calling_returned_messages)
            temp_sum_biases = sum(message.body['biases'] for message in calling_returned_messages)
            
            self.global_weights[iteration] = temp_sum_weights/len(self.active_clients_list)
            self.global_biases[iteration] = temp_sum_biases/len(self.active_clients_list)
            
            # add time server logic takes
            end_call_time = datetime.now()
            server_logic_time = end_call_time - start_call_time
            simulated_time += server_logic_time #Tổng thời gian cho đến bước này
            
            
            # Trả weights với bias trung bình mới về client 
            with ThreadPool(len(active_clients_list)) as returning_pool:
                arguments = []
                
                for client_name in active_clients_list:
                    clientObject = self.agents_dict['client'][client_name]
                    
                    body = {'iteration': iteration, 'return_weights' : self.global_weights, 
                            'return-biases': self.global_biases, 'simulated_time': simulated_time}
                    
                    msg = Message(sender_name=self.server_name, recipient_name=client_name, body=body)
                    
                    arguments.append((clientObject, msg))
                returned_messages = returning_pool.map(self.client_weights_returner, arguments)
            
            
            simulated_time = find_slowest_time(returned_messages)
            start_return_time = datetime.now()
            
            removing_clients = set()
            
            for message in returned_messages:
                if message.body['converged'] == True and message.sender not in converged_clients:
                    converged_clients[message.sender] = iteration
                    removing_clients.add(message.sender)
                    
            end_call_time = datetime.now()
            server_logic_time = end_call_time - start_call_time
            simulated_time += server_logic_time #Tổng thời gian cho đến bước này
            
            active_clients_list -= removing_clients
            
            if len(active_clients_list) < 2:
                self.print
            
            
    def print_convergences(self, converged):
        for client_name in self.
            if client_name in converged:
                print('Client {} converged on iteration {}'.format(client_name, converged[client_name]))
            if client_name not in converged:
                print('Client {} never converged'.format(client_name))

    def final_statistics(self):
        """
        USED FOR RESEARCH PURPOSES.
        """
        # for research purposes
        client_accs = []
        fed_acc = []
        for client_name, client_instance in self.directory.clients.items():
            fed_acc.append(list(client_instance.federated_accuracy.values()))
            client_accs.append(list(client_instance.personal_accuracy.values()))

        
        print('Federated accuracies are {}'.format(dict(zip(self.directory.clients, fed_acc))))