
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
serverObjects = {server_name: Server(server_name=server_name, \
                        active_clients_list=active_clients_list) \
                        for server_name in active_servers_list}

clientObjects = {client_name: Client(client_name, [], [], \
                        active_clients_list = active_clients_list) \