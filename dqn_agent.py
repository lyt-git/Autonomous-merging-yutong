import os
import time
import math
import random
import sys
import csv

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
#from itertools import count
#from PIL import Image

import traci
import sumolib
from sumolib import checkBinary

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device("cpu")



'''Setting parameters used globally'''

'Training parameters'
batch_size = 128
gamma = 0.999

'Exloration rate (learning rate)'
eps_1 = 0.9
eps_end = 0.05
decay = 100
eps_refreshed_times = 0

'update target net every 5 episodes' 
update = 5


'the dimension of input states'
state_dim = 2

'the dimension of the set of outputs'
action_dim = 2 








'''Set up path to SUMO'''

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
    
sys.path.append(os.path.join('c:', os.sep, 'whatever', 'path', 'to', 'sumo', 'tools'))

if_show_gui = True

if not if_show_gui:
    sumoBinary = checkBinary('sumo')
else:
    sumoBinary = checkBinary('sumo-gui')
cfgfile = "E:\\SUMO\\FYPcode\\myhighway\\myhw.sumocfg"








 



Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))








class ReplayBuffer(object):

    def __init__(self, length):
        self.memory = deque([],maxlen=length)

    def append(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return  random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)




    


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_size)
        )

    def forward(self, x):
        return self.fc(x)
    
policy = DQN(state_dim,action_dim).to(device)
target = DQN(state_dim,action_dim).to(device)
target.load_state_dict(policy.state_dict())
target.eval()

optimizer = optim.RMSprop(policy.parameters())
memory = ReplayBuffer(10000)







def act_choose(state):

    global eps_refreshed_times
    eps_now = eps_end + (eps_1-eps_end)*math.exp(-1.*eps_refreshed_times/decay)
    eps_refreshed_times += 1
    
    #print(eps_now)
    if np.random.rand() <= eps_now:
        return random.randrange(action_dim)
    else:
        state = torch.from_numpy(state)
        state = Variable(state).float().cpu()
        Q_val = policy(state)
        _, act_choice = torch.max(Q_val, 0)
        return int(act_choice)





    
def train():
    if len(memory) < batch_size:
        return 
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    not_end_index = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    next_state_index = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    #print(action_batch)
    reward_batch = torch.cat(batch.reward)

    '''Compute Q value'''
    Qt = policy(state_batch).gather(1, action_batch)
    #print("Q",Qt)
 
    Vt+1 = torch.zeros(batch_size, device=device)
    Vt+1[not_end_index] = target(next_state_index).max(1)[0].detach()

    Qt+1 = (Vt+1 * gamma) + reward_batch
    #print("Qexp",Qt+1.unsqueeze(1))

    criterion = nn.SmoothL1Loss()
    loss = criterion(Qt, Qt+1.unsqueeze(1))

    optimizer.zero_grad()
loss.backward()

    for param in policy.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()














def addego():
    #traci.route.add("ego_route", ["b", "c"])
    traci.vehicle.add("ego_car", "ego_route", typeID="ego_car")
    return "ego_car"

def action_space(ID, act):
    if act == 0:
        traci.vehicle.setSpeed(ID, 0)
    elif act == 1:
        traci.vehicle.setSpeed(ID, 20)
    else:
        traci.vehicle.setSpeed(ID, 0)

def observation_space():
    ID_list = traci.vehicle.getIDList()
    ego_pos = 0
    dist = [1]
    dist_to_merge = [0]
    dist_front = [100]
    dist_back = [-20]
    for i in ID_list:
        if i == "ego_car":
            ego_pos = traci.vehicle.getPosition(i)[0]
        else:
            d = traci.vehicle.getPosition(i)[0]
            dist.append(d)
            if d < 35:
                dist_to_merge.append(d)
            if d > ego_pos:
                dist_front.append(d)
            elif d <= ego_pos:
                dist_back.append(d)
            
    closest_to_merge = np.max(dist_to_merge)
    closest_front = np.min(dist_front)
    closest_back = np.max(dist_back)
    closest_front = closest_front-ego_pos
    closest_back = ego_pos-closest_back
    
    stateA = 35 - closest_to_merge

    stateB = ego_pos

    state = np.ascontiguousarray([stateA, stateB], dtype=np.float32)
    state = torch.from_numpy(state)
    #print(state)        
    return state
                    








'''Main training loop'''


num_episodes = 300
success = 0
r_array = []
for i_iter in range(num_episodes):
    
    traci.start([sumoBinary, "-c" , cfgfile])
    traci.route.add("ego_route", ["b", "c"])
    iterations = 256
    ego_start = random.randint(50,100)
    run_state = (0,0)
    state_array = np.asarray(run_state)
    done = 0
    reward_tot = 0
    
    for step in range(0,iterations):

        '''Ego car init'''
        if step == ego_start:
            egoID = addego()
            traci.vehicle.setSpeedMode(egoID, 0)
            traci.vehicle.setSpeed(egoID, 0)

        '''Select an action'''
        if step > ego_start:
            if "ego_car" in traci.vehicle.getIDList():
                action_choice = act_choose(state_array)
                action_space(egoID, action_choice)
            else:
                pass

        '''Step count'''
        traci.simulationStep()

        '''Gain rewards'''
        warn_list = traci.simulation.getCollidingVehiclesIDList();
        if warn_list!=():
            #print("\nCollision")
            reward = -100
            done = 1
        else:
            reward = -0.1
            done += 1/iterations

        if step > ego_start:
            if "ego_car" in traci.vehicle.getIDList():
                pass
            else:
                #print("\nEgo car reached destination")
                reward = 100
                done = 1
                success += 1

        if step > ego_start:
            reward = torch.tensor([reward], device=device)

            '''Observe new space'''
            next_state_array = observation_space()
            #print(next_state_array)
            next_state_array = np.asarray(next_state_array)

            '''Push the memory into replay buffer'''
            'Form tensor'
            state_memo = torch.tensor([state_array], device=device)
            action_memo = torch.tensor([[action_choice]], device=device)
            next_state_memo = torch.tensor([next_state_array], device=device)
            'Push memory'
            memory.append(state_memo, action_memo, next_state_memo, reward)

            '''Next state'''
            state_array = next_state_array
            #print(state_array)

            '''Optim'''
            if i_iter < 100:
                train()

            reward_tot += reward

        if done == 1:
            break
            traci.close()

    if i_iter % update == 0:
        target.load_state_dict(policy.state_dict())

    print(i_iter,'\n')
    print(reward_tot)
    r_array.append(reward_tot)
    
    traci.close()

print(r_array)


