
import numpy as np
import math
import sys
import tqdm

import pandas as pd


#from 1RL_ICUProject_Finalstatespace.ipynb


# state_space_df = pd.read_csv('statespace.csv',header=None)
# state_space_df.head()


#state space plus dopamine values
col_names = ['F0','F1', 'F2', 'F3', 'F4','F5', 'F6', 'F7', 'F8','F9','Dopamine']

# state_space_dop = pd.read_csv('statespace_Dop.csv', names=col_names)  #representation was created only with Dense layers
state_space_dop = pd.read_csv('statespace_LG_Dop.csv', names=col_names)  #representation was created with LSTM,GRU and Dense layers
state_space_dop.head()


col_nm=['VentMode', 'heartrate', 'meanpressure', 'sysPressure', 'diasPressure',
       'artheartrate', 'osat', 'FiO2', 'resprate', 'TidalVol', 'PIP',
       'AirwayP', 'PEEP', 'ALB', 'ALT', 'AST', 'BICARBART', 'BILITOTAL', 'BUN',
       'CA', 'CL', 'CO2', 'CR', 'FIB', 'GLU', 'HCT', 'HGB', 'IONCA', 'K',
       'MCH', 'MCHC', 'MCV', 'MG', 'NA', 'NH3', 'O2SATART', 'PCO2ART', 'PHART',
       'PHOS', 'PLT', 'PO2ART', 'PT', 'PT1', 'PT2', 'PTT', 'PTT1', 'PTT2',
       'RBC', 'TP', 'TRIG', 'WBC', 'Dopamine', 'Fentanyl', 'DopamineAction']
icu_arr_df = pd.read_csv('ICU_dataA1.csv', names=col_nm, header=None)  



#conditionally assign patientstate (1: ok, -1 :deteriorate) - 
icu_arr_df['patientstate'] = 1  #initialise
icu_arr_df.loc[((icu_arr_df['heartrate'] >=0.7) | (icu_arr_df['osat'] <=0.6)  | (icu_arr_df['resprate'] <0.75)), 'patientstate'] = 0    #deteriorate
icu_arr_df



encoded_val_data_part = icu_arr_df.iloc[1001:2000]  ##statespace data was created from this part of icu_arr_df
encoded_val_data_part_patientstate= encoded_val_data_part['patientstate']
# encoded_val_data_part_patientstate.columns=['patientstate']
encoded_val_data_part_patientstate



encoded_val_data_part_patientstate_df = pd.DataFrame(encoded_val_data_part_patientstate)
encoded_val_data_part_patientstate_df

#Add patient state to state_space_dop

state_space_dop_patientstate = state_space_dop.copy()


state_space_dop_patientstate



# extracted_col = encoded_val_data_part_patientstate_df['patientstate']
# icu_arr_df_patientstate['patientstate']= icu_arr_df_patientstate.join(extracted_col)
state_space_dop_patientstate.insert(10, "patientstate", encoded_val_data_part_patientstate_df.iloc[:,0].values)
state_space_dop_patientstate


state_space_dop_patientstate.to_csv('state_space_dop_patientstate.csv')


# Synthetic - add patient state

# In[120]:


col_nm=['VentMode', 'heartrate', 'meanpressure', 'sysPressure', 'diasPressure',
       'artheartrate', 'osat', 'FiO2', 'resprate', 'TidalVol', 'PIP',
       'AirwayP', 'PEEP', 'ALB', 'ALT', 'AST', 'BICARBART', 'BILITOTAL', 'BUN',
       'CA', 'CL', 'CO2', 'CR', 'FIB', 'GLU', 'HCT', 'HGB', 'IONCA', 'K',
       'MCH', 'MCHC', 'MCV', 'MG', 'NA', 'NH3', 'O2SATART', 'PCO2ART', 'PHART',
       'PHOS', 'PLT', 'PO2ART', 'PT', 'PT1', 'PT2', 'PTT', 'PTT1', 'PTT2',
       'RBC', 'TP', 'TRIG', 'WBC', 'Dopamine', 'Fentanyl', 'DopamineAction']
syn_arr_df = pd.read_csv('synth_dataA1.csv', names=col_nm, header=None)  



#conditionally assign patientstate (1: ok, -1 :deteriorate) - 
syn_arr_df['patientstate'] = 1  #initialise
syn_arr_df.loc[((syn_arr_df['heartrate'] >=0.7) | (syn_arr_df['osat'] <=0.6)  | (syn_arr_df['resprate'] <0.75)), 'patientstate'] = 0    #deteriorate
syn_arr_df


#Get synthetic data
#state space plus dopamine values
col_names = ['F0','F1', 'F2', 'F3', 'F4','F5', 'F6', 'F7', 'F8','F9','Dopamine']
state_space_Syn_dop = pd.read_csv('statespace_Syn_Dop.csv', names=col_names)
state_space_Syn_dop.head()





# dfLMF = pd.read_csv('dfLMF.csv')     #synthetic data was created from dfLMF

# dfLMF.drop('Unnamed: 0',axis=1)


# #conditionally assign patientstate (1: ok, -1 :deteriorate) - 
# dfLMF['patientstate'] = 1  #initialise

# dfLMF.loc[((dfLMF['heartrate'] >=160) | (dfLMF['osat'] <97)  | (dfLMF['resprate'] >16)), 'patientstate'] = 0    #deteriorate
# dfLMF.drop('Unnamed: 0',axis=1)

encoded_val_data_part_syn_patientstate.unique()



encoded_val_data_part_syn = dfLMF.iloc[1001:2000]  ##statespace data was created from this part of dfLMF
encoded_val_data_part_syn_patientstate= encoded_val_data_part_syn['patientstate']
# encoded_val_data_part_patientstate.columns=['patientstate']
encoded_val_data_part_syn_patientstate



#add patientstate column to syn_space


state_space_Syn_dop_patientstate = state_space_Syn_dop.copy()


state_space_Syn_dop_patientstate.insert(10, "patientstate", encoded_val_data_part_patientstate_df.iloc[:,0].values)
state_space_Syn_dop_patientstate


# In[144]:


state_space_Syn_dop_patientstate.to_csv('state_space_Syn_dop_patientstate.csv')

training_df = pd.read_csv('training_data_withreward.csv')
training_df_reward = training_df.drop("Unnamed: 0", axis='columns')
training_df_reward.head()



#conditionally assign patientstate (1: ok, -1 :deteriorate) - not used?
training_df_reward['patientstate'] = 1  #initialise
training_df_reward.loc[((training_df_reward['heartrate'] >=0.7) | (training_df_reward['osat'] <=0.6)  | (training_df_reward['resprate'] <0.75)), 'patientstate'] = 0    #deteriorate



# FINAL TRIAL - DO from HERE


import time
import copy
import numpy as np
import pandas as pd
import chainer
import chainer.functions as F
import chainer.links as L
from plotly import tools
from plotly.graph_objs import *
from plotly.offline import init_notebook_mode, iplot, iplot_mpl


#https://www.kaggle.com/code/itoeiji/deep-reinforcement-learning-on-stock-data
class Environment1:
    
    def __init__(self, data, history_t=90):
        self.data = data
        self.history_t = history_t
        self.reset()
        
    def reset(self):
        self.t = 0
        self.done = False
        self.pr = 0
        self.positions = []
        self.position_value = 0
        self.history = [0 for _ in range(self.history_t)]
        return [self.position_value] + self.history # obs
    
    def step(self, act):
        reward = 0
        pr = 0
        # act = 0: same dopamine, 1: decrease dopamine, 2: increase dopamine
        # act = 0: 
        if act == 1: #decrease
            self.positions.append(self.data.iloc[self.t, :]['Dopamine'])
            reward += pr
            self.pr += pr
        elif act == 2: # increase
            if len(self.positions) == 0:
                reward = -1
            else:
                pr = 0
                for p in self.positions:
                    pr += (self.data.iloc[self.t, :]['Dopamine'] - p)
                reward -= pr
                self.pr -= pr
                self.positions = []
        
        # set next time
        self.t += 1
        self.position_value = 0
        for p in self.positions:
            self.position_value += (self.data.iloc[self.t, :]['Dopamine'] - p)
        self.history.pop(0)
        self.history.append(self.data.iloc[self.t, :]['Dopamine'] - self.data.iloc[(self.t-1), :]['Dopamine'])
        
        # clipping reward
        if reward > 0:
            reward = 1
        elif reward < 0:
            reward = -1
        
        return [self.position_value] + self.history, reward, self.done # obs, reward, done


for label in labels:

    if not label in (0, 1):
        raise Exception('Labels must satisfy label == 0 or label == 1.')




#https://www.kaggle.com/code/itoeiji/deep-reinforcement-learning-on-stock-data
class Environment1:             # TRIAL ON AUG 25
    
    def __init__(self, data, history_t=90):
        self.data = data
        self.history_t = history_t
#         n_actions = 2
#         self.action_space = spaces.Discrete(n=n_actions)
        self.reset()
        
    def reset(self):
        self.t = 0
        self.done = False
        self.pr = 0
        self.positions = []
        self.position_value = 0
        self.history = [0 for _ in range(self.history_t)]
        return [self.position_value] + self.history # obs
    
    def step(self, act):
        reward = 0
        pr = 0
        # act = 0: same dopamine, 1: decrease dopamine, 2: increase dopamine
        # act = 0: 
        if act == 1: #decrease
            self.positions.append(self.data.iloc[self.t, :]['Dopamine'])
            reward += pr
            self.pr += pr
            if self.data.iloc[self.t, :]['patientstate'] == 1: #health not deteriorated
                reward += 1
            else:
                reward += 0
        elif act == 2: # increase
            if len(self.positions) == 0:
                reward = -1
            else:
                pr = 0
                for p in self.positions:
                    pr += (self.data.iloc[self.t, :]['Dopamine'] - p)
                reward -= pr
                self.pr -= pr
                self.positions = []
                if self.data.iloc[self.t, :]['patientstate'] == 1: #health not deteriorated
                    reward += 0
                else:
                    reward += -1   #penalised - since eventhough dopamine is increased, still health declines

        # set next time
        self.t += 1
        self.position_value = 0
        for p in self.positions:
            self.position_value += (self.data.iloc[self.t, :]['Dopamine'] - p)
        self.history.pop(0)
        self.history.append(self.data.iloc[self.t, :]['Dopamine'] - self.data.iloc[(self.t-1), :]['Dopamine'])
        
        # clipping reward
        if reward > 0:
            reward = 1
        elif reward < 0:
            reward = -1
        
        return [self.position_value] + self.history, reward, self.done # obs, reward, done



env = Environment1(training_df_reward)

#https://www.kaggle.com/code/itoeiji/deep-reinforcement-learning-on-stock-data

# DQN - ORIGINAL

def train_dqn(env):

    class Q_Network(chainer.Chain):

        def __init__(self, input_size, hidden_size, output_size):
            super(Q_Network, self).__init__(
                fc1 = L.Linear(input_size, hidden_size),
                fc2 = L.Linear(hidden_size, hidden_size),
                fc3 = L.Linear(hidden_size, output_size)
            )
            
            
        #########Aug 26 #https://github.com/5misakamikoto5/dqn/blob/master/agent_dir/agent_dqn.py
        def forward(self, inputs):
            print("forward",inputs)
            h = F.relu(self.fc1(inputs))         #activation function  is relu
            h = F.relu(self.fc2(h))
            return self.fc3(h)
        
        ########Aug 26

        def __call__(self, x):
            h = F.relu(self.fc1(x))         #activation function  is relu
            h = F.relu(self.fc2(h))
            y = self.fc3(h)
            return y

        def reset(self):
            self.zerograds()

    Q = Q_Network(input_size=env.history_t+1, hidden_size=100, output_size=3)
    Q_ast = copy.deepcopy(Q)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(Q)

#     epoch_num = 100
#     epoch_num = 200
    epoch_num = 10 
#     epoch_num = 50
    step_max = len(env.data)-1
    memory_size = 200
    batch_size = 20
    epsilon = 1.0
    epsilon_decrease = 1e-3
    epsilon_min = 0.1
    start_reduce_epsilon = 200
    train_freq = 10
    update_q_freq = 20
    gamma = 0.97
    show_log_freq = 5

    memory = []
    total_step = 0
    total_rewards = []
    total_losses = []
    total_maes1 = [] ##
    total_accuracies1 = [] ##

    start = time.time()
    for epoch in range(epoch_num):

        pobs = env.reset()
        step = 0
        done = False
        total_reward = 0
        total_loss = 0
        total_mae1 = 0 ## 
        total_acc1=0 ##

        while not done and step < step_max:

            # select act
            pact = np.random.randint(2)         # was 3
            if np.random.rand() > epsilon:
                pact = Q(np.array(pobs, dtype=np.float32).reshape(1, -1))
                pact = np.argmax(pact.data)

            # act
            obs, reward, done = env.step(pact)

            # add memory
            memory.append((pobs, pact, reward, obs, done))
            if len(memory) > memory_size:
                memory.pop(0)

            # train or update q
            if len(memory) == memory_size:
                if total_step % train_freq == 0:
                    shuffled_memory = np.random.permutation(memory)
                    memory_idx = range(len(shuffled_memory))
                    for i in memory_idx[::batch_size]:
                        batch = np.array(shuffled_memory[i:i+batch_size])
                        b_pobs = np.array(batch[:, 0].tolist(), dtype=np.float32).reshape(batch_size, -1)
                        b_pact = np.array(batch[:, 1].tolist(), dtype=np.int32)
                        b_reward = np.array(batch[:, 2].tolist(), dtype=np.int32)
                        b_obs = np.array(batch[:, 3].tolist(), dtype=np.float32).reshape(batch_size, -1)
                        b_done = np.array(batch[:, 4].tolist(), dtype=np.bool)

                        q = Q(b_pobs)
                        maxq = np.max(Q_ast(b_obs).data, axis=1)
                        target = copy.deepcopy(q.data)
                        for j in range(batch_size):
                            target[j, b_pact[j]] = b_reward[j]+gamma*maxq[j]*(not b_done[j])
                        Q.reset()
                        loss = F.mean_squared_error(q, target)
                        mae1 =  F.mean_absolute_error(q, target)  #https://docs.chainer.org/en/stable/reference/generated/chainer.functions.mean_absolute_error.html
#                         acc1 = F.accuracy(q, target).array  #https://docs.chainer.org/en/stable/reference/functions.html#evaluation-functions
                        total_loss += loss.data
                        total_mae1 += mae1.data  ##
#                         total_acc1 += acc1.data  ##
                        loss.backward()
                        optimizer.update()

                if total_step % update_q_freq == 0:
                    Q_ast = copy.deepcopy(Q)

            # epsilon
            if epsilon > epsilon_min and total_step > start_reduce_epsilon:
                epsilon -= epsilon_decrease

            # next step
            total_reward += reward
            pobs = obs
            step += 1
            total_step += 1

        total_rewards.append(total_reward)
        total_losses.append(total_loss)
        total_maes1.append(total_mae1)      ##
#         total_accuracies1.append(total_acc1)      ##

        if (epoch+1) % show_log_freq == 0:
            log_reward = sum(total_rewards[((epoch+1)-show_log_freq):])/show_log_freq
            log_loss = sum(total_losses[((epoch+1)-show_log_freq):])/show_log_freq
            log_mae = sum(total_maes1[((epoch+1)-show_log_freq):])/show_log_freq #####
#             log_acc = sum(total_accuracies1[((epoch+1)-show_log_freq):])/show_log_freq #####
            elapsed_time = time.time()-start
            print('\t'.join(map(str, [epoch+1, epsilon, total_step, log_mae, log_reward, log_loss, elapsed_time])))
            start = time.time()

    return Q, total_maes1, total_losses, total_rewards  ##


#https://www.kaggle.com/code/itoeiji/deep-reinforcement-learning-on-stock-data

# Q learning - Sep 3

def train_Q(env):


    total_reward = 0
    epoch_num = 10

    penalties, rewards = 0
    done = False
    while not done:
        pact = np.random.randint(2)
        obs, reward, done = env.step(pact)
         # next step
        total_reward += reward


    return total_reward



q_table = np.zeros([env.observation_space.n, env.action_space.n])


#Q-learning Algorithm
#https://github.com/ronanmmurphy/Q-Learning-Algorithm/blob/main/Q-Learning%20Algorithm.py
def Q_Learning(env):
    x = 0
    episodes=100
    total_reward = 0
    epoch_num = 10

    penalties, rewards = 0
    #iterate through best path for each episode
    while(x < episodes):
        pact = np.random.randint(2)
            #get current rewrard and add to array for plot
        obs, reward, done = env.step(pact)
        total_reward += reward

            #get state, assign reward to each Q_value in state
#             i,j = self.State.state
#             for a in self.actions:
        new_Q[a] = round(reward,3)
        env.reset()
           

        #copy new Q values to Q table
        Q = new_Q.copy()
    #print final Q table output
    print(Q)




episodes = 100
Q_Learning(Environment1(training_df_reward))



#https://www.kaggle.com/code/itoeiji/deep-reinforcement-learning-on-stock-data
class Environment2:             # TRIAL ON sep3
    
    def __init__(self, data, history_t=90):
        self.data = data
        self.observation_space=data
        self.action_space=2
        self.history_t = history_t
#         n_actions = 2
#         self.action_space = spaces.Discrete(n=n_actions)
        self.reset()
        
    def reset(self):
        self.t = 0
        self.done = False
        self.pr = 0
        self.positions = []
        self.position_value = 0
        self.history = [0 for _ in range(self.history_t)]
        return [self.position_value] + self.history # obs
    
    def step(self, act):
        reward = 0
        pr = 0
        # act = 0: same dopamine, 1: decrease dopamine, 2: increase dopamine
        # act = 0: 
        if act == 1: #decrease
            self.positions.append(self.data.iloc[self.t, :]['Dopamine'])
            reward += pr
            self.pr += pr
            if self.data.iloc[self.t, :]['patientstate'] == 1: #health not deteriorated
                reward += 1
            else:
                reward += 0
        elif act == 2: # increase
            if len(self.positions) == 0:
                reward = -1
            else:
                pr = 0
                for p in self.positions:
                    pr += (self.data.iloc[self.t, :]['Dopamine'] - p)
                reward -= pr
                self.pr -= pr
                self.positions = []
                if self.data.iloc[self.t, :]['patientstate'] == 1: #health not deteriorated
                    reward += 0
                else:
                    reward += -1   #penalised - since eventhough dopamine is increased, still health declines

        # set next time
        self.t += 1
        self.position_value = 0
        for p in self.positions:
            self.position_value += (self.data.iloc[self.t, :]['Dopamine'] - p)
        self.history.pop(0)
        self.history.append(self.data.iloc[self.t, :]['Dopamine'] - self.data.iloc[(self.t-1), :]['Dopamine'])
        
        # clipping reward
        if reward > 0:
            reward = 1
        elif reward < 0:
            reward = -1
        
        return [self.position_value] + self.history, reward, self.done # obs, reward, done


env2 = Environment2(training_df_reward)



#env = Environment1(training_df_reward)
import numpy as np


from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory



from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D,Reshape 
# from keras.layers import Activation, Dense, Reshape 
nb_actions = 2
obs_sp = env2.observation_space
# print(obs_sp)
# trdata = np.expand_dims(obs_sp, axis=-1)
print(env2.observation_space.shape)
# print(trdata.shape)

#https://pythonprogramming.net/deep-q-learning-dqn-reinforcement-learning-python-tutorial/

model = Sequential()
model.add(Reshape((1000, 55,1),input_shape=env2.observation_space.shape))
# model.add(Conv2D(256, (3, 3)))
model.add(Dense(16))
# model.add(Conv2D(256, (3, 3), input_shape=env2.observation_space.shape)) 
model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))

model.add(Dense(16))
# model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))

model.add(Dense(env2.action_space, activation='relu'))  
model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])


print(model.summary())
      

policy = EpsGreedyQPolicy()

memory = SequentialMemory(limit=50000, window_length=55)
dqn = DQNAgent(model=model, nb_actions=2, memory=memory, policy=policy)
# dqn = DQNAgent(model=model, nb_actions=2, memory=memory, nb_steps_warmup=10,target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

dqn.fit(env2, nb_steps=1000, visualize=True, verbose=2)


#https://tomroth.com.au/dqn-simple/

## Building the nnet that approximates q 
n_actions = 2  # dim of output layer 
input_dim = env2.observation_space.shape[0]  # dim of input layer 
model = Sequential()
model.add(Dense(64, input_dim = input_dim , activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(n_actions, activation = 'linear'))
model.compile(optimizer=Adam(), loss = 'mse')

n_episodes = 1000
gamma = 0.99
epsilon = 1
minibatch_size = 32
r_sums = []  # stores rewards of each epsiode 
replay_memory = [] # replay memory holds s, a, r, s'
mem_max_size = 100000
#https://tomroth.com.au/dqn-simple/


#https://www.kaggle.com/code/itoeiji/deep-reinforcement-learning-on-stock-data

# DQN  - AUG 27

def train_dqn(env):

    class Q_Network(chainer.Chain):

        def __init__(self, input_size, hidden_size, output_size):
            super(Q_Network, self).__init__(
                fc1 = L.Linear(input_size, hidden_size),
                fc2 = L.Linear(hidden_size, hidden_size),
                fc3 = L.Linear(hidden_size, output_size)
            )
            
        


        def __call__(self, x):
            h = F.relu(self.fc1(x))         #activation function  is relu
            h = F.relu(self.fc2(h))
            y = self.fc3(h)
            return y

        def reset(self):
            self.zerograds()

    Q = Q_Network(input_size=env.history_t+1, hidden_size=100, output_size=3)
    Q_ast = copy.deepcopy(Q)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(Q)

#     epoch_num = 100
#     epoch_num = 200
    epoch_num = 1000 
#     epoch_num = 50
    step_max = len(env.data)-1
    memory_size = 200
    batch_size = 20
    epsilon = 1.0
    epsilon_decrease = 1e-3
    epsilon_min = 0.1
    start_reduce_epsilon = 200
    train_freq = 10
    update_q_freq = 20
    gamma = 0.97
    show_log_freq = 5

    memory = []
    total_step = 0
    total_rewards = []
    total_losses = []
    total_maes1 = [] ##
    total_accuracies1 = [] ##

    start = time.time()
    for epoch in range(epoch_num):

        pobs = env.reset()
        step = 0
        done = False
        total_reward = 0
        total_loss = 0
        total_mae1 = 0 ## 
        total_acc1=0 ##

        while not done and step < step_max:

            # select act
#             pact = np.random.randint(0,2)         #action - 0 or 1
            pact = np.random.randint(2) 
#             if np.random.rand() > epsilon:
            if pact > epsilon:
                pact = Q(np.array(pobs, dtype=np.float32).reshape(1, -1))
                pact = np.argmax(pact.data)
#                 print("epsilon: ", epsilon)
#                 print("inside pact: ", pact)
                
#             print("pact: ", pact)
            # act
            obs, reward, done = env.step(pact)

            # add memory
            memory.append((pobs, pact, reward, obs, done))
            if len(memory) > memory_size:
                memory.pop(0)
                



            # train or update q
            if len(memory) == memory_size:
                if total_step % train_freq == 0:
                    shuffled_memory = np.random.permutation(memory)
                    memory_idx = range(len(shuffled_memory))
                    ####
#                     print(memory_idx)
                    ####
                    for i in memory_idx[::batch_size]:
                        batch = np.array(shuffled_memory[i:i+batch_size])
                        b_pobs = np.array(batch[:, 0].tolist(), dtype=np.float32).reshape(batch_size, -1)
                        b_pact = np.array(batch[:, 1].tolist(), dtype=np.int32)
                        b_reward = np.array(batch[:, 2].tolist(), dtype=np.int32)
                        b_obs = np.array(batch[:, 3].tolist(), dtype=np.float32).reshape(batch_size, -1)
                        b_done = np.array(batch[:, 4].tolist(), dtype=np.bool)

                        q = Q(b_pobs)
                        maxq = np.max(Q_ast(b_obs).data, axis=1)
                        target = copy.deepcopy(q.data)
                        for j in range(batch_size):
                            target[j, b_pact[j]] = b_reward[j]+gamma*maxq[j]*(not b_done[j])
                        Q.reset()
                        loss = F.mean_squared_error(q, target)
                        mae1 =  F.mean_absolute_error(q, target)  #https://docs.chainer.org/en/stable/reference/generated/chainer.functions.mean_absolute_error.html
#                         acc1 = F.accuracy(q, target).array  #https://docs.chainer.org/en/stable/reference/functions.html#evaluation-functions
                        total_loss += loss.data
                        total_mae1 += mae1.data  ##
#                         total_acc1 += acc1.data  ##
                        loss.backward()
                        optimizer.update()
        
                       

                if total_step % update_q_freq == 0:
                    Q_ast = copy.deepcopy(Q)

            # epsilon
            if epsilon > epsilon_min and total_step > start_reduce_epsilon:
                epsilon -= epsilon_decrease

            # next step
            total_reward += reward
            pobs = obs
            step += 1
            total_step += 1

        total_rewards.append(total_reward)
        total_losses.append(total_loss)
        total_maes1.append(total_mae1)      ##
#         total_accuracies1.append(total_acc1)      ##

        if (epoch+1) % show_log_freq == 0:
            log_reward = sum(total_rewards[((epoch+1)-show_log_freq):])/show_log_freq
            log_loss = sum(total_losses[((epoch+1)-show_log_freq):])/show_log_freq
            log_mae = sum(total_maes1[((epoch+1)-show_log_freq):])/show_log_freq #####
#             log_acc = sum(total_accuracies1[((epoch+1)-show_log_freq):])/show_log_freq #####
            elapsed_time = time.time()-start
            print('\t'.join(map(str, [epoch+1, epsilon, total_step, log_mae, log_reward, log_loss, elapsed_time])))
            start = time.time()
 


    return b_pact, maxq, Q, total_maes1, total_losses, total_rewards  ##



#https://www.kaggle.com/code/itoeiji/deep-reinforcement-learning-on-stock-data
a10_actions_dqn_O, a10_maxq_dqn_O, a10_Q_dqn_O, a10_total_maes1_dqn_O, a10_total_losses_dqn_O, a10_total_rewards_dqn_O = train_dqn(Environment1(training_df_reward))



#https://www.kaggle.com/code/itoeiji/deep-reinforcement-learning-on-stock-data
actions_dqn_O, maxq_dqn_O, Q_dqn_O, total_maes1_dqn_O, total_losses_dqn_O, total_rewards_dqn_O = train_dqn(Environment1(training_df_reward))


Q_dqn_O, total_maes1_dqn_O, total_losses_dqn_O, total_rewards_dqn_O = train_dqn(Environment1(training_df_reward))



actions_dqn_S, maxq_dqn_S, Q_dqn_S, total_maes1_dqn_S, total_losses_dqn_S, total_rewards_dqn_S = train_dqn(Environment1(state_space_dop_patientstate))


actions_dqn_S_Syn, maxq_dqn_S_Syn, Q_dqn_S_Syn, total_maes1_dqn_S_Syn, total_losses_dqn_S_Syn, total_rewards_dqn_S_Syn = train_dqn(Environment1(state_space_Syn_dop_patientstate))

#Run DDQN first

actions_ddqn_O, maxq_ddqn_O,Q_ddqn_O, total_maes1_ddqn_O, total_losses_ddqn_O, total_rewards_ddqn_O = train_ddqn(Environment1(training_df_reward))


actions_ddqn_S, maxq_ddqn_S, Q_ddqn_S, total_maes1_ddqn_S, total_losses_ddqn_S, total_rewards_ddqn_S = train_ddqn(Environment1(state_space_dop_patientstate))


#synthetic
actions_ddqn_S_Syn, maxq_ddqn_S_Syn, Q_ddqn_S_Syn, total_maes1_ddqn_S_Syn, total_losses_ddqn_S_Syn, total_rewards_ddqn_S_Syn = train_ddqn(Environment1(state_space_Syn_dop_patientstate))

#Run DDDQN first
#DDDQN
actions_dddqn_O, maxq_dddqn_O, Q_dddqn_O, total_maes1_dddqn_O, total_losses_dddqn_O, total_rewards_dddqn_O = train_dddqn(Environment1(training_df_reward))


#DDDQN - state_space_temp
actions_dddqn_S, maxq_dddqn_S, Q_dddqn_S, total_maes1_dddqn_S, total_losses_dddqn_S, total_rewards_dddqn_S = train_dddqn(Environment1(state_space_dop_patientstate))


#DDDQN - state_space_temp - synthetic
actions_dddqn_S_Syn, maxq_dddqn_S_Syn, Q_dddqn_S_Syn, total_maes1_dddqn_S_Syn,total_losses_dddqn_S_Syn, total_rewards_dddqn_S_Syn = train_dddqn(Environment1(state_space_Syn_dop_patientstate))


#Run DDDQN first
#DDDQN
actions_dddqn_O, maxq_dddqn_O, Q_dddqn_O, total_maes1_dddqn_O, total_losses_dddqn_O, total_rewards_dddqn_O = train_dddqn(Environment1(training_df_reward))


#DDDQN - state_space_temp
actions_dddqn_S, maxq_dddqn_S, Q_dddqn_S, total_maes1_dddqn_S, total_losses_dddqn_S, total_rewards_dddqn_S = train_dddqn(Environment1(state_space_dop_patientstate))


#DDDQN - state_space_temp - synthetic
actions_dddqn_S_Syn, maxq_dddqn_S_Syn, Q_dddqn_S_Syn, total_maes1_dddqn_S_Syn,total_losses_dddqn_S_Syn, total_rewards_dddqn_S_Syn = train_dddqn(Environment1(state_space_Syn_dop_patientstate))





# START PLOTTING EVERYTHING



#DQN - original dataset
print ("DQN - Baseline")
plot_loss_reward(total_losses_dqn_O, total_rewards_dqn_O)



#DQN - State Repr
print ("DQN - State Repr")
plot_loss_reward(total_losses_dqn_S, total_rewards_dqn_S)




#DQN - Synthetic
print ("DQN - Synthetic")
plot_loss_reward(total_losses_dqn_S_Syn, total_rewards_dqn_S_Syn)




#DDQN - original dataset
print ("DDQN")
plot_loss_reward(total_losses_ddqn_O, total_rewards_ddqn_O)



#DDQN - State Repr
print ("DDQN - State Repr")
plot_loss_reward(total_losses_ddqn_S, total_rewards_ddqn_S)



#DDQN - Synthetic
print ("DDQN - Synthetic")
plot_loss_reward(total_losses_ddqn_S_Syn, total_rewards_ddqn_S_Syn)




#DDDQN - original dataset
print ("DDDQN")
plot_loss_reward(total_losses_dddqn_O, total_rewards_dddqn_O)



#DDDQN - State Repr
print ("DDDQN - State Repr")
plot_loss_reward(total_losses_dddqn_S, total_rewards_dddqn_S)




#DDDQN - Synthetic
print ("DDDQN - Synthetic")
plot_loss_reward(total_losses_dddqn_S_Syn, total_rewards_dddqn_S_Syn)


#DQN - original dataset
plot_loss_reward(total_losses_dqn_O, total_rewards_dqn_O)


actions__dqn_O, maxq_dqn_O, Q_dqn_O, total_maes1_dqn_O, total_losses_dqn_O, total_rewards_dqn_O = train_dqn(Environment1(training_df_reward))

Q_dqn_O, total_maes1_dqn_O, total_losses_dqn_O, total_rewards_dqn_O = train_dqn(Environment1(training_df_reward))



#DQN - original dataset
plot_loss_reward(total_losses_dqn_O, total_rewards_dqn_O)


#https://www.kaggle.com/code/itoeiji/deep-reinforcement-learning-on-stock-data
def plot_loss_reward(total_losses, total_rewards):

    figure = tools.make_subplots(rows=1, cols=2, subplot_titles=('loss', 'reward'), print_grid=False)
    figure.append_trace(Scatter(y=total_losses, mode='lines', line=dict(color='blue')), 1, 1)
    figure.append_trace(Scatter(y=total_rewards, mode='lines', line=dict(color='green')), 1, 2)
    figure['layout']['xaxis1'].update(title='episode')
    figure['layout']['xaxis2'].update(title='episode')
    figure['layout'].update(height=400, width=900, showlegend=False)
    iplot(figure)
    


#DQN - original dataset
plot_loss_reward(total_losses_dqn_O, total_rewards_dqn_O)


Q_dqn_S, total_maes1_dqn_S, total_losses_dqn_S, total_rewards_dqn_S = train_dqn(Environment1(state_space_dop))

#state_space_df

Q_dqn_S, total_losses_dqn_S, total_rewards_dqn_S = train_dqn(Environment1(state_space_dop))



##state_space_df
plot_loss_reward(total_losses_dqn_S, total_rewards_dqn_S)


# In[18]:


#state_space_df - synthetic

Q_dqn_S_Syn, total_losses_dqn_S_Syn, total_rewards_dqn_S_Syn = train_dqn(Environment1(state_space_Syn_dop))


##state_space_df synthetic
plot_loss_reward(total_losses_dqn_S_Syn, total_rewards_dqn_S_Syn)


#https://www.kaggle.com/code/itoeiji/deep-reinforcement-learning-on-stock-data
# Double DQN

def train_ddqn(env):

    class Q_Network(chainer.Chain):

        def __init__(self, input_size, hidden_size, output_size):
            super(Q_Network, self).__init__(
                fc1 = L.Linear(input_size, hidden_size),
                fc2 = L.Linear(hidden_size, hidden_size),
                fc3 = L.Linear(hidden_size, output_size)
            )

        def __call__(self, x):
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            y = self.fc3(h)
            return y

        def reset(self):
            self.zerograds()

    Q = Q_Network(input_size=env.history_t+1, hidden_size=100, output_size=3)
    Q_ast = copy.deepcopy(Q)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(Q)

    epoch_num = 1000
    step_max = len(env.data)-1
    memory_size = 200
    batch_size = 50
    epsilon = 1.0
    epsilon_decrease = 1e-3
    epsilon_min = 0.1
    start_reduce_epsilon = 200
    train_freq = 10
    update_q_freq = 20
    gamma = 0.97
    show_log_freq = 5

    memory = []
    total_step = 0
    total_rewards = []
    total_losses = []
    total_maes1 = [] ##
    total_accuracies1 = [] ##

    start = time.time()
    for epoch in range(epoch_num):

        pobs = env.reset()
        step = 0
        done = False
        total_reward = 0
        total_loss = 0
        total_mae1 = 0 ## 
        total_acc1=0 ##

        while not done and step < step_max:

            # select act
            pact = np.random.randint(2)
            if np.random.rand() > epsilon:
                pact = Q(np.array(pobs, dtype=np.float32).reshape(1, -1))
                pact = np.argmax(pact.data)

            # act
            obs, reward, done = env.step(pact)

            # add memory
            memory.append((pobs, pact, reward, obs, done))
            if len(memory) > memory_size:
                memory.pop(0)

            # train or update q
            if len(memory) == memory_size:
                if total_step % train_freq == 0:
                    shuffled_memory = np.random.permutation(memory)
                    memory_idx = range(len(shuffled_memory))
                    for i in memory_idx[::batch_size]:
                        batch = np.array(shuffled_memory[i:i+batch_size])
                        b_pobs = np.array(batch[:, 0].tolist(), dtype=np.float32).reshape(batch_size, -1)
                        b_pact = np.array(batch[:, 1].tolist(), dtype=np.int32)
                        b_reward = np.array(batch[:, 2].tolist(), dtype=np.int32)
                        b_obs = np.array(batch[:, 3].tolist(), dtype=np.float32).reshape(batch_size, -1)
                        b_done = np.array(batch[:, 4].tolist(), dtype=np.bool)

                        q = Q(b_pobs)
                        """ <<< DQN -> Double DQN
                        maxq = np.max(Q_ast(b_obs).data, axis=1)
                        === """
                        indices = np.argmax(q.data, axis=1)
                        maxqs = Q_ast(b_obs).data
                        """ >>> """
                        target = copy.deepcopy(q.data)
                        for j in range(batch_size):
                            """ <<< DQN -> Double DQN
                            target[j, b_pact[j]] = b_reward[j]+gamma*maxq[j]*(not b_done[j])
                            === """
                            target[j, b_pact[j]] = b_reward[j]+gamma*maxqs[j, indices[j]]*(not b_done[j])
                            """ >>> """
                        Q.reset()
                        loss = F.mean_squared_error(q, target)
                        mae1 =  F.mean_absolute_error(q, target) 
                        total_loss += loss.data
                        total_mae1 += mae1.data  ##
                        loss.backward()
                        optimizer.update()

                if total_step % update_q_freq == 0:
                    Q_ast = copy.deepcopy(Q)

            # epsilon
            if epsilon > epsilon_min and total_step > start_reduce_epsilon:
                epsilon -= epsilon_decrease

            # next step
            total_reward += reward
            pobs = obs
            step += 1
            total_step += 1

        total_rewards.append(total_reward)
        total_losses.append(total_loss)
        total_maes1.append(total_mae1)      ##

        if (epoch+1) % show_log_freq == 0:
            log_reward = sum(total_rewards[((epoch+1)-show_log_freq):])/show_log_freq
            log_loss = sum(total_losses[((epoch+1)-show_log_freq):])/show_log_freq
            log_mae = sum(total_maes1[((epoch+1)-show_log_freq):])/show_log_freq #####
            elapsed_time = time.time()-start
            print('\t'.join(map(str, [epoch+1, epsilon, total_step, log_reward, log_loss, elapsed_time])))
            start = time.time()
            
    return  b_pact, maxqs, Q, total_maes1, total_losses, total_rewards


Q_ddqn_O, total_losses_ddqn_O, total_rewards_ddqn_O = train_ddqn(Environment1(training_df_reward))



plot_loss_reward(total_losses_ddqn_O, total_rewards_ddqn_O)



Q_ddqn_S, total_losses_ddqn_S, total_rewards_ddqn_S = train_ddqn(Environment1(state_space_dop))


plot_loss_reward(total_losses_ddqn_S, total_rewards_ddqn_S)


#synthetic
Q_ddqn_S_Syn, total_losses_ddqn_S_Syn, total_rewards_ddqn_S_Syn = train_ddqn(Environment1(state_space_Syn_dop))


plot_loss_reward(total_losses_ddqn_S_Syn, total_rewards_ddqn_S_Syn)

:


#https://www.kaggle.com/code/itoeiji/deep-reinforcement-learning-on-stock-data
# Dueling Double DQN

def train_dddqn(env):

    """ <<< Double DQN -> Dueling Double DQN
    class Q_Network(chainer.Chain):

        def __init__(self, input_size, hidden_size, output_size):
            super(Q_Network, self).__init__(
                fc1 = L.Linear(input_size, hidden_size),
                fc2 = L.Linear(hidden_size, hidden_size),
                fc3 = L.Linear(hidden_size, output_size)
            )

        def __call__(self, x):
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            y = self.fc3(h)
            return y

        def reset(self):
            self.zerograds()
    === """
    class Q_Network(chainer.Chain):

        def __init__(self, input_size, hidden_size, output_size):
            super(Q_Network, self).__init__(
                fc1 = L.Linear(input_size, hidden_size),
                fc2 = L.Linear(hidden_size, hidden_size),
                fc3 = L.Linear(hidden_size, hidden_size//2),
                fc4 = L.Linear(hidden_size, hidden_size//2),
                state_value = L.Linear(hidden_size//2, 1),
                advantage_value = L.Linear(hidden_size//2, output_size)
            )
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.output_size = output_size

        def __call__(self, x):
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            hs = F.relu(self.fc3(h))
            ha = F.relu(self.fc4(h))
            state_value = self.state_value(hs)
            advantage_value = self.advantage_value(ha)
            advantage_mean = (F.sum(advantage_value, axis=1)/float(self.output_size)).reshape(-1, 1)
            q_value = F.concat([state_value for _ in range(self.output_size)], axis=1) + (advantage_value - F.concat([advantage_mean for _ in range(self.output_size)], axis=1))
            return q_value

        def reset(self):
            self.zerograds()
    """ >>> """

    Q = Q_Network(input_size=env.history_t+1, hidden_size=100, output_size=3)
    Q_ast = copy.deepcopy(Q)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(Q)

    epoch_num = 1000
    step_max = len(env.data)-1
    memory_size = 200
    batch_size = 50
    epsilon = 1.0
    epsilon_decrease = 1e-3
    epsilon_min = 0.1
    start_reduce_epsilon = 200
    train_freq = 10
    update_q_freq = 20
    gamma = 0.97
    show_log_freq = 5

    memory = []
    total_step = 0
    total_rewards = []
    total_losses = []
    total_maes1 = [] ##
    total_accuracies1 = [] ##
    
    start = time.time()
    for epoch in range(epoch_num):

        pobs = env.reset()
        step = 0
        done = False
        total_reward = 0
        total_loss = 0
        total_mae1 = 0 ## 
        total_acc1=0 ##

        while not done and step < step_max:

            # select act
            pact = np.random.randint(2)
            if np.random.rand() > epsilon:
                pact = Q(np.array(pobs, dtype=np.float32).reshape(1, -1))
                pact = np.argmax(pact.data)

            # act
            obs, reward, done = env.step(pact)

            # add memory
            memory.append((pobs, pact, reward, obs, done))
            if len(memory) > memory_size:
                memory.pop(0)

            # train or update q
            if len(memory) == memory_size:
                if total_step % train_freq == 0:
                    shuffled_memory = np.random.permutation(memory)
                    memory_idx = range(len(shuffled_memory))
                    for i in memory_idx[::batch_size]:
                        batch = np.array(shuffled_memory[i:i+batch_size])
                        b_pobs = np.array(batch[:, 0].tolist(), dtype=np.float32).reshape(batch_size, -1)
                        b_pact = np.array(batch[:, 1].tolist(), dtype=np.int32)
                        b_reward = np.array(batch[:, 2].tolist(), dtype=np.int32)
                        b_obs = np.array(batch[:, 3].tolist(), dtype=np.float32).reshape(batch_size, -1)
                        b_done = np.array(batch[:, 4].tolist(), dtype=np.bool)

                        q = Q(b_pobs)
                        """ <<< DQN -> Double DQN
                        maxq = np.max(Q_ast(b_obs).data, axis=1)
                        === """
                        indices = np.argmax(q.data, axis=1)
                        maxqs = Q_ast(b_obs).data
                        """ >>> """
                        target = copy.deepcopy(q.data)
                        for j in range(batch_size):
                            """ <<< DQN -> Double DQN
                            target[j, b_pact[j]] = b_reward[j]+gamma*maxq[j]*(not b_done[j])
                            === """
                            target[j, b_pact[j]] = b_reward[j]+gamma*maxqs[j, indices[j]]*(not b_done[j])
                            """ >>> """
                        Q.reset()
                        loss = F.mean_squared_error(q, target)
                        mae1 =  F.mean_absolute_error(q, target) 
                        
                        total_loss += loss.data
                        total_mae1 += mae1.data  ##
                        loss.backward()
                        optimizer.update()

                if total_step % update_q_freq == 0:
                    Q_ast = copy.deepcopy(Q)

            # epsilon
            if epsilon > epsilon_min and total_step > start_reduce_epsilon:
                epsilon -= epsilon_decrease

            # next step
            total_reward += reward
            pobs = obs
            step += 1
            total_step += 1

        total_rewards.append(total_reward)
        total_losses.append(total_loss)
        total_maes1.append(total_mae1)

        if (epoch+1) % show_log_freq == 0:
            log_reward = sum(total_rewards[((epoch+1)-show_log_freq):])/show_log_freq
            log_loss = sum(total_losses[((epoch+1)-show_log_freq):])/show_log_freq
            log_mae = sum(total_maes1[((epoch+1)-show_log_freq):])/show_log_freq #####
            elapsed_time = time.time()-start
            print('\t'.join(map(str, [epoch+1, epsilon, total_step, log_reward, log_loss, elapsed_time])))
            start = time.time()
            
    return b_pact, maxqs, Q, total_maes1, total_losses, total_rewards


#DDDQN
p, mq, q11,mae_ddd,tl,tr = train_dddqn(Environment1(training_df_reward))



#DDDQN
Q_dddqn_O, total_losses_dddqn_O, total_rewards_dddqn_O = train_dddqn(Environment1(training_df_reward))




#DDDQN
plot_loss_reward(total_losses_dddqn_O, total_rewards_dddqn_O)


#DDDQN - state_space_temp
Q_dddqn_S, total_losses_dddqn_S, total_rewards_dddqn_S = train_dddqn(Environment1(state_space_dop))



#DDDQN - state_space_temp
plot_loss_reward(total_losses_dddqn_S, total_rewards_dddqn_S)



#DDDQN - state_space_temp - synthetic
Q_dddqn_S_Syn, total_losses_dddqn_S_Syn, total_rewards_dddqn_S_Syn = train_dddqn(Environment1(state_space_Syn_dop))


#DDDQN - state_space_temp - synthetic
plot_loss_reward(total_losses_dddqn_S_Syn, total_rewards_dddqn_S_Syn)



total_losses_dddqn_S #Average of first 5 episodes = 8.542735396651551  (i.e. 7.810614259447902,10.771573897451162,14.867561586201191,6.66842009103857,2.59550714911893,)
            # similarly, total_rewards_dddqn_S = average of first 5 episodes (rewards) = -14.8 (i.e. -92.0,19.0, 20.0,-5.0,-16.0,)



#https://www.kaggle.com/code/itoeiji/deep-reinforcement-learning-on-stock-data
# Dueling Double DQN

def train_dddqn_hyper(env):

    """ <<< Double DQN -> Dueling Double DQN
    class Q_Network(chainer.Chain):

        def __init__(self, input_size, hidden_size, output_size):
            super(Q_Network, self).__init__(
                fc1 = L.Linear(input_size, hidden_size),
                fc2 = L.Linear(hidden_size, hidden_size),
                fc3 = L.Linear(hidden_size, output_size)
            )

        def __call__(self, x):
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            y = self.fc3(h)
            return y

        def reset(self):
            self.zerograds()
    === """
    class Q_Network(chainer.Chain):

        def __init__(self, input_size, hidden_size, output_size):
            super(Q_Network, self).__init__(
                fc1 = L.Linear(input_size, hidden_size),
                fc2 = L.Linear(hidden_size, hidden_size),
                fc3 = L.Linear(hidden_size, hidden_size//2),
                fc4 = L.Linear(hidden_size, hidden_size//2),
                state_value = L.Linear(hidden_size//2, 1),
                advantage_value = L.Linear(hidden_size//2, output_size)
            )
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.output_size = output_size

        def __call__(self, x):
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            hs = F.relu(self.fc3(h))
            ha = F.relu(self.fc4(h))
            state_value = self.state_value(hs)
            advantage_value = self.advantage_value(ha)
            advantage_mean = (F.sum(advantage_value, axis=1)/float(self.output_size)).reshape(-1, 1)
            q_value = F.concat([state_value for _ in range(self.output_size)], axis=1) + (advantage_value - F.concat([advantage_mean for _ in range(self.output_size)], axis=1))
            return q_value

        def reset(self):
            self.zerograds()
    """ >>> """

    Q = Q_Network(input_size=env.history_t+1, hidden_size=100, output_size=3)
    Q_ast = copy.deepcopy(Q)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(Q)

    epoch_num = 5000 #changed from 1000
    step_max = len(env.data)-1
    memory_size = 200
    batch_size = 100
    batch_size = 50
    epsilon = 1.0
    epsilon_decrease = 1e-3
#     epsilon_min = 0.1
    epsilon_min = 0.05 #changed for hyperparameter tuning
    start_reduce_epsilon = 200
    train_freq = 10
    update_q_freq = 20
    gamma = 0.97
    show_log_freq = 5

    memory = []
    total_step = 0
    total_rewards = []
    total_losses = []
    total_maes1 = [] ##
    total_accuracies1 = [] ##
    
    start = time.time()
    for epoch in range(epoch_num):

        pobs = env.reset()
        step = 0
        done = False
        total_reward = 0
        total_loss = 0
        total_mae1 = 0 ## 
        total_acc1=0 ##

        while not done and step < step_max:

            # select act
            pact = np.random.randint(2)
            if np.random.rand() > epsilon:
                pact = Q(np.array(pobs, dtype=np.float32).reshape(1, -1))
                pact = np.argmax(pact.data)

            # act
            obs, reward, done = env.step(pact)

            # add memory
            memory.append((pobs, pact, reward, obs, done))
            if len(memory) > memory_size:
                memory.pop(0)

            # train or update q
            if len(memory) == memory_size:
                if total_step % train_freq == 0:
                    shuffled_memory = np.random.permutation(memory)
                    memory_idx = range(len(shuffled_memory))
                    for i in memory_idx[::batch_size]:
                        batch = np.array(shuffled_memory[i:i+batch_size])
                        b_pobs = np.array(batch[:, 0].tolist(), dtype=np.float32).reshape(batch_size, -1)
                        b_pact = np.array(batch[:, 1].tolist(), dtype=np.int32)
                        b_reward = np.array(batch[:, 2].tolist(), dtype=np.int32)
                        b_obs = np.array(batch[:, 3].tolist(), dtype=np.float32).reshape(batch_size, -1)
                        b_done = np.array(batch[:, 4].tolist(), dtype=np.bool)

                        q = Q(b_pobs)
                        """ <<< DQN -> Double DQN
                        maxq = np.max(Q_ast(b_obs).data, axis=1)
                        === """
                        indices = np.argmax(q.data, axis=1)
                        maxqs = Q_ast(b_obs).data
                        """ >>> """
                        target = copy.deepcopy(q.data)
                        for j in range(batch_size):
                            """ <<< DQN -> Double DQN
                            target[j, b_pact[j]] = b_reward[j]+gamma*maxq[j]*(not b_done[j])
                            === """
                            target[j, b_pact[j]] = b_reward[j]+gamma*maxqs[j, indices[j]]*(not b_done[j])
                            """ >>> """
                        Q.reset()
                        loss = F.mean_squared_error(q, target)
                        mae1 =  F.mean_absolute_error(q, target) 
                        
                        total_loss += loss.data
                        total_mae1 += mae1.data  ##
                        loss.backward()
                        optimizer.update()

                if total_step % update_q_freq == 0:
                    Q_ast = copy.deepcopy(Q)

            # epsilon
            if epsilon > epsilon_min and total_step > start_reduce_epsilon:
                epsilon -= epsilon_decrease

            # next step
            total_reward += reward
            pobs = obs
            step += 1
            total_step += 1

        total_rewards.append(total_reward)
        total_losses.append(total_loss)
        total_maes1.append(total_mae1)

        if (epoch+1) % show_log_freq == 0:
            log_reward = sum(total_rewards[((epoch+1)-show_log_freq):])/show_log_freq
            log_loss = sum(total_losses[((epoch+1)-show_log_freq):])/show_log_freq
            log_mae = sum(total_maes1[((epoch+1)-show_log_freq):])/show_log_freq #####
            elapsed_time = time.time()-start
            print('\t'.join(map(str, [epoch+1, epsilon, total_step, log_reward, log_loss, elapsed_time])))
            start = time.time()
            
    return b_pact, maxqs, Q, total_maes1, total_losses, total_rewards

