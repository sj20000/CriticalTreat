

# https://github.com/DMU-XMU/Episodic-Memory-assisted-Approach-for-Sepsis-Treatment/blob/main/continuous/D3QN.py


import tensorflow as tf
import numpy as np
import math
import os
from tqdm import *
import numpy as np
import pandas as pd
from pandas import DataFrame
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle
import copy
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
REWARD_THRESHOLD = 20
reg_lambda = 5
per_alpha = 0.6 # PER hyperparameter
per_epsilon = 0.01 # PER hyperparameter
batch_size = 32
gamma = 0.99 # discount factor
num_steps = 100000 # How many steps to train for
load_model = False #Whether to load a saved model.
save_dir = "./dqn_normal-/"
save_path = "./dqn_normal-/ckpt"#The path to save our model to.
tau = 0.001 #Rate to update target network toward primary network
save_results = False
# SET THIS TO FALSE
clip_reward = True

#sj -copied state_features.txt as is from https://github.com/thxsxth/sepsisrl/blob/master/data/state_features.txt
with open('./data/state_features.txt') as f:
    state_features = f.read().split()
print (state_features)

df = pd.read_csv('./data_o/rl_train_data_final_cont.csv')  #sj -changed data to data_o
val_df = pd.read_csv('./data_o/rl_val_data_final_cont.csv') #sj -changed data to data_o
test_df = pd.read_csv('./data_o/rl_test_data_final_cont.csv') #sj -changed data to data_o
# PER important weights and params
per_flag = True
beta_start = 0.9
df['prob'] = abs(df['reward'])
temp = 1.0/df['prob']
df['imp_weight'] = pow((1.0/len(df) * temp), beta_start)

action_map = {}
count = 0
for iv in range(5):
    for vaso in range(5):
        action_map[(iv,vaso)] = count
        count += 1




#get an id with max records for a particular icustayid

#select a single stayid for the time being - for testing
ICU_train_df_203934 = df.loc[df['icustayid'] == 203934.0]
ICU_train_df_203934


ICU_train_df_203934 = ICU_train_df_203934.iloc[:,:-4]
ICU_train_df_203934 = ICU_train_df_203934.iloc[:,4:]
ICU_train_df_203934


# Splitting dataset into train/val/test        - YES USED 
val_time = 10
test_time = 20
training_data = ICU_train_df_203934.iloc[:val_time]
validation_data = ICU_train_df_203934.iloc[val_time:test_time]
test_data = ICU_train_df_203934.iloc[test_time:]



#/MSC8002/FINAL_MSC8002/MSC8002/Code/1RL_ICUProject_Finalstatespace.ipynb

#https://medium.com/analytics-vidhya/introduction-to-2-dimensional-lstm-autoencoder-47c238fd827f       
import numpy as np
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)        
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)
# Timesteps will define how many Elements we have
TIME_STEPS = 1

X_train, y_train = create_dataset(training_data, training_data, TIME_STEPS)


print(X_train.shape)




from keras.layers import *



#MSC8002/FINAL_MSC8002/MSC8002/Code/1RL_ICUProject_Finalstatespace.ipynb


import tensorflow as tf
from tensorflow import keras

    
# This is the dimension of the original space
# input_dim = 58

# Encoding information
encoding_dim = 10 ## This is the dimension of the latent space (encoding space)

X_t=X_train.reshape(-1,58)

# X_t=X_train.values.reshape(-1,58)
col_num = X_t.shape[1]
input_dim = Input(shape=(col_num,))

X_val, y_val = create_dataset(validation_data, validation_data, TIME_STEPS)
X_v=X_val.reshape(-1,58)
    
encoded = keras.layers.Dense(encoding_dim, activation='relu')(input_dim)
# Decoding information
decoded = keras.layers.Dense(col_num, activation='sigmoid')(encoded)
# Autoencoder information (encoder + decoder)
autoencoder = keras.Model(inputs=input_dim, outputs=decoded)

# Train the autoencoder
autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
autoencoder.fit(X_t, X_t, epochs=10, batch_size=100, shuffle=True, validation_data=(X_v, X_v))


#######################################################################
#LATENT REPRESENTATION 

# Encoder information for feature extraction
encoder = keras.Model(inputs=input_dim, outputs=encoded)
encoded_input = Input(shape=(encoding_dim,))
encoded_output = encoder.predict(X_v)

# Show the encoded values
print(encoded_output[:5])
#######################################################################



# statespace for the icustayid = 203934
np.savetxt('statespace_203934.csv', encoded_output, delimiter=',')


#https://stackoverflow.com/questions/69269890/keras-attributeerror-sequential-object-has-no-attribute-nested-inputs
input_dim = 58
inp = Input((input_dim,))



import matplotlib.pyplot as plt          
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(15,15))
X_val, y_val = create_dataset(validation_data, validation_data, TIME_STEPS)
X_v=X_val.reshape(-1,58)
model_history = autoencoder.fit(X_t, X_t, epochs=10, batch_size=100, shuffle=True, validation_data=(X_v, X_v))



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




# h/MSC8002/FINAL_MSC8002/MSC8002/Code/1RL_ICUProject.ipynb
n_episodes = 1000
gamma = 0.99
epsilon = 1
minibatch_size = 32
r_sums = []  # stores rewards of each epsiode 
replay_memory = [] # replay memory holds s, a, r, s'
mem_max_size = 100000
#https://tomroth.com.au/dqn-simple/


# OneDrive/MSC8002/FINAL_MSC8002/MSC8002/Code/1RL_ICUProject.ipynb

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


# OneDrive/MSC8002/FINAL_MSC8002/MSC8002/Code/1RL_ICUProject.ipynb

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
    epoch_num = 1000  ###FOR ABLATION STUDY -episodes
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


env = Environment1(training_data)


a10_actions_dqn_O, a10_maxq_dqn_O, a10_Q_dqn_O, a10_total_maes1_dqn_O, a10_total_losses_dqn_O, a10_total_rewards_dqn_O = train_dqn(Environment1(training_data))

