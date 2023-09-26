

'''Import Libraries'''
import numpy as np
import pandas as pd 


dfMIMICtable=pd.read_csv('MIMICtable.csv')
dfMIMICtable




# https://github.com/DMU-XMU/Episodic-Memory-assisted-Approach-for-Sepsis-Treatment/blob/main/preprocessing/process_interventions.ipynb
# file:///C:/Users/sjt20/Downloads/s10489-022-04099-7.pdf

# The treatment of sepsis: an episodic memory-assisted deep reinforcement learning approach - Dayang Liang & Huiyi Deng & Yunlong Liu

# This notebook takes the input dataset, finds quartiles for the medical interventions (IV in, max vaso in)
# for each block.
# We then discretise actions in the original dataset according to what bin they fall in, and then save 
# a new dataframe with the discretised actions.


orig_data = dfMIMICtable.copy()
# assume we're using input_4hourly and max_dose_vaso as the input params for now
interventions = orig_data[["max_dose_vaso", "input_4hourly"]]


adjusted_vaso = interventions["max_dose_vaso"][interventions["max_dose_vaso"] >0]
adjusted_iv = interventions["input_4hourly"][interventions["input_4hourly"]>0]




vaso_quartiles = adjusted_vaso.quantile([0.25,0.50,0.75])
iv_quartiles = adjusted_iv.quantile([0.25,0.5,0.75])


vq = np.array(vaso_quartiles)

ivq = np.array(iv_quartiles)

import copy
discretised_int = copy.deepcopy(interventions)



discretised_int['vaso_input'] = discretised_int['max_dose_vaso']
discretised_int['vaso_input'][interventions['max_dose_vaso'] == 0.0] = 0
discretised_int['vaso_input'][(interventions['max_dose_vaso'] > 0.0) & (interventions['max_dose_vaso'] < vq[0])] = 1
discretised_int['vaso_input'][(interventions['max_dose_vaso'] >= vq[0]) & (interventions['max_dose_vaso'] < vq[1])] = 2
discretised_int['vaso_input'][(interventions['max_dose_vaso'] >= vq[1]) & (interventions['max_dose_vaso'] < vq[2])] = 3
a = interventions['max_dose_vaso'] >= vq[2]
discretised_int['vaso_input'][a] = 4


discretised_int['iv_input'] = discretised_int['input_4hourly']
discretised_int['iv_input'][interventions['input_4hourly'] == 0.0] = 0
discretised_int['iv_input'][(interventions['input_4hourly'] > 0.0) & (interventions['input_4hourly'] < ivq[0])] = 1
discretised_int['iv_input'][(interventions['input_4hourly'] >=  ivq[0]) & (interventions['input_4hourly'] <  ivq[1])] = 2
discretised_int['iv_input'][(interventions['input_4hourly'] >=  ivq[1]) & (interventions['input_4hourly'] < ivq[2])] = 3
discretised_int['iv_input'][(interventions['input_4hourly'] >=  ivq[2])] = 4


discretised_int['vaso_input'].plot.hist()



discretised_int['iv_input'].plot.hist()



disc_inp_data = copy.deepcopy(orig_data)


disc_inp_data['vaso_input'] = discretised_int['vaso_input']
disc_inp_data['iv_input'] = discretised_int['iv_input']



disc_inp_data['vaso_input'].value_counts()


disc_inp_data['iv_input'].value_counts()



# https://github.com/DMU-XMU/Episodic-Memory-assisted-Approach-for-Sepsis-Treatment/blob/main/preprocessing/process_interventions.ipynb
disc_inp_data.to_csv('./data/discretised_input_data.csv', index=False)



dfdiscretised_input_data=pd.read_csv('./data/discretised_input_data.csv')


# https://github.com/DMU-XMU/Episodic-Memory-assisted-Approach-for-Sepsis-Treatment/blob/main/preprocessing/preprocess_data.ipynb

# add rewards - sparsely for now; reward function shaping comes in a separate script
dfdiscretised_input_data['reward'] = 0
for i in dfdiscretised_input_data.index:
    if i == 0:
        continue
    else:
        if dfdiscretised_input_data.loc[i, 'icustayid'] != dfdiscretised_input_data.loc[i-1, 'icustayid']:
            if dfdiscretised_input_data.loc[i-1, 'died_in_hosp'] == 1:
                dfdiscretised_input_data.loc[i-1,'reward'] = -100
            elif dfdiscretised_input_data.loc[i-1, 'died_in_hosp'] == 0:
                dfdiscretised_input_data.loc[i-1,'reward'] = 100
            else:
                print ("error in row", i-1)
if dfdiscretised_input_data.loc[len(dfdiscretised_input_data)-1, 'died_in_hosp'] == 1:
    dfdiscretised_input_data.loc[len(dfdiscretised_input_data)-1, 'reward'] = -100
elif dfdiscretised_input_data.loc[len(dfdiscretised_input_data)-1, 'died_in_hosp'] == 0:
     dfdiscretised_input_data.loc[len(dfdiscretised_input_data)-1, 'reward'] = 100
print (dfdiscretised_input_data['reward'].value_counts())


# https://github.com/DMU-XMU/Episodic-Memory-assisted-Approach-for-Sepsis-Treatment/blob/main/preprocessing/preprocess_data.ipynb
# now split into train/validation/test sets
import random
unique_ids = dfdiscretised_input_data['icustayid'].unique()
random.shuffle(unique_ids)
train_sample = 0.7
val_sample = 0.1
test_sample = 0.2
train_num = int(len(unique_ids) * 0.7)
val_num = int(len(unique_ids)*0.1) + train_num
train_ids = unique_ids[:train_num]
val_ids = unique_ids[train_num:val_num]
test_ids = unique_ids[val_num:]



# https://github.com/DMU-XMU/Episodic-Memory-assisted-Approach-for-Sepsis-Treatment/blob/main/preprocessing/preprocess_data.ipynb

from pandas import DataFrame
train_set = DataFrame()
train_set = dfdiscretised_input_data.loc[dfdiscretised_input_data['icustayid'].isin(train_ids)]

val_set = DataFrame()
val_set = dfdiscretised_input_data.loc[dfdiscretised_input_data['icustayid'].isin(val_ids)]

test_set = DataFrame()
test_set = dfdiscretised_input_data.loc[dfdiscretised_input_data['icustayid'].isin(test_ids)]


binary_fields = ['gender','mechvent','re_admission']
norm_fields= ['age','Weight_kg','GCS','HR','SysBP','MeanBP','DiaBP','RR','Temp_C','FiO2_1',
    'Potassium','Sodium','Chloride','Glucose','Magnesium','Calcium',
    'Hb','WBC_count','Platelets_count','PTT','PT','Arterial_pH','paO2','paCO2',
    'Arterial_BE','HCO3','Arterial_lactate','SOFA','SIRS','Shock_Index',
    'PaO2_FiO2','cumulated_balance', 'elixhauser', 'Albumin', u'CO2_mEqL', 'Ionised_Ca']
log_fields = ['max_dose_vaso','SpO2','BUN','Creatinine','SGOT','SGPT','Total_bili','INR',
              'input_total','input_4hourly','output_total','output_4hourly', 'bloc']


# normalise binary fields
train_set[binary_fields] = train_set[binary_fields] - 0.5 
val_set[binary_fields] = val_set[binary_fields] - 0.5 
test_set[binary_fields] = test_set[binary_fields] - 0.5 




# normal distn fields
for item in norm_fields:
    av = train_set[item].mean()
    std = train_set[item].std()
    print(item,av,std)
    train_set[item] = (train_set[item] - av) / std
    val_set[item] = (val_set[item] - av) / std
    test_set[item] = (test_set[item] - av) / std



# log normal fields
train_set[log_fields] = np.log(0.1 + train_set[log_fields])
val_set[log_fields] = np.log(0.1 + val_set[log_fields])
test_set[log_fields] = np.log(0.1 + test_set[log_fields])
for item in log_fields:
    av = train_set[item].mean()
    std = train_set[item].std()
    print(item,av,std)
    train_set[item] = (train_set[item] - av) / std
    val_set[item] = (val_set[item] - av) / std
    test_set[item] = (test_set[item] - av) / std



train_set.to_csv('./data/rl_train_set_unscaled.csv',index = False)
val_set.to_csv('./data/rl_val_set_unscaled.csv', index = False)
test_set.to_csv('./data/rl_test_set_unscaled.csv', index = False)



# scale features to [0,1] in train set, similar in val and test
import copy
scalable_fields = copy.deepcopy(binary_fields)
scalable_fields.extend(norm_fields)
scalable_fields.extend(log_fields)
for col in scalable_fields:
    minimum = min(train_set[col])
    maximum = max(train_set[col])
    train_set[col] = (train_set[col] - minimum)/(maximum-minimum)
    val_set[col] = (val_set[col] - minimum)/(maximum-minimum)
    test_set[col] = (test_set[col] - minimum)/(maximum-minimum)
    
    
train_set.to_csv('./data/rl_train_set_scaled.csv',index = False)
val_set.to_csv('./data/rl_val_set_scaled.csv', index = False)
test_set.to_csv('./data/rl_test_set_scaled.csv', index = False)



# https://github.com/DMU-XMU/Episodic-Memory-assisted-Approach-for-Sepsis-Treatment/blob/main/preprocessing/new_rewards.ipynb

#  Reward function shaping in this script


df_train = pd.read_csv('./data/rl_train_set_scaled.csv')
df_val =  pd.read_csv('./data/rl_val_set_scaled.csv')
df_test = pd.read_csv('./data/rl_test_set_scaled.csv')





# Smaller rewards to improve stability in continuous models
df_train.loc[df_train['reward'] > 15,'reward'] = 15
df_train.loc[df_train['reward'] < -15,'reward'] = -15

df_val.loc[df_val['reward'] > 15,'reward'] = 15
df_val.loc[df_val['reward'] < -15,'reward'] = -15

df_test.loc[df_test['reward'] > 15,'reward'] = 15
df_test.loc[df_test['reward'] < -15,'reward'] = -15


orig_df = pd.read_csv('./MIMICtable.csv')


# In[8]:


c0 = -0.1/4
c1 = -0.5/4
c2 = -2

# add rewards
orig_df['shaped_reward'] = 0
for i in orig_df.index:
    if i == 0:
        continue
    if orig_df.loc[i, 'icustayid'] == orig_df.loc[i-1, 'icustayid']:
        sofa_cur = orig_df.loc[i,'SOFA']
        sofa_prev = orig_df.loc[i-1,'SOFA']
        lact_cur = orig_df.loc[i,'Arterial_lactate']
        lact_prev = orig_df.loc[i-1,'Arterial_lactate']
        reward = 0
        if sofa_cur == sofa_prev and sofa_cur != 0:
            reward += c0
        reward += c1*(sofa_cur-sofa_prev)
        reward += c2*np.tanh(lact_cur - lact_prev)
        orig_df.loc[i-1,'shaped_reward'] = reward
    if i % 10000 == 0:
        print(i)


print (orig_df['shaped_reward'].value_counts())


df_train['reward'].hist(bins=100)
df_train['reward'].value_counts()#+df_test['reward'].value_counts()+df_val['reward'].value_counts()



train_ids = df_train['icustayid'].unique()
val_ids = df_val['icustayid'].unique()
test_ids = df_test['icustayid'].unique()


train_rewards = orig_df.loc[orig_df['icustayid'].isin(train_ids)]['shaped_reward']



# check this works as expected
val_rewards = orig_df.loc[orig_df['icustayid'].isin(val_ids)]['shaped_reward']


test_rewards = orig_df.loc[orig_df['icustayid'].isin(test_ids)]['shaped_reward']


# check that this sums to the total number of data items -- needs to be re-run
len(df_train) + len(df_test) + len(df_val)



df_train['reward'] += np.array(train_rewards)
df_val['reward'] += np.array(val_rewards)
df_test['reward'] += np.array(test_rewards)


uids = df_train['icustayid'].unique()
len_trajecties = []
for uid in uids:
    len_trajecties.append(len(df_train[df_train['icustayid']==uid]))


mid_len = int((sum(len_trajecties)-len(len_trajecties))/len(len_trajecties))//2+1


length = 0
uid = 0 
#C4 = 0.5#0.1
C4 = 0.10
for t in range(1,len(df_train)):
    if df_train.loc[t,'icustayid']!=df_train.loc[t-1,'icustayid']:
        length = 0
        uid =+1
        continue
    df_train.loc[t-1,'reward']+= (length-mid_len)*(C4+ length*0.003)
    length +=1
length = 0
uid = 0 
for t in range(1,len(df_test)):
    if df_test.loc[t,'icustayid']!=df_test.loc[t-1,'icustayid']:
        length = 0
        uid =+1
        continue
    df_test.loc[t-1,'reward']+= (length-mid_len)*(C4+ length*0.003)
    length +=1
length = 0
uid = 0 
for t in range(1,len(df_val)):
    if df_val.loc[t,'icustayid']!=df_val.loc[t-1,'icustayid']:
        length = 0
        uid =+1
        continue
    df_val.loc[t-1,'reward']+= (length-mid_len)*(C4+ length*0.003)
    length +=1
length = 1



unique_ids = df_test['icustayid'].unique()
phys_vals = []
gamma = 0.99
for uid in unique_ids:
    traj = df_test.loc[df_test['icustayid'] == uid]
    ret = 0
    reversed_traj = traj.iloc[::-1]
    for row in reversed_traj.index:
        ret = reversed_traj.loc[row,'reward'] + gamma*ret
    phys_vals.append(ret)
np.mean(phys_vals)


df_train.to_csv('./data_o/rl_train_data_final_or.csv',index=False)
df_val.to_csv('./data_o/rl_val_data_final_or.csv', index=False)
df_test.to_csv('./data_o/rl_test_data_final_or.csv',index=False)


# Episodic-Memory-assisted-Approach-for-Sepsis-Treatment/continuous/draw.ipynb



#commented in https://github.com/DMU-XMU/Episodic-Memory-assisted-Approach-for-Sepsis-Treatment/blob/main/continuous/draw.ipynb
######### add MC-return ##########
def add_mc_return(df):
    unique_ids = df['icustayid'].unique()
    gamma = 0.99
    index = 0
    mc = []
    df['Gt'] = 0
    for uid in unique_ids:
        traj = df.loc[df['icustayid'] == uid]
        ret = 0
        reversed_traj = traj.iloc[::-1]
        for row in reversed_traj.index:
            ret = reversed_traj.loc[row,'reward'] + gamma*ret
            mc.append(ret)
        for r in  mc[::-1]:
            df.loc[index,'Gt'] = r
            index +=1
            mc = []
    print("done!")
    return df
df_train = add_mc_return(df_train)
df_train.to_csv('./data_o/rl_train_data_final_cont.csv',index=False)




# Similar to above
df_test = add_mc_return(df_test)
df_test.to_csv('./data_o/rl_test_data_final_cont.csv',index=False)

df_val = add_mc_return(df_val)
df_val.to_csv('./data_o/rl_val_data_final_cont.csv',index=False)




# START HERE



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

df = pd.read_csv('./data_o/rl_train_data_final_cont.csv')
val_df = pd.read_csv('./data_o/rl_val_data_final_cont.csv')
test_df = pd.read_csv('./data_o/rl_test_data_final_cont.csv')
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
ICU_train_df_212772 = ICU_train_df.loc[ICU_train_df['icustayid'] == 212772.0]
ICU_train_df_212772



ICU_train_df_212772 = ICU_train_df_212772.iloc[:,4:]



# Splitting dataset into train/val/test        -
test_time = 20
training_data = ICU_train_df_212772.iloc[:val_time]
validation_data = ICU_train_df_212772.iloc[val_time:test_time]
test_data = ICU_train_df_212772.iloc[test_time:]




#http://localhost:8888/notebooks/OneDrive/MSC8002/FINAL_MSC8002/MSC8002/Code/1RL_ICUProject_Finalstatespace.ipynb

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

# In[107]:


#/MSC8002/FINAL_MSC8002/MSC8002/Code/1RL_ICUProject_Finalstatespace.ipynb


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



# statespace for the icustayid = 212772
np.savetxt('statespace_212772.csv', encoded_output, delimiter=',')



#https://stackoverflow.com/questions/69269890/keras-attributeerror-sequential-object-has-no-attribute-nested-inputs
input_dim = 58
inp = Input((input_dim,))



#https://ekamperi.github.io/machine%20learning/2021/01/21/encoder-decoder-model.html
from pygame import color

color=(255, 0, 0)
c=np.random.random(999)

class TestEncoder(tf.keras.callbacks.Callback):
    def __init__(self, x_test, y_test,inp):
        super(TestEncoder, self).__init__()
        self.x_test = x_test
        self.y_test = y_test
        self.inp = inp
        self.current_epoch = 0

    def on_epoch_begin(self, epoch, logs={}):
        self.current_epoch = self.current_epoch + 1
        
#         autoencoder = Model(inputs=inp, outputs=decoder(encoder(inp)))

        encoder_model = keras.Model(inputs=self.inp, outputs=self.model.get_layer('encoder_output').output)

#         encoder_model = Model(inputs=self.inp,
#                               outputs=self.model.get_layer('encoder_output').output)
        
        encoder_output = encoder_model(self.x_test)
        plt.subplot(4, 3, self.current_epoch)
        plt.scatter(encoder_output[:, 0],
                    encoder_output[:, 1], s=20, alpha=0.8,
                    cmap='Set1', c=c)
#                     cmap='Set1', c=self.y_test[0:self.x_test.shape[0]])
#         plt.xlim(-9, 9)
#         plt.ylim(-9, 9)
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')


# In[109]:


import matplotlib.pyplot as plt          
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(15,15))
X_val, y_val = create_dataset(validation_data, validation_data, TIME_STEPS)
X_v=X_val.reshape(-1,58)
model_history = autoencoder.fit(X_t, X_t, epochs=10, batch_size=100, shuffle=True, validation_data=(X_v, X_v))

# model_history = autoencoder.fit(X_t, X_t, epochs=12, batch_size=32, verbose=0,
#                                 callbacks=[TestEncoder(X_v,X_v, inp)])




# http://localhost:8888/notebooks/OneDrive/MSC8002/FINAL_MSC8002/MSC8002/Code/1RL_ICUProject.ipynb
n_episodes = 1000
gamma = 0.99
epsilon = 1
minibatch_size = 32
r_sums = []  # stores rewards of each epsiode 
replay_memory = [] # replay memory holds s, a, r, s'
mem_max_size = 100000
#https://tomroth.com.au/dqn-simple/


# In[112]:



# http://localhost:8888/notebooks/OneDrive/MSC8002/FINAL_MSC8002/MSC8002/Code/1RL_ICUProject.ipynb

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



pd.set_option('display.max_rows', None)
df[df['vaso_input'].isin([1,2,3,4])]




# /MSC8002/FINAL_MSC8002/MSC8002/Code/1RL_ICUProject.ipynb

#https://www.kaggle.com/code/itoeiji/deep-reinforcement-learning-on-stock-data
class Environment1:             # TRIAL 
    
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




a10_actions_dqn_O, a10_maxq_dqn_O, a10_Q_dqn_O, a10_total_maes1_dqn_O, a10_total_losses_dqn_O, a10_total_rewards_dqn_O = train_dqn(Environment1(training_df_reward))




hidden_1_size = 128
hidden_2_size = 128
#  Q-network uses Leaky ReLU activation
class Qnetwork():
    def __init__(self):
        self.phase = tf.placeholder(tf.bool)

        self.num_actions = 25

        self.input_size = len(state_features)

        self.state = tf.placeholder(tf.float32, shape=[None, self.input_size], name="input_state")

        self.fc_1 = tf.contrib.layers.fully_connected(self.state, hidden_1_size, activation_fn=None)
        self.fc_1_bn = tf.contrib.layers.batch_norm(self.fc_1, center=True, scale=True, is_training=self.phase)
        self.fc_1_ac = tf.maximum(self.fc_1_bn, self.fc_1_bn * 0.01)
        self.fc_2 = tf.contrib.layers.fully_connected(self.fc_1_ac, hidden_2_size, activation_fn=None)
        self.fc_2_bn = tf.contrib.layers.batch_norm(self.fc_2, center=True, scale=True, is_training=self.phase)
        self.fc_2_ac = tf.maximum(self.fc_2_bn, self.fc_2_bn * 0.01)

        # advantage and value streams
        self.streamA, self.streamV = tf.split(self.fc_2_ac, 2, axis=1)
        self.AW = tf.Variable(tf.random_normal([hidden_2_size // 2, self.num_actions]))
        self.VW = tf.Variable(tf.random_normal([hidden_2_size // 2, 1]))
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)

        # Then combine them together to get our final Q-values.
        self.q_output = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))

        self.predict = tf.argmax(self.q_output, 1, name='predict')  # vector of length batch size

        # Below we obtain the loss by taking the sum of squares difference between the target and predicted Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, self.num_actions, dtype=tf.float32)

        # Importance sampling weights for PER, used in network update
        self.imp_weights = tf.placeholder(shape=[None], dtype=tf.float32)

        # select the Q values for the actions that would be selected
        self.Q = tf.reduce_sum(tf.multiply(self.q_output, self.actions_onehot),
                               reduction_indices=1)  # batch size x 1 vector

        # regularisation penalises the network when it produces rewards that are above the
        # reward threshold, to ensure reasonable Q-value predictions
        self.reg_vector = tf.maximum(tf.abs(self.Q) - REWARD_THRESHOLD, 0)
        self.reg_term = tf.reduce_sum(self.reg_vector)

        self.abs_error = tf.abs(self.targetQ - self.Q)

        self.td_error = tf.square(self.targetQ - self.Q)

        # below is the loss when we are not using PER
        self.old_loss = tf.reduce_mean(self.td_error)

        # as in the paper, to get PER loss we weight the squared error by the importance weights
        self.per_error = tf.multiply(self.td_error, self.imp_weights)

        # total loss is a sum of PER loss and the regularisation term
        if per_flag:
            self.loss = tf.reduce_mean(self.per_error) + reg_lambda * self.reg_term
        else:
            self.loss = self.old_loss + reg_lambda * self.reg_term

        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            # Ensures that we execute the update_ops before performing the model update, so batchnorm works
            self.update_model = self.trainer.minimize(self.loss)
def update_target_graph(tf_vars,tau):
    total_vars = len(tf_vars)
    op_holder = []
    for idx,var in enumerate(tf_vars[0:int(total_vars/2)]):
        op_holder.append(tf_vars[idx+int(total_vars/2)].assign((var.value()*tau) + ((1-tau)*tf_vars[idx+int(total_vars/2)].value())))
    return op_holder
def update_target(op_holder,sess):
    for op in op_holder:
        sess.run(op)


def process_train_batch(size):
    if per_flag:
        # uses prioritised exp replay
        a = df.sample(n=size, weights=df['prob'])
    else:
        a = df.sample(n=size)
    states = None
    actions = None
    rewards = None
    next_states = None
    done_flags = None
    for i in a.index:
        cur_state = a.loc[i, state_features]
        iv = int(a.loc[i, 'iv_input'])
        vaso = int(a.loc[i, 'vaso_input'])
        action = action_map[iv, vaso]
        reward = a.loc[i, 'reward']

        if clip_reward:
            if reward > 1: reward = 1
            if reward < -1: reward = -1

        if i != df.index[-1]:
            # if not terminal step in trajectory
            if df.loc[i, 'icustayid'] == df.loc[i + 1, 'icustayid']:
                next_state = df.loc[i + 1, state_features]
                done = 0
            else:
                # trajectory is finished
                next_state = np.zeros(len(cur_state))
                done = 1
        else:
            # last entry in df is the final state of that trajectory
            next_state = np.zeros(len(cur_state))
            done = 1

        if states is None:
            states = copy.deepcopy(cur_state)
        else:
            states = np.vstack((states, cur_state))

        if actions is None:
            actions = [action]
        else:
            actions = np.vstack((actions, action))

        if rewards is None:
            rewards = [reward]
        else:
            rewards = np.vstack((rewards, reward))

        if next_states is None:
            next_states = copy.deepcopy(next_state)
        else:
            next_states = np.vstack((next_states, next_state))

        if done_flags is None:
            done_flags = [done]
        else:
            done_flags = np.vstack((done_flags, done))

    return (states, np.squeeze(actions), np.squeeze(rewards), next_states, np.squeeze(done_flags), a)
# extract chunks of length size from the relevant dataframe, and yield these to the caller
def process_eval_batch(size, eval_type=None):
    if eval_type is None:
        raise Exception('Provide eval_type to process_eval_batch')
    elif eval_type == 'train':
        a = df.copy()
    elif eval_type == 'val':
        a = val_df.copy()
    elif eval_type == 'test':
        a = test_df.copy()
    else:
        raise Exception('Unknown eval_type')
    count = 0
    while count < len(a.index):
        states = None
        actions = None
        rewards = None
        next_states = None
        done_flags = None

        start_idx = count
        end_idx = min(len(a.index), count + size)
        segment = a.index[start_idx:end_idx]

        for i in segment:
            cur_state = a.loc[i, state_features]
            iv = int(a.loc[i, 'iv_input'])
            vaso = int(a.loc[i, 'vaso_input'])
            action = action_map[iv, vaso]
            reward = a.loc[i, 'reward']

            if clip_reward:
                if reward > 1: reward = 1
                if reward < -1: reward = -1

            if i != a.index[-1]:
                # if not terminal step in trajectory
                if a.loc[i, 'icustayid'] == a.loc[i + 1, 'icustayid']:
                    next_state = a.loc[i + 1, state_features]
                    done = 0
                else:
                    # trajectory is finished
                    next_state = np.zeros(len(cur_state))
                    done = 1
            else:
                # last entry in df is the final state of that trajectory
                next_state = np.zeros(len(cur_state))
                done = 1

            if states is None:
                states = copy.deepcopy(cur_state)
            else:
                states = np.vstack((states, cur_state))

            if actions is None:
                actions = [action]
            else:
                actions = np.vstack((actions, action))

            if rewards is None:
                rewards = [reward]
            else:
                rewards = np.vstack((rewards, reward))

            if next_states is None:
                next_states = copy.deepcopy(next_state)
            else:
                next_states = np.vstack((next_states, next_state))

            if done_flags is None:
                done_flags = [done]
            else:
                done_flags = np.vstack((done_flags, done))

        yield (states, np.squeeze(actions), np.squeeze(rewards), next_states, np.squeeze(done_flags), a)

        count += size
def do_eval(eval_type):
    gen = process_eval_batch(size=1000, eval_type=eval_type)

    phys_q_ret = []
    actions_ret = []
    agent_q_ret = []
    actions_taken_ret = []
    agent_qsa_ret = []
    error_ret = 0

    for b in gen:
        states, actions, rewards, next_states, done_flags, _ = b

        # firstly get the chosen actions at the next timestep
        actions_from_q1 = sess.run(mainQN.predict, feed_dict={mainQN.state: next_states, mainQN.phase: 0})

        # Q values for the next timestep from target network, as part of the Double DQN update
        Q2 = sess.run(targetQN.q_output, feed_dict={targetQN.state: next_states, targetQN.phase: 0})

        # handles the case when a trajectory is finished
        end_multiplier = 1 - done_flags

        # target Q value using Q values from target, and actions from main
        double_q_value = Q2[range(len(Q2)), actions_from_q1]

        # definition of target Q
        targetQ = rewards + (gamma * double_q_value * end_multiplier)

        # get the output q's, actions, and loss
        q_output, actions_taken, abs_error = sess.run([mainQN.q_output, mainQN.predict, mainQN.abs_error],                                                       feed_dict={mainQN.state: states,
                                                                 mainQN.targetQ: targetQ,
                                                                 mainQN.actions: actions,
                                                                 mainQN.phase: False})

        # return the relevant q values and actions
        phys_q = q_output[range(len(q_output)), actions]
        agent_q = q_output[range(len(q_output)), actions_taken]
        error = np.mean(abs_error)

        #       update the return vals
        phys_q_ret.extend(phys_q)
        actions_ret.extend(actions)
        agent_qsa_ret.extend(q_output)  # qsa
        agent_q_ret.extend(agent_q)  # q
        actions_taken_ret.extend(actions_taken)  # a
        error_ret += error

    return agent_qsa_ret, phys_q_ret, actions_ret, agent_q_ret, actions_taken_ret, error_ret
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # Don't use all GPUs
config.allow_soft_placement = True  # Enable manual control
def do_save_results():
    # get the chosen actions for the train, val, and test set when training is complete.
    agent_qsa_train, _, _, agent_q_train, agent_actions_train, _ = do_eval(eval_type='train')
    agent_qsa_val, _, _, agent_q_val, agent_actions_val, _ = do_eval(eval_type='val')
    agent_qsa_test, _, _, agent_q_test, agent_actions_test, _ = do_eval(eval_type='test')
    print("length IS ", len(agent_actions_train))

    # save everything for later - they're used in policy evaluation and when generating plots
    with open(save_dir + 'dqn_normal_actions_train.p', 'wb') as f:
        pickle.dump(agent_actions_train, f)
    with open(save_dir + 'dqn_normal_actions_val.p', 'wb') as f:
        pickle.dump(agent_actions_val, f)
    with open(save_dir + 'dqn_normal_actions_test.p', 'wb') as f:
        pickle.dump(agent_actions_test, f)

    with open(save_dir + 'dqn_normal_q_train.p', 'wb') as f:
        pickle.dump(agent_q_train, f)
    with open(save_dir + 'dqn_normal_q_val.p', 'wb') as f:
        pickle.dump(agent_q_val, f)
    with open(save_dir + 'dqn_normal_q_test.p', 'wb') as f:
        pickle.dump(agent_q_test, f)

    with open(save_dir + 'dqn_normal_qsa_train.p', 'wb') as f:
        pickle.dump(agent_qsa_train, f)
    with open(save_dir + 'dqn_normal_qsa_val.p', 'wb') as f:
        pickle.dump(agent_qsa_val, f)
    with open(save_dir + 'dqn_normal_qsa_test.p', 'wb') as f:
        pickle.dump(agent_qsa_test, f)

    return


# The main training loop is here

tf.reset_default_graph()
mainQN = Qnetwork()
targetQN = Qnetwork()
av_q_list = []
saver = tf.train.Saver(tf.global_variables())
init = tf.global_variables_initializer()
trainables = tf.trainable_variables()
target_ops = update_target_graph(trainables, tau)

# Make a path for our model to be saved in.
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

with tf.Session(config=config) as sess:
    if load_model == True:
        print('Trying to load model...')
        try:
            restorer = tf.train.import_meta_graph(save_path + '.meta')
            restorer.restore(sess, tf.train.latest_checkpoint(save_dir))
            print("Model restored")
        except IOError:
            print("No previous model found, running default init")
            sess.run(init)
        try:
            per_weights = pickle.load(open(save_dir + "per_weights.p", "rb"))
            imp_weights = pickle.load(open(save_dir + "imp_weights.p", "rb"))

            # the PER weights, governing probability of sampling, and importance sampling
            # weights for use in the gradient descent updates
            df['prob'] = per_weights
            df['imp_weight'] = imp_weights
            print("PER and Importance weights restored")
        except IOError:
            print("No PER weights found - default being used for PER and importance sampling")
    else:
        print("Running default init")
        sess.run(init)
    print("Init done")

    net_loss = 0.0
    for i in tqdm(range(num_steps)):
        if save_results:
            print("Calling do save results")
            do_save_results()
            break

        states, actions, rewards, next_states, done_flags, sampled_df = process_train_batch(batch_size)
        actions_from_q1 = sess.run(mainQN.predict, feed_dict={mainQN.state: next_states, mainQN.phase: 1})

        cur_act = sess.run(mainQN.predict, feed_dict={mainQN.state: states, mainQN.phase: 1})

        Q2 = sess.run(targetQN.q_output, feed_dict={targetQN.state: next_states, targetQN.phase: 1})

        end_multiplier = 1 - done_flags

        double_q_value = Q2[range(len(Q2)), actions_from_q1]

        double_q_value[double_q_value > REWARD_THRESHOLD] = REWARD_THRESHOLD
        double_q_value[double_q_value < -REWARD_THRESHOLD] = -REWARD_THRESHOLD

        targetQ = rewards + (gamma * double_q_value * end_multiplier)

        # Calculate the importance sampling weights for PER
        imp_sampling_weights = np.array(sampled_df['imp_weight'] / float(max(df['imp_weight'])))
        imp_sampling_weights[np.isnan(imp_sampling_weights)] = 1
        imp_sampling_weights[imp_sampling_weights <= 0.001] = 0.001

        # Train with the batch
        _, loss, error = sess.run([mainQN.update_model, mainQN.loss, mainQN.abs_error],                                   feed_dict={mainQN.state: states,
                                             mainQN.targetQ: targetQ,
                                             mainQN.actions: actions,
                                             mainQN.phase: True,
                                             mainQN.imp_weights: imp_sampling_weights})

        update_target(target_ops, sess)

        net_loss += sum(error)

        # Set the selection weight/prob to the abs prediction error and update the importance sampling weight
        new_weights = pow((error + per_epsilon), per_alpha)
        df.loc[df.index.isin(sampled_df.index), 'prob'] = new_weights
        temp = 1.0 / new_weights
        df.loc[df.index.isin(sampled_df.index), 'imp_weight'] = pow(((1.0 / len(df)) * temp), beta_start)

        if i % 1000 == 0 and i > 0:
            saver.save(sess, save_path)
            print("Saved Model, step is " + str(i))

            av_loss = net_loss / (1000.0 * batch_size)
            print("Average loss is ", av_loss)
            net_loss = 0.0

            print("Saving PER and importance weights")
            with open(save_dir + 'per_weights.p', 'wb') as f:
                pickle.dump(df['prob'], f)
            with open(save_dir + 'imp_weights.p', 'wb') as f:
                pickle.dump(df['imp_weight'], f)

        if i % 70000 == 0 and i > 0:
            print("physactions ", actions)
            print("chosen actions ", cur_act)
            # run an evaluation on the validation set
            _, phys_q, phys_actions, agent_q, agent_actions, mean_abs_error = do_eval(eval_type='val')
            print(mean_abs_error)
            print(np.mean(phys_q))
            print(np.mean(agent_q))
    #             if (i % 5000==0) and i > 0:
    #                 print ("Saving results")
            do_save_results()
    do_save_results()



# https://github.com/DMU-XMU/Episodic-Memory-assisted-Approach-for-Sepsis-Treatment/blob/main/continuous/D3QN.py

hidden_1_size = 128
hidden_2_size = 128
# tf.compat.v1.disable_eager_execution()

#  Q-network uses Leaky ReLU activation
class Qnetwork():
    def __init__(self):
        self.phase = tf.compat.v1.placeholder(tf.bool)

        self.num_actions = 25

        self.input_size = len(state_features)

        self.state = tf.compat.v1.placeholder(tf.float32, shape=[None, self.input_size], name="input_state")

        self.fc_1 = tf.compat.v1.layers.dense (self.state, hidden_1_size, activation=None)
#         self.fc_1_bn = tf.compat.v1.layers.batch_norm(self.fc_1, center=True, scale=True, is_training=self.phase)
#         self.fc_1_ac = tf.maximum(self.fc_1_bn, self.fc_1_bn * 0.01)
#         self.fc_2 = tf.compat.v1.layers.dense (self.fc_1_ac, hidden_2_size, activation=None)
        self.fc_2 = tf.compat.v1.layers.dense (self.fc_1, hidden_2_size, activation=None)
#         self.fc_2_bn = tf.compat.v1.layers.batch_norm(self.fc_2, center=True, scale=True, is_training=self.phase)
#         self.fc_2_ac = tf.maximum(self.fc_2_bn, self.fc_2_bn * 0.01)

        # advantage and value streams
#     self.streamA, self.streamV = tf.split(self.fc_2_ac, 2, axis=1)
        self.streamA, self.streamV = tf.split(self.fc_2, 2, axis=1)
        self.AW = tf.Variable(tf.random.normal([hidden_2_size // 2, self.num_actions]))
        self.VW = tf.Variable(tf.random.normal([hidden_2_size // 2, 1]))
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)

        # Then combine them together to get our final Q-values.
        self.q_output = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keepdims=True))

        self.predict = tf.argmax(self.q_output, 1, name='predict')  # vector of length batch size

        # Below we obtain the loss by taking the sum of squares difference between the target and predicted Q values.
        self.targetQ = tf.compat.v1.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.compat.v1.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, self.num_actions, dtype=tf.float32)

        # Importance sampling weights for PER, used in network update
        self.imp_weights = tf.compat.v1.placeholder(shape=[None], dtype=tf.float32)

        # select the Q values for the actions that would be selected
        self.Q = tf.reduce_sum(tf.multiply(self.q_output, self.actions_onehot))  # batch size x 1 vector

        # regularisation penalises the network when it produces rewards that are above the
        # reward threshold, to ensure reasonable Q-value predictions
        self.reg_vector = tf.maximum(tf.abs(self.Q) - REWARD_THRESHOLD, 0)
        self.reg_term = tf.reduce_sum(self.reg_vector)

        self.abs_error = tf.abs(self.targetQ - self.Q)

        self.td_error = tf.square(self.targetQ - self.Q)

        # below is the loss when we are not using PER
        self.old_loss = tf.reduce_mean(self.td_error)

        # as in the paper, to get PER loss we weight the squared error by the importance weights
        self.per_error = tf.multiply(self.td_error, self.imp_weights)

        # total loss is a sum of PER loss and the regularisation term
        if per_flag:
            self.loss = tf.reduce_mean(self.per_error) + reg_lambda * self.reg_term
        else:
            self.loss = self.old_loss + reg_lambda * self.reg_term

        self.trainer = tf.optimizers.Adam(learning_rate=0.0001)
        self.update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        

#https://stackoverflow.com/questions/68879963/valueerror-tape-is-required-when-a-tensor-loss-is-passed

        W = tf.Variable(tf.random.normal([1]), trainable = True, name='weight')
        b = tf.Variable(tf.random.normal([1]), trainable = True, name='bias')
        
        #         cost =  lambda :tf.reduce_mean(tf.square(tx*tw+tb-ty)) #https://datascience.stackexchange.com/questions/77151/no-gradients-provided-for-any-variable-variable0-variable0
        optimizer = tf.keras.optimizers.SGD(0.01)
        cost = self.loss
        self.update_model = optimizer.minimize(cost, var_list = [W, b],tape=tf.GradientTape())
        
        
    
def update_target_graph(tf_vars,tau):
    total_vars = len(tf_vars)
    op_holder = []
    for idx,var in enumerate(tf_vars[0:int(total_vars/2)]):
        op_holder.append(tf_vars[idx+int(total_vars/2)].assign((var.value()*tau) + ((1-tau)*tf_vars[idx+int(total_vars/2)].value())))
    return op_holder
def update_target(op_holder,sess):
    for op in op_holder:
        sess.run(op)


def process_train_batch(size):
    if per_flag:
        # uses prioritised exp replay
        a = df.sample(n=size, weights=df['prob'])
    else:
        a = df.sample(n=size)
    states = None
    actions = None
    rewards = None
    next_states = None
    done_flags = None
    for i in a.index:
        cur_state = a.loc[i, state_features]
        iv = int(a.loc[i, 'iv_input'])
        vaso = int(a.loc[i, 'vaso_input'])
        action = action_map[iv, vaso]
        reward = a.loc[i, 'reward']

        if clip_reward:
            if reward > 1: reward = 1
            if reward < -1: reward = -1

        if i != df.index[-1]:
            # if not terminal step in trajectory
            if df.loc[i, 'icustayid'] == df.loc[i + 1, 'icustayid']:
                next_state = df.loc[i + 1, state_features]
                done = 0
            else:
                # trajectory is finished
                next_state = np.zeros(len(cur_state))
                done = 1
        else:
            # last entry in df is the final state of that trajectory
            next_state = np.zeros(len(cur_state))
            done = 1

        if states is None:
            states = copy.deepcopy(cur_state)
        else:
            states = np.vstack((states, cur_state))

        if actions is None:
            actions = [action]
        else:
            actions = np.vstack((actions, action))

        if rewards is None:
            rewards = [reward]
        else:
            rewards = np.vstack((rewards, reward))

        if next_states is None:
            next_states = copy.deepcopy(next_state)
        else:
            next_states = np.vstack((next_states, next_state))

        if done_flags is None:
            done_flags = [done]
        else:
            done_flags = np.vstack((done_flags, done))

    return (states, np.squeeze(actions), np.squeeze(rewards), next_states, np.squeeze(done_flags), a)
# extract chunks of length size from the relevant dataframe, and yield these to the caller

def process_eval_batch(size, eval_type=None):
    if eval_type is None:
        raise Exception('Provide eval_type to process_eval_batch')
    elif eval_type == 'train':
        a = df.copy()
    elif eval_type == 'val':
        a = val_df.copy()
    elif eval_type == 'test':
        a = test_df.copy()
    else:
        raise Exception('Unknown eval_type')
    count = 0
    while count < len(a.index):
        states = None
        actions = None
        rewards = None
        next_states = None
        done_flags = None

        start_idx = count
        end_idx = min(len(a.index), count + size)
        segment = a.index[start_idx:end_idx]

        for i in segment:
            cur_state = a.loc[i, state_features]
            iv = int(a.loc[i, 'iv_input'])
            vaso = int(a.loc[i, 'vaso_input'])
            action = action_map[iv, vaso]
            reward = a.loc[i, 'reward']

            if clip_reward:
                if reward > 1: reward = 1
                if reward < -1: reward = -1

            if i != a.index[-1]:
                # if not terminal step in trajectory
                if a.loc[i, 'icustayid'] == a.loc[i + 1, 'icustayid']:
                    next_state = a.loc[i + 1, state_features]
                    done = 0
                else:
                    # trajectory is finished
                    next_state = np.zeros(len(cur_state))
                    done = 1
            else:
                # last entry in df is the final state of that trajectory
                next_state = np.zeros(len(cur_state))
                done = 1

            if states is None:
                states = copy.deepcopy(cur_state)
            else:
                states = np.vstack((states, cur_state))

            if actions is None:
                actions = [action]
            else:
                actions = np.vstack((actions, action))

            if rewards is None:
                rewards = [reward]
            else:
                rewards = np.vstack((rewards, reward))

            if next_states is None:
                next_states = copy.deepcopy(next_state)
            else:
                next_states = np.vstack((next_states, next_state))

            if done_flags is None:
                done_flags = [done]
            else:
                done_flags = np.vstack((done_flags, done))

        yield (states, np.squeeze(actions), np.squeeze(rewards), next_states, np.squeeze(done_flags), a)

        count += size
        
def do_eval(eval_type):
    gen = process_eval_batch(size=1000, eval_type=eval_type)

    phys_q_ret = []
    actions_ret = []
    agent_q_ret = []
    actions_taken_ret = []
    agent_qsa_ret = []
    error_ret = 0

    for b in gen:
        states, actions, rewards, next_states, done_flags, _ = b

        # firstly get the chosen actions at the next timestep
        actions_from_q1 = sess.run(mainQN.predict, feed_dict={mainQN.state: next_states, mainQN.phase: 0})

        # Q values for the next timestep from target network, as part of the Double DQN update
        Q2 = sess.run(targetQN.q_output, feed_dict={targetQN.state: next_states, targetQN.phase: 0})

        # handles the case when a trajectory is finished
        end_multiplier = 1 - done_flags

        # target Q value using Q values from target, and actions from main
        double_q_value = Q2[range(len(Q2)), actions_from_q1]

        # definition of target Q
        targetQ = rewards + (gamma * double_q_value * end_multiplier)

        # get the output q's, actions, and loss
        q_output, actions_taken, abs_error = sess.run([mainQN.q_output, mainQN.predict, mainQN.abs_error],                                                       feed_dict={mainQN.state: states,
                                                                 mainQN.targetQ: targetQ,
                                                                 mainQN.actions: actions,
                                                                 mainQN.phase: False})

        # return the relevant q values and actions
        phys_q = q_output[range(len(q_output)), actions]
        agent_q = q_output[range(len(q_output)), actions_taken]
        error = np.mean(abs_error)

        #       update the return vals
        phys_q_ret.extend(phys_q)
        actions_ret.extend(actions)
        agent_qsa_ret.extend(q_output)  # qsa
        agent_q_ret.extend(agent_q)  # q
        actions_taken_ret.extend(actions_taken)  # a
        error_ret += error

    return agent_qsa_ret, phys_q_ret, actions_ret, agent_q_ret, actions_taken_ret, error_ret
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # Don't use all GPUs
config.allow_soft_placement = True  # Enable manual control
def do_save_results():
    # get the chosen actions for the train, val, and test set when training is complete.
    agent_qsa_train, _, _, agent_q_train, agent_actions_train, _ = do_eval(eval_type='train')
    agent_qsa_val, _, _, agent_q_val, agent_actions_val, _ = do_eval(eval_type='val')
    agent_qsa_test, _, _, agent_q_test, agent_actions_test, _ = do_eval(eval_type='test')
    print("length IS ", len(agent_actions_train))

    # save everything for later - they're used in policy evaluation and when generating plots
    with open(save_dir + 'dqn_normal_actions_train.p', 'wb') as f:
        pickle.dump(agent_actions_train, f)
    with open(save_dir + 'dqn_normal_actions_val.p', 'wb') as f:
        pickle.dump(agent_actions_val, f)
    with open(save_dir + 'dqn_normal_actions_test.p', 'wb') as f:
        pickle.dump(agent_actions_test, f)

    with open(save_dir + 'dqn_normal_q_train.p', 'wb') as f:
        pickle.dump(agent_q_train, f)
    with open(save_dir + 'dqn_normal_q_val.p', 'wb') as f:
        pickle.dump(agent_q_val, f)
    with open(save_dir + 'dqn_normal_q_test.p', 'wb') as f:
        pickle.dump(agent_q_test, f)

    with open(save_dir + 'dqn_normal_qsa_train.p', 'wb') as f:
        pickle.dump(agent_qsa_train, f)
    with open(save_dir + 'dqn_normal_qsa_val.p', 'wb') as f:
        pickle.dump(agent_qsa_val, f)
    with open(save_dir + 'dqn_normal_qsa_test.p', 'wb') as f:
        pickle.dump(agent_qsa_test, f)

    return


# https://github.com/DMU-XMU/Episodic-Memory-assisted-Approach-for-Sepsis-Treatment/blob/main/continuous/D3QN.py

# The main training loop is here
tf.compat.v1.disable_eager_execution()

tf.compat.v1.reset_default_graph()
mainQN = Qnetwork()
targetQN = Qnetwork()
av_q_list = []
# saver = tf.compat.v1.train.Saver(tf.global_variables())
init = tf.compat.v1.global_variables_initializer()
trainables = tf.compat.v1.trainable_variables()
target_ops = update_target_graph(trainables, tau)

# Make a path for our model to be saved in.
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

with tf.compat.v1.Session(config=config) as sess:
    if load_model == True:
        print('Trying to load model...')
        try:
            restorer = tf.train.import_meta_graph(save_path + '.meta')
            restorer.restore(sess, tf.train.latest_checkpoint(save_dir))
            print("Model restored")
        except IOError:
            print("No previous model found, running default init")
            sess.run(init)
        try:
            per_weights = pickle.load(open(save_dir + "per_weights.p", "rb"))
            imp_weights = pickle.load(open(save_dir + "imp_weights.p", "rb"))

            # the PER weights, governing probability of sampling, and importance sampling
            # weights for use in the gradient descent updates
            df['prob'] = per_weights
            df['imp_weight'] = imp_weights
            print("PER and Importance weights restored")
        except IOError:
            print("No PER weights found - default being used for PER and importance sampling")
    else:
        print("Running default init")
        sess.run(init)
    print("Init done")

    net_loss = 0.0
    for i in tqdm(range(num_steps)):
        if save_results:
            print("Calling do save results")
            do_save_results()
            break

        states, actions, rewards, next_states, done_flags, sampled_df = process_train_batch(batch_size)
        actions_from_q1 = sess.run(mainQN.predict, feed_dict={mainQN.state: next_states, mainQN.phase: 1})

        cur_act = sess.run(mainQN.predict, feed_dict={mainQN.state: states, mainQN.phase: 1})

        Q2 = sess.run(targetQN.q_output, feed_dict={targetQN.state: next_states, targetQN.phase: 1})

        end_multiplier = 1 - done_flags

        double_q_value = Q2[range(len(Q2)), actions_from_q1]

        double_q_value[double_q_value > REWARD_THRESHOLD] = REWARD_THRESHOLD
        double_q_value[double_q_value < -REWARD_THRESHOLD] = -REWARD_THRESHOLD

        targetQ = rewards + (gamma * double_q_value * end_multiplier)

        # Calculate the importance sampling weights for PER
        imp_sampling_weights = np.array(sampled_df['imp_weight'] / float(max(df['imp_weight'])))
        imp_sampling_weights[np.isnan(imp_sampling_weights)] = 1
        imp_sampling_weights[imp_sampling_weights <= 0.001] = 0.001

        # Train with the batch
        _, loss, error = sess.run([mainQN.update_model, mainQN.loss, mainQN.abs_error],                                   feed_dict={mainQN.state: states,
                                             mainQN.targetQ: targetQ,
                                             mainQN.actions: actions,
                                             mainQN.phase: True,
                                             mainQN.imp_weights: imp_sampling_weights})

        update_target(target_ops, sess)

        net_loss += sum(error)

        # Set the selection weight/prob to the abs prediction error and update the importance sampling weight
        new_weights = pow((error + per_epsilon), per_alpha)
        df.loc[df.index.isin(sampled_df.index), 'prob'] = new_weights
        temp = 1.0 / new_weights
        df.loc[df.index.isin(sampled_df.index), 'imp_weight'] = pow(((1.0 / len(df)) * temp), beta_start)

        if i % 1000 == 0 and i > 0:
            saver.save(sess, save_path)
            print("Saved Model, step is " + str(i))

            av_loss = net_loss / (1000.0 * batch_size)
            print("Average loss is ", av_loss)
            net_loss = 0.0

            print("Saving PER and importance weights")
            with open(save_dir + 'per_weights.p', 'wb') as f:
                pickle.dump(df['prob'], f)
            with open(save_dir + 'imp_weights.p', 'wb') as f:
                pickle.dump(df['imp_weight'], f)

        if i % 70000 == 0 and i > 0:
            print("physactions ", actions)
            print("chosen actions ", cur_act)
            # run an evaluation on the validation set
            _, phys_q, phys_actions, agent_q, agent_actions, mean_abs_error = do_eval(eval_type='val')
            print(mean_abs_error)
            print(np.mean(phys_q))
            print(np.mean(agent_q))
    #             if (i % 5000==0) and i > 0:
    #                 print ("Saving results")
            do_save_results()
    do_save_results()

# https://github.com/hernanborre/sepsis-reinforcement-learning/blob/master/continuous/q_network.ipynb
#-all same
# https://github.com/aniruddhraghu/sepsisrl/blob/master/continuous/q_network.ipynb



dfTr=pd.read_csv('sepsis_final_data_RAW_withTimes.csv')
dfTr
