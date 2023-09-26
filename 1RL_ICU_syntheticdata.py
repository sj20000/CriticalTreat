
"""1RL_ICU_syntheticdata.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/
"""

! pip install ydata-synthetic
!pip install numpy==1.19.3

#originally based on:
#https://papers.nips.cc/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html
#https://github.com/jsyoon0823/TimeGAN

#https://colab.research.google.com/github/ydataai/ydata-synthetic/blob/master/examples/timeseries/TimeGAN_Synthetic_stock_data.ipynb#scrollTo=aEIlLGWpjtWL

#https://towardsdatascience.com/synthetic-time-series-data-a-gan-approach-869a984f2239

#1. Importing the required libs for the exercise

from os import path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ydata_synthetic.synthesizers import ModelParameters
from ydata_synthetic.preprocessing.timeseries import processed_stock
from ydata_synthetic.synthesizers.timeseries import TimeGAN

"""Define Model hyperparameters
Networks:

Generator
Discriminator
Embedder
Recovery Network
TimeGAN is a Generative model based on RNN networks. In this package the implemented version follows a very simple architecture that is shared by the four elements of the GAN.

Similarly to other parameters, the architectures of each element should be optimized and tailored to the data.
"""

#2. Define Model Hyperparameters
#https://github.com/archity/synthetic-data-gan/blob/main/timeseries-data/energy-data-synthesize.ipynb

#Specific to TimeGANs
# seq_len=24
seq_len=1
# n_seq = 43
n_seq = 54
hidden_dim=24
gamma=1

noise_dim = 32
dim = 128
batch_size = 128

log_step = 100
learning_rate = 5e-4

gan_args = ModelParameters(batch_size=batch_size,
                           lr=learning_rate,
                           noise_dim=noise_dim,
                           layers_dim=dim)

# original_dataset = pd.read_csv('LabMonitorReward.csv')
# original_dataset.head()

original_dataset = pd.read_csv('dfLMF.csv')
original_dataset.head()

original_dataset.drop(columns=original_dataset.columns[0], axis=1, inplace=True)
original_dataset.head()

original_dataset.dtypes

original_dataset.shape

#convert start_date to DateTime format
original_dataset['DateTime'] = pd.to_datetime(original_dataset['DateTime'])

#https://towardsdatascience.com/modeling-and-generating-time-series-data-using-timegan-29c00804f54d
try:
    ICU_df = original_dataset.set_index('DateTime').sort_index()
except:
    ICU_df=original_dataset
#3. Read the Input data

from ydata_synthetic.preprocessing.timeseries.utils import real_data_loading
# Data transformations to be applied prior to be used with the synthesizer model
ICU_data = real_data_loading(ICU_df.values, seq_len=seq_len)

print(len(ICU_data), ICU_data[0].shape)

ICU_df

ICU_df.columns

# !pip install numpy==1.19.3
# (to avoid the error: - with synth.train(ICU_data, train_steps=500)
# NotImplementedError: Cannot convert a symbolic Tensor (Embedder/GRU_1/strided_slice:0) to a numpy array. This error may indicate that you're trying to pass a Tensor to a NumPy call, which is not supported
#  )

#https://towardsdatascience.com/modeling-and-generating-time-series-data-using-timegan-29c00804f54d
#https://github.com/archity/synthetic-data-gan/blob/main/timeseries-data/energy-data-synthesize.ipynb

from ydata_synthetic.synthesizers.timeseries import TimeGAN
#4. Training the TimeGAN synthetizer
synth = TimeGAN(model_parameters=gan_args, hidden_dim=hidden_dim, seq_len=seq_len, n_seq=n_seq, gamma=1)
synth.train(ICU_data, train_steps=500)
synth.save('synth_icu.pkl')

#5. Generating Synthetic ICU Data
synth_data = synth.sample(len(ICU_data))

len(synth_data)

len(ICU_data)

ICU_data[0].shape

synth_data[0].shape

(ICU_data)

import csv
for i in range(len(synth_data)):
    with open("synth_dataA1.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerows(synth_data[i])

from google.colab import files
files.download("synth_dataA1.csv")

for i in range(len(ICU_data)):
    with open("ICU_dataA1.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerows(ICU_data[i])

from google.colab import files
files.download("ICU_dataA1.csv")

#save the original ICU_data array and synth_data array to csv files:

#https://stackoverflow.com/questions/21899683/saving-arrays-in-an-array-to-csv-file-using-python
import csv

# with open("ICU_array.csv", "a") as f:
#     writer = csv.writer(f)
#     writer.writerows(ICU_data)


# with open("synth_array.csv", "a") as f:
#     writer = csv.writer(f)
#     writer.writerows(synth_data)

# #https://pythonguides.com/python-write-a-list-to-csv/
# with open("ICU_dataA.csv", "w",  lineterminator = '\n' ) as f:
#    writer = csv.writer(f)
#    writer.writerows(ICU_data)

# from google.colab import files
# files.download("ICU_dataA.csv")

# np.savetxt('ICU_dataA.csv', ICU_data, delimiter=',')
# np.savetxt('synth_dataA.csv', synth_data, delimiter=',')

# #https://stackoverflow.com/questions/49394737/exporting-data-from-google-colab-to-local-machine
# from google.colab import files
# files.download("ICU_array.csv")

# files.download("synth_array.csv")

# import numpy as np
# import pandas as pd

# dfICU_A = pd.read_csv('ICU_array.csv')
# dfICU_A.head()

# ICU_array = dfICU_A.to_numpy()
# ICU_array



# import numpy as np
# import pandas as pd

# dfICU_A = pd.read_csv('ICU_dataA.csv')
# dfICU_A.head()

# ICU_arrayA = dfICU_A.to_numpy()
# ICU_arrayA

# dfSynth_A = pd.read_csv('synth_array.csv')
# dfSynth_A.head()

# Synth_array = dfSynth_A.to_numpy()
# Synth_array

len(ICU_data)

n_seq

seq_len

hidden_dim

#https://github.com/archity/synthetic-data-gan/blob/main/timeseries-data/energy-data-synthesize.ipynb

cols = ["heartrate", "meanpressure", "osat", "resprate",'ALB',	'ALT',	'AST',	'BICARBART',	'BILITOTAL',\
        'BUN',  'CA',  'CL','CO2','CR' , 'FIB' , 'GLU', 'HCT' ,'HGB'  ,'IONCA','K','MCH' ,'MCHC','MCV', \
        'MG' ,'NA','NH3','O2SATART','PCO2ART','PHART','PHOS', 'PLT' ,'PO2ART','PT','PT1' ,'PT2' , \
        'PTT' ,'PTT1',  'PTT2','RBC', 'TP','TRIG','WBC' ]

# Plotting some generated samples. Both Synthetic and Original data are still standardized with values between [0, 1]
fig, axes = plt.subplots(nrows=7, ncols=6, figsize=(15, 10))
axes=axes.flatten()

time = list(range(1,42))
obs = np.random.randint(len(ICU_data))

for j, col in enumerate(cols):
    df = pd.DataFrame({'Real': ICU_data[obs][:, j],
                   'Synthetic': synth_data[obs][:, j]})
    df.plot(ax=axes[j],
            title = col,
            secondary_y='Synthetic data', style=['-', '--'])
fig.tight_layout()

plt.show()


import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# Necessary Packages
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_absolute_error

#https://github.com/jsyoon0823/TimeGAN/blob/master/utils.py
def extract_time (data):
  """Returns Maximum sequence length and each sequence length.

  Args:
    - data: original data

  Returns:
    - time: extracted time information
    - max_seq_len: maximum sequence length
  """
  time = list()
  max_seq_len = 0
  for i in range(len(data)):
    max_seq_len = max(max_seq_len, len(data[i][:,0]))
    time.append(len(data[i][:,0]))

  return time, max_seq_len

 #https://github.com/jsyoon0823/TimeGAN/blob/master/metrics/predictive_metrics.py
def predictive_score_metrics (ori_data, generated_data):
  """Report the performance of Post-hoc RNN one-step ahead prediction.

  Args:
    - ori_data: original data
    - generated_data: generated synthetic data

  Returns:
    - predictive_score: MAE of the predictions on the original data
  """
  # Initialization on the Graph
  tf.compat.v1.reset_default_graph()

  # Basic Parameters
  no, seq_len, dim = np.asarray(ori_data).shape

  # Set maximum sequence length and each sequence length
  ori_time, ori_max_seq_len = extract_time(ori_data)
  generated_time, generated_max_seq_len = extract_time(ori_data)
  max_seq_len = max([ori_max_seq_len, generated_max_seq_len])

  ## Builde a post-hoc RNN predictive network
  # Network parameters
  hidden_dim = int(dim/2)
  iterations = 5000
  batch_size = 128

  # Input place holders
  X = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len-1, dim-1], name = "myinput_x")
  T = tf.compat.v1.placeholder(tf.int32, [None], name = "myinput_t")
  Y = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len-1, 1], name = "myinput_y")

  # Predictor function
  def predictor (x, t):
    """Simple predictor function.

    Args:
      - x: time-series data
      - t: time information

    Returns:
      - y_hat: prediction
      - p_vars: predictor variables
    """
    with tf.compat.v1.variable_scope("predictor", reuse = tf.compat.v1.AUTO_REUSE) as vs:
      p_cell = tf.compat.v1.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh, name = 'p_cell')
      p_outputs, p_last_states = tf.compat.v1.nn.dynamic_rnn(p_cell, x, dtype=tf.float32, sequence_length = t)
      y_hat_logit = tf.compat.v1.layers.dense(p_outputs, 1, activation=None)  #https://stackoverflow.com/questions/58359881/tf-contrib-layers-fully-connected-in-tensorflow-2
      y_hat = tf.nn.sigmoid(y_hat_logit)
      p_vars = [v for v in tf.compat.v1.all_variables() if v.name.startswith(vs.name)]

    return y_hat, p_vars

  y_pred, p_vars = predictor(X, T)
  # Loss for the predictor
  p_loss = tf.compat.v1.losses.absolute_difference(Y, y_pred)
  # optimizer
  p_solver = tf.optimizers.Adam().minimize(p_loss, var_list = p_vars, tape=tf.GradientTape(persistent=False)) #https://stackoverflow.com/questions/63461478/tape-is-required-when-a-tensor-loss-is-passed

  ## Training
  # Session start
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  # Training using Synthetic dataset
  for itt in range(iterations):

    # Set mini-batch
    idx = np.random.permutation(len(generated_data))
    train_idx = idx[:batch_size]

    X_mb = list(generated_data[i][:-1,:(dim-1)] for i in train_idx)
    T_mb = list(generated_time[i]-1 for i in train_idx)
    Y_mb = list(np.reshape(generated_data[i][1:,(dim-1)],[len(generated_data[i][1:,(dim-1)]),1]) for i in train_idx)

    # Train predictor
    _, step_p_loss = sess.run([p_solver, p_loss], feed_dict={X: X_mb, T: T_mb, Y: Y_mb})

  ## Test the trained model on the original data
  idx = np.random.permutation(len(ori_data))
  train_idx = idx[:no]

  X_mb = list(ori_data[i][:-1,:(dim-1)] for i in train_idx)
  T_mb = list(ori_time[i]-1 for i in train_idx)
  Y_mb = list(np.reshape(ori_data[i][1:,(dim-1)], [len(ori_data[i][1:,(dim-1)]),1]) for i in train_idx)

  # Prediction
  pred_Y_curr = sess.run(y_pred, feed_dict={X: X_mb, T: T_mb})

  # Compute the performance in terms of MAE
  MAE_temp = 0
  for i in range(no):
    MAE_temp = MAE_temp + mean_absolute_error(Y_mb[i], pred_Y_curr[i,:,:])

  predictive_score = MAE_temp / no

  return predictive_score

np.asarray(ICU_data).shape

np.asarray(synth_data[0:2602]).shape

tf.compat.v1.reset_default_graph()

# Basic Parameters
no, seq_len, dim = np.asarray(ICU_data).shape

# Set maximum sequence length and each sequence length
ori_time, ori_max_seq_len = extract_time(ICU_data)
generated_time, generated_max_seq_len = extract_time(ICU_data)
max_seq_len = max([ori_max_seq_len, generated_max_seq_len])

## Builde a post-hoc RNN predictive network
# Network parameters
hidden_dim = int(dim/2)
iterations = 5000
batch_size = 128

# Input place holders
X = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len-1, dim-1], name = "myinput_x")
T = tf.compat.v1.placeholder(tf.int32, [None], name = "myinput_t")
Y = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len-1, 1], name = "myinput_y")

# Predictor function
def predictor (x, t):
  """Simple predictor function.

  Args:
    - x: time-series data
    - t: time information

  Returns:
    - y_hat: prediction
    - p_vars: predictor variables
  """
  with tf.compat.v1.variable_scope("predictor", reuse = tf.compat.v1.AUTO_REUSE) as vs:
    p_cell = tf.compat.v1.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh, name = 'p_cell')
    p_outputs, p_last_states = tf.compat.v1.nn.dynamic_rnn(p_cell, x, dtype=tf.float32, sequence_length = t)
    y_hat_logit = tf.compat.v1.layers.dense(p_outputs, 1, activation=None)  #https://stackoverflow.com/questions/58359881/tf-contrib-layers-fully-connected-in-tensorflow-2
    y_hat = tf.nn.sigmoid(y_hat_logit)
    p_vars = [v for v in tf.compat.v1.all_variables() if v.name.startswith(vs.name)]

  return y_hat, p_vars

y_pred, p_vars = predictor(X, T)
# Loss for the predictor
p_loss = tf.compat.v1.losses.absolute_difference(Y, y_pred)

# Optimizer  https://medium.com/practical-coding/from-minimize-to-tf-gradienttape-58a1aae6ce26
opt = tf.optimizers.Adam(learning_rate=0.01)
# Manually compute the gradient
# The scalar variable to minimize
x11 = tf.Variable(initial_value=0, name='x11', trainable=True, dtype=tf.float32)
with tf.GradientTape(persistent=False) as t:
    # Loss function
    y = (x11 - 4) ** 2
gradients = t.gradient(y, [x11])

# Apply the gradient
p_solver =opt.apply_gradients(zip(gradients, [x11]))

# optimizer
# p_solver = tf.optimizers.Adam().minimize(p_loss, var_list = p_vars, tape=tf.GradientTape(persistent=False))
#https://stackoverflow.com/questions/63461478/tape-is-required-when-a-tensor-loss-is-passed

## Training
# Session start
sess =  tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
synth_data1 = synth_data[0:2602]
# Training using Synthetic dataset
for itt in range(iterations):

  # Set mini-batch
  idx = np.random.permutation(len(synth_data1))
  train_idx = idx[:batch_size]

  X_mb = list(synth_data1[i][:-1,:(dim-1)] for i in train_idx)
  T_mb = list(generated_time[i]-1 for i in train_idx)
  Y_mb = list(np.reshape(synth_data1[i][1:,(dim-1)],[len(synth_data1[i][1:,(dim-1)]),1]) for i in train_idx)

  # Train predictor
  _, step_p_loss = sess.run([p_solver, p_loss], feed_dict={X: X_mb, T: T_mb, Y: Y_mb})

## Test the trained model on the original data
idx = np.random.permutation(len(ICU_data))
train_idx = idx[:no]

X_mb = list(ICU_data[i][:-1,:(dim-1)] for i in train_idx)
T_mb = list(ori_time[i]-1 for i in train_idx)
Y_mb = list(np.reshape(ICU_data[i][1:,(dim-1)], [len(ICU_data[i][1:,(dim-1)]),1]) for i in train_idx)

# Prediction
pred_Y_curr = sess.run(y_pred, feed_dict={X: X_mb, T: T_mb})

# Compute the performance in terms of MAE
MAE_temp = 0
for i in range(no):
  MAE_temp = MAE_temp + mean_absolute_error(Y_mb[i], pred_Y_curr[i,:,:])

predictive_score = MAE_temp / no

#https://github.com/jsyoon0823/TimeGAN/blob/master/main_timegan.py

metric_iteration =10
metric_results =dict()
# 2. Predictive score
predictive_score = list()

  # return ori_data, generated_data, metric_results

temp_pred = predictive_score_metrics(ICU_data, synth_data)

for tt in range(metric_iteration):
  temp_pred = predictive_score_metrics(ICU_data, synth_data[0:2602])
  predictive_score.append(temp_pred)

metric_results['predictive'] = np.mean(predictive_score)

print(metric_results)

for tt in range(metric_iteration):
  temp_pred = predictive_score_metrics(ICU_data, synth_data)

"""DO FROM HERE - since the data is already saved in the 2 csv files"""

import numpy as np
import pandas as pd

col_names =['VentMode', 'heartrate', 'meanpressure', 'sysPressure', 'diasPressure',
       'artheartrate', 'osat', 'FiO2', 'resprate', 'TidalVol', 'PIP',
       'AirwayP', 'PEEP', 'ALB', 'ALT', 'AST', 'BICARBART', 'BILITOTAL', 'BUN',
       'CA', 'CL', 'CO2', 'CR', 'FIB', 'GLU', 'HCT', 'HGB', 'IONCA', 'K',
       'MCH', 'MCHC', 'MCV', 'MG', 'NA', 'NH3', 'O2SATART', 'PCO2ART', 'PHART',
       'PHOS', 'PLT', 'PO2ART', 'PT', 'PT1', 'PT2', 'PTT', 'PTT1', 'PTT2',
       'RBC', 'TP', 'TRIG', 'WBC', 'Dopamine', 'Fentanyl', 'DopamineAction']
dfICU = pd.read_csv('ICU_dataA1.csv',names=col_names)
dfICU.head()

dfSyn = pd.read_csv('synth_dataA1.csv',names=col_names)
dfSyn.head()

len(dfICU)

len(dfSyn)

dfSyn1 = dfSyn[0:5204]
len(dfSyn1)

ICU_array_all = dfICU.iloc[:,1:8].to_numpy()
ICU_array_hr = dfICU['heartrate'].to_numpy()
ICU_array_hr



Syn_array_all = dfSyn1.iloc[:,1:8].to_numpy()
Syn_array_hr = dfSyn1['heartrate'].to_numpy()
Syn_array_hr

#Summarize the metrics here as a pandas dataframe    #correct
from sklearn.metrics import r2_score, mean_absolute_error,  mean_squared_error, mean_squared_log_error


# Dopamine_ICU_Arr = dfICU['DopamineAction'].to_numpy()
# Dopamine_Syn_Arr = dfSyn1['DopamineAction'].to_numpy()

# real_predictions = ts_real.predict(X_stock_test)
# synth_predictions = ts_synth.predict(X_stock_test)

metrics_dict = {'r2': [r2_score(Syn_array_all, ICU_array_all)],
                'MAE': [mean_absolute_error(Syn_array_all, ICU_array_all)],
                'MSE': [mean_squared_error(Syn_array_all, ICU_array_all)],
                'MRLE': [mean_squared_log_error(Syn_array_all, ICU_array_all)]}

results = pd.DataFrame(metrics_dict)

results



from tensorflow._api.v2.compat.v1 import disable_control_flow_v2

import matplotlib.pyplot as plt
ICU_array_alldata = dfICU.iloc[:,:].to_numpy()
Syn_array_alldata = dfSyn1.iloc[:,:].to_numpy()

# cols = ['heartrate', 'meanpressure', 'sysPressure', 'diasPressure', \
#        'artheartrate', 'osat', 'FiO2', 'resprate', 'TidalVol', 'PIP',  \
#        'AirwayP', 'PEEP', 'ALB', 'ALT', 'AST', 'BICARBART', 'BILITOTAL', 'BUN',         \
#        'CA', 'CL', 'CO2', 'CR', 'FIB', 'GLU', 'HCT', 'HGB', 'IONCA', 'K',  \
#        'MCH', 'MCHC', 'MCV', 'MG', 'NA', 'NH3', 'O2SATART', 'PCO2ART', 'PHART',  \
#        'PHOS', 'PLT', 'PO2ART', 'PT', 'PT1', 'PT2', 'PTT', 'PTT1', 'PTT2',  \
#        'RBC', 'TP', 'TRIG', 'WBC']


cols = ["heartrate", "meanpressure", "osat", "resprate",'ALB',	'ALT',	'AST',	'BICARBART',	'BILITOTAL',\
        'BUN',  'CA',  'CL','CO2','CR' , 'FIB' , 'GLU', 'HCT' ,'HGB'  ,'IONCA','K','MCH' ,'MCHC','MCV', \
        'MG' ,'NA','NH3','O2SATART','PCO2ART','PHART','PHOS', 'PLT' ,'PO2ART','PT','PT1' ,'PT2' , \
        'PTT' ,'PTT1',  'PTT2','RBC', 'TP','TRIG','WBC' ]

# Plotting some generated samples. Both Synthetic and Original data are still standardized with values between [0, 1]
fig, axes = plt.subplots(nrows=14, ncols=3, figsize=(15, 15))
axes=axes.flatten()

time = list(range(1,42))
obs = np.random.randint(len(dfICU))

for j, col in enumerate(cols):

  # dfICU.iloc[2714,: 39]
#   DF11=dfICU.iloc[2714,: 32]
# df11=pd.DataFrame(DF11)
# df11.reset_index(drop=True, inplace=True)
# df11.columns=['Real']
# df11
  if col == 'heartrate':
    dfI_hr=dfICU['heartrate']
    dfS_hr=dfSyn1['heartrate']
  elif col == 'meanpressure':
    dfI_mp=dfICU['meanpressure']
    dfS_mp=dfSyn1['meanpressure']
  elif col == 'osat':
    dfI_os=dfICU['osat']
    dfS_os=dfSyn1['osat']
  elif col == 'resprate':
    dfI_rp=dfICU['resprate']
    dfS_rp=dfSyn1['resprate']
  elif col == 'ALB':
    dfI_ALB=dfICU['ALB']
    dfS_ALB=dfSyn1['ALB']
  elif col == 'ALT':
    dfI_ALT=dfICU['ALT']
    dfS_ALT=dfSyn1['ALT']
  elif col == 'AST':
    dfI_AST=dfICU['AST']
    dfS_AST=dfSyn1['AST']
  elif col == 'BICARBART':
    dfI_BICARBART=dfICU['BICARBART']
    dfS_BICARBART=dfSyn1['BICARBART']
  elif col == 'BILITOTAL':
    dfI_BILITOTAL=dfICU['BILITOTAL']
    dfS_BILITOTAL=dfSyn1['BILITOTAL']
  elif col == 'BUN':
    dfI_BUN=dfICU['BUN']
    dfS_BUN=dfSyn1['BUN']
  elif col == 'CA':
    dfI_CA=dfICU['CA']
    dfS_CA=dfSyn1['CA']
  elif col == 'CL':
    dfI_CL=dfICU['CL']
    dfS_CL=dfSyn1['CL']
  elif col == 'CO2':
    dfI_CO2=dfICU['CO2']
    dfS_CO2=dfSyn1['CO2']
  elif col == 'CR':
    dfI_CR=dfICU['CR']
    dfS_CR=dfSyn1['CR']

df_hr = pd.concat([dfI_hr, dfS_hr], axis=1, join='inner')
# df_hr.plot( style=['-', '--'])

df_mp = pd.concat([dfI_mp, dfS_mp], axis=1, join='inner')
df_os = pd.concat([dfI_os, dfS_os], axis=1, join='inner')
df_rp = pd.concat([dfI_rp, dfS_rp], axis=1, join='inner')


df_ALB = pd.concat([dfI_ALB, dfS_ALB], axis=1, join='inner')
df_ALT = pd.concat([dfI_ALT, dfS_ALT], axis=1, join='inner')
df_AST = pd.concat([dfI_AST, dfS_AST], axis=1, join='inner')


df_BICARBART = pd.concat([dfI_BICARBART, dfS_BICARBART], axis=1, join='inner')
df_BILITOTAL = pd.concat([dfI_BILITOTAL, dfS_BILITOTAL], axis=1, join='inner')
df_BUN = pd.concat([dfI_BUN, dfS_BUN], axis=1, join='inner')
df_CA = pd.concat([dfI_CA, dfS_CA], axis=1, join='inner')
df_CL = pd.concat([dfI_CL, dfS_CL], axis=1, join='inner')
df_CO2 = pd.concat([dfI_CO2, dfS_CO2], axis=1, join='inner')
df_CR = pd.concat([dfI_CR, dfS_CR], axis=1, join='inner')



df = pd.DataFrame({'Real': dfI_hr, 'Synthetic': dfS_hr})
df.plot( style=['-', '--'])

  # df1=dfICU.iloc[obs,: j]
  # df11=pd.DataFrame(df1)
  # df11.reset_index(drop=True, inplace=True)
  # df11.columns=['Real']

  # df2=dfSyn1.iloc[obs,: j]
  # df22=pd.DataFrame(df2)
  # df22.reset_index(drop=True, inplace=True)
  # df22.columns=['Synthetic']


# df_RS = pd.concat([df11, df22], axis=1, join='inner')
# display(df_RS)
# for i in range(41):
# for k, col in enumerate(cols):
#   df_plot=df_RS.iloc[k,:]
  # df = pd.DataFrame({'Real': dfICU.iloc[obs,: j], 'Synthetic': len(dfSyn1.iloc[obs,: j]})
  # df_plot.plot(ax=axes[k],title = col, secondary_y='Synthetic', style=['-', '--'])
  # df_plot.plot('Real',style=['-', '--'])
  # df_plot.set_index('Real').plot()
fig.tight_layout()

plt.show()

# df_hr.columns=['Real','Synthetic']
# df_hr.head()

df_hr.columns=['Real','Synthetic']
df_mp.columns=['Real','Synthetic']
df_os.columns=['Real','Synthetic']
df_rp.columns=['Real','Synthetic']
df_ALB.columns=['Real','Synthetic']
df_ALT.columns=['Real','Synthetic']
df_AST.columns=['Real','Synthetic']
df_BICARBART.columns=['Real','Synthetic']
df_BILITOTAL.columns=['Real','Synthetic']
df_BUN.columns=['Real','Synthetic']
df_CA.columns=['Real','Synthetic']
df_CL.columns=['Real','Synthetic']
df_CO2.columns=['Real','Synthetic']
df_CR.columns=['Real','Synthetic']

hr_real_array = df_hr['Real'].to_numpy()
hr_real_array
hr_syn_array = df_hr['Synthetic'].to_numpy()
hr_syn_array

df_hr['Heartrate'] = df_hr['Real'].rolling(1200).sum()               #used
df_hr['Heartrate_syn'] = df_hr['Synthetic'].rolling(1200).sum()
import seaborn as sns
fig, ax = plt.subplots(figsize=(5, 5))
sns.set_style("darkgrid", {"grid.color": ".6"})
sns.lineplot(data=df_hr['Heartrate'], label='Real', ax=ax)
sns.lineplot(data=df_hr['Heartrate_syn'], label='Synthetic', ax=ax).set(title="Heart Rate")

# ax.set(xlabel=None)
# ax.set(ylabel=None)
plt.ylim(700, 900)
plt.show()

df_mp['MeanPressure'] = df_mp['Real'].rolling(1200).sum()               #used
df_mp['MeanPressure_syn'] = df_mp['Synthetic'].rolling(1200).sum()

fig, ax = plt.subplots(figsize=(5, 5))
sns.set_style("darkgrid", {"grid.color": ".6"})
sns.lineplot(data=df_mp['MeanPressure'], label='Real', ax=ax)
sns.lineplot(data=df_mp['MeanPressure_syn'], label='Synthetic', ax=ax).set(title="Mean Pressure")
# ax.set(xlabel=None)
# ax.set(ylabel=None)
plt.ylim(500, 900)
plt.show()

df_os['O2SAT'] = df_os['Real'].rolling(1200).sum()               #used
df_os['O2SAT_syn'] = df_os['Synthetic'].rolling(1200).sum()

fig, ax = plt.subplots(figsize=(5, 5))
sns.set_style("darkgrid", {"grid.color": ".6"})
sns.lineplot(data=df_os['O2SAT'], label='Real', ax=ax)
sns.lineplot(data=df_os['O2SAT_syn'], label='Synthetic', ax=ax).set(title="O2 SAT")
# ax.set(xlabel=None)
# ax.set(ylabel=None)
plt.ylim(500, 900)
plt.show()

df_rp['RespRate'] = df_rp['Real'].rolling(1200).sum()               #used
df_rp['RespRate_syn'] = df_rp['Synthetic'].rolling(1200).sum()

fig, ax = plt.subplots(figsize=(5, 5))
sns.set_style("darkgrid", {"grid.color": ".6"})
sns.lineplot(data=df_rp['RespRate'], label='Real', ax=ax)
sns.lineplot(data=df_rp['RespRate_syn'], label='Synthetic', ax=ax).set(title="Resp Rate")
# ax.set(xlabel=None)
# ax.set(ylabel=None)
plt.ylim(-600, 1000)
plt.show()

df_ALB['ALB'] = df_ALB['Real'].rolling(1200).sum()               #used
df_ALB['ALB_syn'] = df_ALB['Synthetic'].rolling(1200).sum()

fig, ax = plt.subplots(figsize=(5, 5))
sns.set_style("darkgrid", {"grid.color": ".6"})
sns.lineplot(data=df_ALB['ALB'], label='Real', ax=ax)
sns.lineplot(data=df_ALB['ALB_syn'], label='Synthetic', ax=ax).set(title="ALB")
# ax.set(xlabel=None)
# ax.set(ylabel=None)
plt.ylim(-10, 10)
plt.show()

df_ALT['ALT'] = df_ALT['Real'].rolling(1200).sum()               #used
df_ALT['ALT_syn'] = df_ALT['Synthetic'].rolling(1200).sum()

fig, ax = plt.subplots(figsize=(5, 5))
sns.set_style("darkgrid", {"grid.color": ".6"})
sns.lineplot(data=df_ALT['ALT'], label='Real', ax=ax)
sns.lineplot(data=df_ALT['ALT_syn'], label='Synthetic', ax=ax).set(title="ALT")
# ax.set(xlabel=None)
# ax.set(ylabel=None)
plt.ylim(-10, 10)
plt.show()

df_BICARBART['BICARBART'] = df_BICARBART['Real'].rolling(1200).sum()               #used
df_BICARBART['BICARBART_syn'] = df_BICARBART['Synthetic'].rolling(1200).sum()

fig, ax = plt.subplots(figsize=(5, 5))
sns.set_style("darkgrid", {"grid.color": ".6"})
sns.lineplot(data=df_BICARBART['BICARBART'], label='Real', ax=ax)
sns.lineplot(data=df_BICARBART['BICARBART_syn'], label='Synthetic', ax=ax).set(title="BICARBART")
# ax.set(xlabel=None)
# ax.set(ylabel=None)
plt.ylim(400, 1800)
plt.show()

df_CO2['CO2'] = df_CO2['Real'].rolling(1200).sum()               #used
df_CO2['CO2_syn'] = df_CO2['Synthetic'].rolling(1200).sum()

fig, ax = plt.subplots(figsize=(5, 5))
sns.set_style("darkgrid", {"grid.color": ".6"})
sns.lineplot(data=df_CO2['CO2'], label='Real', ax=ax)
sns.lineplot(data=df_CO2['CO2_syn'], label='Synthetic', ax=ax).set(title="CO2")
# ax.set(xlabel=None)
# ax.set(ylabel=None)
plt.ylim(100, 800)
plt.show()