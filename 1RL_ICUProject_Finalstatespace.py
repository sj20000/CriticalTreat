

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


# dfLabMonitorR = pd.read_csv('LabMonitorReward.csv')
# dfLabMonitorR.head()


# STEP 1 - PREPROCESSING  (DONE IN GOOGLE COLAB google colab with MTS_ICU.ipynb)



dfLabMonitorR = pd.read_csv('dfMonitorLabFinal.csv')   #preprocessing was done in google colab with MTS_ICU.ipynb
dfLabMonitorR.head()


# STEP 2 - ADD FLOWSHEET INFO (DONE IN THIS NOTEBOOK)


# pd.set_option('display.max_rows', None)
# # dfLabMonitorR['DateTime']
# dfLabMonitorR



#drop outliers - already outliers are removed
# dfLabMonitorR_Prep = dfLabMonitorR.drop(dfLabMonitorR[dfLabMonitorR.meanpressure > 20000].index)
# dfLabMonitorR_Prep = dfLabMonitorR_Prep.drop(dfLabMonitorR_Prep[dfLabMonitorR_Prep.heartrate < 20].index)
# dfLabMonitorR_Prep = dfLabMonitorR_Prep.drop(dfLabMonitorR_Prep[dfLabMonitorR_Prep.meanpressure < 20].index)



state_features = ['heartrate','meanpressure','osat','resprate']



#copy dopamine and fentanyl
dfLabMonitorFlow = dfLabMonitorR.copy()
dfLabMonitorFlow['Dopamine'] = 0
dfLabMonitorFlow['Fentanyl'] = 0


#dopamine was started from 1993-08-16 22:00:00. However there are no monitor records for this time period. 
#Therefore add rows - TO DO (and update dopamine =12.5 for this row) - do this later
x_df = dfLabMonitorFlow.loc[(dfLabMonitorFlow['DateTime'] >= '1993-08-16 22:00:00') & (dfLabMonitorFlow['DateTime'] < '1993-08-16 22:15:00')]
x_df


dfLabMonitorFlow["DateTime"] = dfLabMonitorFlow["DateTime"].astype('datetime64[ns]')



#update dopamine = 15
dfLabMonitorFlow.loc[(dfLabMonitorFlow['DateTime'] >= '1993-08-16 22:15:00') & (dfLabMonitorFlow['DateTime'] < '1993-08-17 1:00:00'),'Dopamine'] = 15
#update dopamine = 10
dfLabMonitorFlow.loc[(dfLabMonitorFlow['DateTime'] >= '1993-08-17 1:00:00') & (dfLabMonitorFlow['DateTime'] < '1993-08-17 4:00:00'),'Dopamine'] = 10

#update dopamine = 15
dfLabMonitorFlow.loc[(dfLabMonitorFlow['DateTime'] >= '1993-08-17 4:00:00') & (dfLabMonitorFlow['DateTime'] < '1993-08-17 4:30:00'),'Dopamine'] = 15


#update dopamine = 12.5
dfLabMonitorFlow.loc[(dfLabMonitorFlow['DateTime'] >= '1993-08-17 4:30:00') & (dfLabMonitorFlow['DateTime'] < '1993-08-17 5:30:00'),'Dopamine'] = 12.5

#update dopamine = 10
dfLabMonitorFlow.loc[(dfLabMonitorFlow['DateTime'] >= '1993-08-17 5:30:00') & (dfLabMonitorFlow['DateTime'] < '1993-08-17 6:30:00'),'Dopamine'] = 10

#update dopamine = 7.5
dfLabMonitorFlow.loc[(dfLabMonitorFlow['DateTime'] >= '1993-08-17 6:30:00') & (dfLabMonitorFlow['DateTime'] < '1993-08-17 9:30:00'),'Dopamine'] = 7.5
#update dopamine = 2.5
dfLabMonitorFlow.loc[dfLabMonitorFlow['DateTime'] >= '1993-08-17 9:30:00','Dopamine'] = 2.5



#dopamine was administered from 1993-08-16 22:00:00
x_df = dfLabMonitorFlow.loc[dfLabMonitorFlow['Dopamine'] ==0]
x_df



#update fentanyl
#fentanyl is administered hourly, so update only for hourly data

dfLabMonitorFlow.loc[dfLabMonitorFlow['DateTime'] == '1993-08-16 23:16:00','Fentanyl'] = 60

dfLabMonitorFlow.loc[dfLabMonitorFlow['DateTime'] == '1993-08-17 00:00:01','Fentanyl'] = 60

dfLabMonitorFlow.loc[dfLabMonitorFlow['DateTime'] == '1993-08-17 01:00:01','Fentanyl'] = 60

dfLabMonitorFlow.loc[dfLabMonitorFlow['DateTime'] == '1993-08-17 02:00:01','Fentanyl'] = 70

dfLabMonitorFlow.loc[dfLabMonitorFlow['DateTime'] == '1993-08-17 03:00:04','Fentanyl'] = 70

dfLabMonitorFlow.loc[dfLabMonitorFlow['DateTime'] == '1993-08-17 04:04:04','Fentanyl'] = 70

dfLabMonitorFlow.loc[dfLabMonitorFlow['DateTime'] == '1993-08-17 05:04:04','Fentanyl'] = 70

dfLabMonitorFlow.loc[dfLabMonitorFlow['DateTime'] == '1993-08-17 06:00:04','Fentanyl'] = 70

dfLabMonitorFlow.loc[dfLabMonitorFlow['DateTime'] == '1993-08-17 07:00:06','Fentanyl'] = 70

dfLabMonitorFlow.loc[dfLabMonitorFlow['DateTime'] == '1993-08-17 08:00:06','Fentanyl'] = 70

dfLabMonitorFlow.loc[dfLabMonitorFlow['DateTime'] == '1993-08-17 09:00:09','Fentanyl'] = 70

dfLabMonitorFlow.loc[dfLabMonitorFlow['DateTime'] == '1993-08-17 10:00:09','Fentanyl'] = 70

dfLabMonitorFlow.loc[dfLabMonitorFlow['DateTime'] == '1993-08-17 11:00:14','Fentanyl'] = 70

dfLabMonitorFlow.loc[dfLabMonitorFlow['DateTime'] == '1993-08-17 12:25:00','Fentanyl'] = 70



#dopamine  = 0 : 0
#dopamine  >0 and < 7 : 1
#dopamine  >= 7 and < 12: 2
#dopamine  > 12 : 3
dfLabMonitorFlow.loc[dfLabMonitorFlow['Dopamine'] == 0 , 'DopamineAction'] = 0
dfLabMonitorFlow.loc[(dfLabMonitorFlow['Dopamine'] >0) & (dfLabMonitorFlow['Dopamine'] < 7), 'DopamineAction'] = 1
dfLabMonitorFlow.loc[(dfLabMonitorFlow['Dopamine'] >=7) & (dfLabMonitorFlow['Dopamine'] < 12), 'DopamineAction'] = 2
dfLabMonitorFlow.loc[dfLabMonitorFlow['Dopamine'] >12, 'DopamineAction'] = 3


#save as csv file
dfLabMonitorFlow.to_csv('dfLMF.csv')



dfLMF = pd.read_csv('dfLMF.csv')   
dfLMF=dfLMF.drop('Unnamed: 0',axis='columns')
dfLMF.head(150)




# STEP 3 - CREATE SYNTHETIC DATA
# 
# #With dfLMF.csv create synthetic data
# RL_ICU_syntheticdata.ipynb - google colab
# 

# The arrays from original data : ICU_data and the synthetic data : synth_data are saved as 
# ICU_dataA1.csv and 
# synth_dataA1.csv


colnames = ['VentMode', 'heartrate', 'meanpressure', 'sysPressure', 'diasPressure',        'artheartrate', 'osat', 'FiO2', 'resprate', 'TidalVol', 'PIP',        'AirwayP', 'PEEP', 'ALB', 'ALT', 'AST', 'BICARBART', 'BILITOTAL', 'BUN',        'CA', 'CL', 'CO2', 'CR', 'FIB', 'GLU', 'HCT', 'HGB', 'IONCA', 'K',        'MCH', 'MCHC', 'MCV', 'MG', 'NA', 'NH3', 'O2SATART', 'PCO2ART', 'PHART',        'PHOS', 'PLT', 'PO2ART', 'PT', 'PT1', 'PT2', 'PTT', 'PTT1', 'PTT2',        'RBC', 'TP', 'TRIG', 'WBC', 'Dopamine', 'Fentanyl', 'DopamineAction']
ICU_arr_df = pd.read_csv('ICU_dataA1.csv',  names=colnames, header=None)



Syn_arr_df = pd.read_csv('synth_dataA1.csv', names=colnames, header=None)

Syn_arr_df.head()


# STEP 4

# Learn the states with RNN




# Splitting dataset into train/val/test        -
val_time = 1000
test_time = 2000
training_data = ICU_arr_df.iloc[:val_time]
validation_data = ICU_arr_df.iloc[val_time:test_time]
test_data = ICU_arr_df.iloc[test_time:]




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


# Create the LSTM AUTOENCODER MODEL




import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, LSTM, Model



#trial 4  - Final - To get the latent representation - 

#https://datascience.stackexchange.com/questions/64412/how-to-extract-features-from-the-encoded-layer-of-an-autoencoder

from tensorflow import keras
    
# This is the dimension of the original space
# input_dim = 54

# Encoding information
encoding_dim = 10 ## This is the dimension of the latent space (encoding space)

X_t=X_train.reshape(-1,54)
col_num = X_t.shape[1]
input_dim = Input(shape=(col_num,))

X_val, y_val = create_dataset(validation_data, validation_data, TIME_STEPS)
X_v=X_val.reshape(-1,54)
    
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


# Show the encoded values
print(encoded_output)



with np.printoptions(threshold=np.inf):
    print(encoded_output)


np.savetxt('statespace.csv', encoded_output, delimiter=',')




#Trial 3

#https://ekamperi.github.io/machine%20learning/2021/01/21/encoder-decoder-model.html

# This is the dimension of the original space
input_dim = 54

# This is the dimension of the latent space (encoding space)
latent_dim = 10

encoder = Sequential([
    Dense(128, activation='relu', input_shape=(input_dim,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(latent_dim, activation='relu', name='encoder_output')
])

decoder = Sequential([
    Dense(64, activation='relu', input_shape=(latent_dim,)),
    Dense(128, activation='relu'),
    Dense(256, activation='relu'),
    Dense(input_dim, activation=None)
])



#https://stackoverflow.com/questions/69269890/keras-attributeerror-sequential-object-has-no-attribute-nested-inputs
inp = Input((input_dim,))
autoencoder = Model(inputs=inp, outputs=decoder(encoder(inp)))
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='mse')



autoencoder.summary()


X_t=X_train.reshape(-1,54)
#TRIAL 3
model_history = autoencoder.fit(X_t, X_t, epochs=10, batch_size=32, verbose=0, validation_split=0.1)


# In[360]:


import matplotlib.pyplot as plt          
get_ipython().run_line_magic('matplotlib', 'inline')
#TRIAL 3
plt.plot(model_history.history['loss'], label='Training loss')
plt.plot(model_history.history['val_loss'], label='Validation loss')
plt.legend();


#https://ekamperi.github.io/machine%20learning/2021/01/21/encoder-decoder-model.html

#Training the autoencoder
plt.plot(model_history.history["loss"])
plt.title("Loss vs. Epoch")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.grid(True)



encoded_x_train = encoder(X_t)
plt.figure(figsize=(6,6))
plt.scatter(encoded_x_train[:, 0], encoded_x_train[:, 1], alpha=.8)
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2');



encoded_x_train = encoder(X_t)
plt.figure(figsize=(6,6))
plt.scatter(encoded_x_train[:, 0], encoded_x_train[:, 1], alpha=.8)
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2');





#https://medium.com/analytics-vidhya/introduction-to-2-dimensional-lstm-autoencoder-47c238fd827f   -
#TRIAL 1
from tensorflow import keras
from tensorflow.python.keras.layers import Input, Dense,RepeatVector,  Dense, Dropout, LSTM
from keras.layers import TimeDistributed
from tensorflow.python.keras.models import Sequential




model_1 = Sequential()
model_1.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))
model_1.add(Dropout(rate=0.2))
model_1.add(RepeatVector(X_train.shape[1]))
model_1.add(LSTM(128, return_sequences=True))
model_1.add(Dropout(rate=0.2))
model_1.add(keras.layers.TimeDistributed(keras.layers.Dense(X_train.shape[2])))
model_1.compile(optimizer='adam', loss='mae')
model_1.summary()

# https://analyticsindiamag.com/introduction-to-lstm-autoencoder-using-keras/
# Adding RepeatVector to the layer means it repeats the input n number of times. 
#The TimeDistibuted layer takes the information from the previous layer and creates a vector with a length of the output layers.


# try out this:
#https://towardsdatascience.com/step-by-step-understanding-lstm-autoencoder-layers-ffab055b6352
###
#TRIAL 2

# define model
model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(LSTM(64, activation='relu', return_sequences=False))
model.add(RepeatVector(X_train.shape[1]))
model.add(LSTM(64, activation='relu', return_sequences=True))
model.add(LSTM(128, activation='relu', return_sequences=True))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(X_train.shape[2])))
model.compile(optimizer='adam', loss='mse')
model.summary()


#TRIAL 2
history = model.fit(X_train, X_train, epochs=10, batch_size=32, validation_split=0.1,        #   
                    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')], shuffle=False)



#TRIAL 2
import matplotlib.pyplot as plt         
get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend();


#TRIAL 1
history_1 = model_1.fit(X_train, X_train, epochs=10, batch_size=32, validation_split=0.1,         
                    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')], shuffle=False)



#TRIAL 1
import matplotlib.pyplot as plt          
get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(history_1.history['loss'], label='Training loss')
plt.plot(history_1.history['val_loss'], label='Validation loss')
plt.legend();


# Test the AUTOENCODER


#TRIAL 1

#https://github.com/adnanmushtaq1996/2D-LSTM-AUTOENCODER/blob/main/2D_LSTM_Autoencoder.ipynb        -

import sklearn.metrics


from sklearn.metrics import mean_squared_error
from math import sqrt


#test data close to what model has seen and check the MSE
X_val, y_val = create_dataset(validation_data, validation_data, TIME_STEPS)
# test = create_dataset(validation_data, validation_data, TIME_STEPS)

# print(X_train.shape)

# a.shape

# X_val1 = np.reshape(X_val,newshape=(-1,5,54))
pred= model_1.predict(X_val)
y_pred=pred.reshape(-1,54)

print(" THE MSE IS   : " ,sklearn.metrics.mean_squared_error(y_val, y_pred))

#calculate RMSE
print(" THE RMSE IS  : " ,sqrt(mean_squared_error(y_val, y_pred)))


# print("The Recreated Output is : ",y_pred)



#TRIAL 2


#https://github.com/adnanmushtaq1996/2D-LSTM-AUTOENCODER/blob/main/2D_LSTM_Autoencoder.ipynb    

import sklearn.metrics


from sklearn.metrics import mean_squared_error
from math import sqrt


#test data close to what model has seen and check the MSE
X_val, y_val = create_dataset(validation_data, validation_data, TIME_STEPS)
# test = create_dataset(validation_data, validation_data, TIME_STEPS)

# print(X_train.shape)

# a.shape

# X_val1 = np.reshape(X_val,newshape=(-1,5,54))
pred= model.predict(X_val)
y_pred=pred.reshape(-1,54)

print(" THE MSE IS   : " ,sklearn.metrics.mean_squared_error(y_val, y_pred))

#calculate RMSE
print(" THE RMSE IS  : " ,sqrt(mean_squared_error(y_val, y_pred)))


# print("The Recreated Output is : ",y_pred)




#TRIAL 1
# Evaluation                 - USED (SAME AS BEFORE)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend();



#save as csv file
dfLabMonitorFlow.to_csv('dfLMF.csv')



FinalICUdf = pd.read_csv('dfLMF.csv')

FinalICUdf.drop(columns=FinalICUdf.columns[0], axis=1, inplace=True)
FinalICUdf.head()



#https://kanokidotorg.github.io/2022/06/28/pandas-create-new-column-based-on-value-in-other-columns-with-multiple-conditions/

def compute_heartrate_risk(df):    
    if (df['heartrate'] >= 75) & (df['heartrate'] >= 75):
        return 1
    elif (df['heartrate'] >= 160) & (df['heartrate'] <= 180):
        return 1
    elif (df['heartrate'] >= 50) & (df['heartrate'] <= 74):
        return 3
    elif (df['heartrate'] >= 181) & (df['heartrate'] <= 220):
        return 3
    elif (df['heartrate'] > 200):
        return 5
    elif (df['heartrate'] < 40):
        return 5
    else:
        return 0
    
FinalICUNextdf['heartrateRisk'] = FinalICUNextdf.apply(compute_heartrate_risk, axis = 1)


# In[7]:


#https://kanokidotorg.github.io/2022/06/28/pandas-create-new-column-based-on-value-in-other-columns-with-multiple-conditions/

def compute_osat_risk(df):    
    if (df['osat'] >=96): #no risk:
        return 0
    elif (df['osat'] >=95) & (df['osat'] <96):
        return 1 #not much risk
    elif (df['osat'] >= 90) & (df['osat'] < 95):
        return 3  # considerable risk
    elif (df['osat'] < 90):
        return 5   #high risk
    
    
    # #oxygen saturation  #https://www.cosinuss.com/en/measured-data/vital-signs/oxygen-saturation/

FinalICUNextdf['osatRisk'] = FinalICUNextdf.apply(compute_osat_risk, axis = 1)


# In[8]:


#https://kanokidotorg.github.io/2022/06/28/pandas-create-new-column-based-on-value-in-other-columns-with-multiple-conditions/

#https://www.healthline.com/health/mean-arterial-pressure#high-map

def compute_bp_risk(df):    
    if (df['meanpressure'] >= 60) & (df['meanpressure'] <= 100):
        return 0
    else:
        return 5
    
FinalICUNextdf['bpRisk'] = FinalICUNextdf.apply(compute_bp_risk, axis = 1)


# In[9]:


#https://kanokidotorg.github.io/2022/06/28/pandas-create-new-column-based-on-value-in-other-columns-with-multiple-conditions/

def compute_resprate_risk(df):    
    if (df['resprate'] >=50) & (df['resprate'] >= 60):
        return 1  #lower risk
    elif (df['resprate'] >= 61) & (df['resprate'] <= 90):
        return 3  #little high risk
    elif (df['resprate'] > 90):
        return 5  #very high risk
    elif (df['resprate'] >= 25) & (df['resprate'] <50):
        return 0  #no risk
    elif (df['resprate'] >= 12) & (df['resprate'] <25):
        return 1  #low risk
    elif (df['resprate'] <12):  # very high risk
        return 5  #Bradypnea
  

    #https://en.wikipedia.org/wiki/Respiratory_rate
    
FinalICUNextdf['resprateRisk'] = FinalICUNextdf.apply(compute_resprate_risk, axis = 1)

