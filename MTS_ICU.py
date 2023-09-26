
"""MTS_ICU.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf

import sklearn.metrics as metrics

from google.colab import drive
drive.mount('googledrive')

"""FLOWSHEET DATA"""

col_names=['Date','Time','Intervention','Value']
#last 9 records - changed separator as tab instead of space, before reading into dataframe

df_flowsheet = pd.read_csv('/content/googledrive/MyDrive/MSC8002/Flowsheet-Data', header=None, delimiter= '\t', names=col_names, encoding='utf-8')
df_flowsheet.head()

df_flowsheet.loc[:,'DateTime'] = pd.to_datetime(df_flowsheet.Date+' '+df_flowsheet.Time)

df_flowsheet_subset=df_flowsheet.drop(['Time','Date'], axis=1)
df_flowsheet_subset.head()

df_flowsheet_pivot=df_flowsheet_subset.pivot(index='DateTime',columns='Intervention',values='Value').reset_index().rename_axis(None,axis=1)
df_flowsheet_pivot

#Fill NAN values with zero
df_flowsheet_pivot['Dopamine'] = df_flowsheet_pivot['Dopamine(mg/kg/min)'].fillna(0)
df_flowsheet_pivot['Fentanyl'] = df_flowsheet_pivot['Fentanyl (mcg/hr)'].fillna(0)
df_flowsheet_pivot['Fluid In'] = df_flowsheet_pivot['Fluid In (cc)'].fillna(0)
df_flowsheet_pivot['Fluid Out'] = df_flowsheet_pivot['Fluid Out (cc/hr)'].fillna(0)
df_flowsheet_pivot['Pavulon'] = df_flowsheet_pivot['Pavulon(mg/hr)'].fillna(0)
df_flowsheet_pivot['Temp'] = df_flowsheet_pivot['Temp'].fillna(0)
df_flowsheet_pivot['Terbutaline'] = df_flowsheet_pivot['Terbutaline (mcg/kg/min)'].fillna(0)
df_flowsheet_pivot['Versed'] = df_flowsheet_pivot['Versed (mg/hr)'].fillna(0)
#drop old columns
df_flowsheet_pivot.drop(df_flowsheet_pivot.columns[[1,2,3,4,5,7,8]], axis=1, inplace=True)
df_flowsheet_pivot

df_flowsheet_pivot.dtypes

#update temp in 0th row from 38.3 C to 38.3 , before datatype conversion
df_flowsheet_pivot.loc[0, ['Temp']] = [38.3]

#convert the  column data type into float

df_flowsheet_pivot['Temp'] = df_flowsheet_pivot['Temp'].astype(float)
df_flowsheet_pivot['Dopamine'] = df_flowsheet_pivot['Dopamine'].astype(float)
df_flowsheet_pivot['Fentanyl'] = df_flowsheet_pivot['Fentanyl'].astype(float)
df_flowsheet_pivot['Fluid In'] = df_flowsheet_pivot['Fluid In'].astype(float)
df_flowsheet_pivot['Fluid Out'] = df_flowsheet_pivot['Fluid Out'].astype(float)
df_flowsheet_pivot['Pavulon'] = df_flowsheet_pivot['Pavulon'].astype(float)
df_flowsheet_pivot['Terbutaline'] = df_flowsheet_pivot['Terbutaline'].astype(float)
df_flowsheet_pivot['Versed'] = df_flowsheet_pivot['Versed'].astype(float)

Dopamine_values = df_flowsheet_pivot["Dopamine"][df_flowsheet_pivot["Dopamine"] >0]
Fentanyl_values = df_flowsheet_pivot["Fentanyl"][df_flowsheet_pivot["Fentanyl"]>0]
Temp_values = df_flowsheet_pivot["Temp"][df_flowsheet_pivot["Temp"] >0]
FluidIn_values = df_flowsheet_pivot["Fluid In"][df_flowsheet_pivot["Fluid In"]>0]
FluidOut_values = df_flowsheet_pivot["Fluid Out"][df_flowsheet_pivot["Fluid Out"] >0]
Pavulon_values = df_flowsheet_pivot["Pavulon"][df_flowsheet_pivot["Pavulon"]>0]
Terbutaline_values = df_flowsheet_pivot["Terbutaline"][df_flowsheet_pivot["Terbutaline"] >0]
Versed_values = df_flowsheet_pivot["Versed"][df_flowsheet_pivot["Versed"]>0]


# Dopamine_quartiles = adjusted_Dopamine.quantile([0.25,0.50,0.75])
# Fentanyl_quartiles = adjusted_Fentanyl.quantile([0.25,0.5,0.75])

Dopamine_quartiles

Fentanyl_quartiles

"""LAB DATA"""

#keep_default_na=False is used because by default, the column: Na is read by pandas as NaN
col_names=['Date','Time','Test','Value']
dfLab = pd.read_csv('/content/googledrive/MyDrive/MSC8002/Lab-Data', delim_whitespace=True, names=col_names, header=None, keep_default_na=False)
dfLab.head()

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
dfLab

dfLab.loc[:,'DateTime'] = pd.to_datetime(dfLab.Date+' '+dfLab.Time)
dfLab_subset=dfLab.drop(['Time','Date'], axis=1)
dfLab_subset.head()

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
dfLab_subset

dfLab_pivot=dfLab_subset.pivot(index='DateTime',columns='Test',values='Value').reset_index().rename_axis(None,axis=1)
dfLab_pivot

dfLab_pivot.fillna(method='ffill', inplace=True)
#fill the rest with backfill
dfLab_pivot.fillna(method='bfill', inplace=True)
dfLab_pivot



"""MONITOR DATA"""

#assign path variable
PATH_monitor = "/content/googledrive/MyDrive/MSC8002/Monitor-Data"
PATH_monitorcode = "/content/googledrive/MyDrive/MSC8002/Monitor-Data-Codes"

col_names=['Time','Code','Value']
dfMonitor = pd.read_csv(PATH_monitor, delim_whitespace=True, header=None, names=col_names)
#print(dfMonitor)

#add date field, join date and time columns
dfMonitor['date'] = [pd.to_datetime('8/16/1993') if x > '12:00:00' else pd.to_datetime('8/17/1993') for x in dfMonitor['Time']]
dfMonitor.loc[:,'DateTime'] = pd.to_datetime(dfMonitor.date.astype(str)+' '+dfMonitor.Time)

dfMonitor_subset=dfMonitor.drop(['Time','date'], axis=1)
dfMonitor_subset.head()

dfMonitor_pivot=dfMonitor.pivot(index='DateTime',columns='Code',values='Value').reset_index().rename_axis(None,axis=1)
dfMonitor_pivot

dfMonitor_pivot.reset_index(inplace=True)
dfMonitor_pivot

# dfMonitor_Codesubset=dfMonitor_pivot[['DateTime',7,19,22, 59,80]].copy()
# dfMonitor_Codesubset

#new-29 july

dfMonitor_Codesubset=dfMonitor_pivot[['DateTime',1,7,19,20,21,22, 59,76,80,81,83,84,85]].copy()
dfMonitor_Codesubset

# dfMonitor_Codesubset.columns = ['DateTime', 'heartrate', 'meanpressure', 'artheartrate', 'osat','resprate']
# dfMonitor_Codesubset

dfMonitor_Codesubset.columns = ['DateTime', 'VentMode', 'heartrate', 'meanpressure', 'sysPressure','diasPressure','artheartrate', \
                                'osat','FiO2', 'resprate','TidalVol','PIP','AirwayP','PEEP']
dfMonitor_Codesubset



#if no ECG heart rate -code-7 (NAN), then copy from arterial heart rate (code=22)
dfMonitor_Codesubset.heartrate.fillna(dfMonitor_Codesubset.artheartrate, inplace=True)
dfMonitor_Codesubset

#now drop arterial heart rate (code=22)
# dfMonitor_Codesubset=dfMonitor_Codesubset.drop(['artheartrate'], axis=1)
# dfMonitor_Codesubset.head()

dfMonitor_Codesubset.fillna(method='ffill', inplace=True)
dfMonitor_Codesubset

#check outliers

#https://careerfoundry.com/en/blog/data-analytics/how-to-find-outliers/
fig = px.box(dfMonitor_Codesubset, y='heartrate')

fig.show()

fig = px.box(dfMonitor_Codesubset, y='meanpressure')

fig.show()

#https://careerfoundry.com/en/blog/data-analytics/how-to-find-outliers/
def find_outliers_IQR(df):

   q1=df.quantile(0.25)

   q3=df.quantile(0.75)

   IQR=q3-q1

   outliers = df[((df<(q1-1.5*IQR)) | (df>(q3+1.5*IQR)))]

   return outliers

outliers = find_outliers_IQR(dfMonitor_Codesubset['VentMode'])

print('number of outliers: '+ str(len(outliers)))

print('max outlier value: '+ str(outliers.max()))

print('min outlier value: '+ str(outliers.min()))

outliers

outliers = find_outliers_IQR(dfMonitor_Codesubset['heartrate'])

print('number of outliers: '+ str(len(outliers)))

print('max outlier value: '+ str(outliers.max()))

print('min outlier value: '+ str(outliers.min()))

outliers



outliers = find_outliers_IQR(dfMonitor_Codesubset['meanpressure'])

print('number of outliers: '+ str(len(outliers)))

print('max outlier value: '+ str(outliers.max()))

print('min outlier value: '+ str(outliers.min()))

outliers

outliers = find_outliers_IQR(dfMonitor_Codesubset['sysPressure'])

print('number of outliers: '+ str(len(outliers)))

print('max outlier value: '+ str(outliers.max()))

print('min outlier value: '+ str(outliers.min()))

outliers

outliers = find_outliers_IQR(dfMonitor_Codesubset['diasPressure'])

print('number of outliers: '+ str(len(outliers)))

print('max outlier value: '+ str(outliers.max()))

print('min outlier value: '+ str(outliers.min()))

outliers

outliers = find_outliers_IQR(dfMonitor_Codesubset['artheartrate'])

print('number of outliers: '+ str(len(outliers)))

print('max outlier value: '+ str(outliers.max()))

print('min outlier value: '+ str(outliers.min()))

outliers

outliers = find_outliers_IQR(dfMonitor_Codesubset['osat'])

print('number of outliers: '+ str(len(outliers)))

print('max outlier value: '+ str(outliers.max()))

print('min outlier value: '+ str(outliers.min()))

outliers

outliers = find_outliers_IQR(dfMonitor_Codesubset['FiO2'])

print('number of outliers: '+ str(len(outliers)))

print('max outlier value: '+ str(outliers.max()))

print('min outlier value: '+ str(outliers.min()))

outliers

outliers = find_outliers_IQR(dfMonitor_Codesubset['resprate'])

print('number of outliers: '+ str(len(outliers)))

print('max outlier value: '+ str(outliers.max()))

print('min outlier value: '+ str(outliers.min()))

outliers

outliers = find_outliers_IQR(dfMonitor_Codesubset['TidalVol'])

print('number of outliers: '+ str(len(outliers)))

print('max outlier value: '+ str(outliers.max()))

print('min outlier value: '+ str(outliers.min()))

outliers

outliers = find_outliers_IQR(dfMonitor_Codesubset['PIP'])

print('number of outliers: '+ str(len(outliers)))

print('max outlier value: '+ str(outliers.max()))

print('min outlier value: '+ str(outliers.min()))

outliers

outliers = find_outliers_IQR(dfMonitor_Codesubset['AirwayP'])

print('number of outliers: '+ str(len(outliers)))

print('max outlier value: '+ str(outliers.max()))

print('min outlier value: '+ str(outliers.min()))

outliers

outliers = find_outliers_IQR(dfMonitor_Codesubset['PEEP'])

print('number of outliers: '+ str(len(outliers)))

print('max outlier value: '+ str(outliers.max()))

print('min outlier value: '+ str(outliers.min()))

outliers



#fill the rest with backfill
dfMonitor_Codesubset.fillna(method='bfill', inplace=True)
dfMonitor_Codesubset

#replace outlier values with Nan
#heartrate
dfMonitor_Codesubset.loc[dfMonitor_Codesubset['heartrate'] == 0, 'heartrate'] = np.NaN
dfMonitor_Codesubset.loc[dfMonitor_Codesubset['heartrate'] == 292, 'heartrate'] = np.NaN

dfMonitor_Codesubset.loc[dfMonitor_Codesubset['meanpressure'] == 1, 'meanpressure'] = np.NaN
dfMonitor_Codesubset.loc[dfMonitor_Codesubset['meanpressure'] == 32000, 'meanpressure'] = np.NaN

dfMonitor_Codesubset.loc[dfMonitor_Codesubset['sysPressure'] == 0, 'sysPressure'] = np.NaN
dfMonitor_Codesubset.loc[dfMonitor_Codesubset['sysPressure'] == 32000, 'sysPressure'] = np.NaN

dfMonitor_Codesubset.loc[dfMonitor_Codesubset['diasPressure'] == -2, 'diasPressure'] = np.NaN
dfMonitor_Codesubset.loc[dfMonitor_Codesubset['diasPressure'] == -12, 'diasPressure'] = np.NaN
dfMonitor_Codesubset.loc[dfMonitor_Codesubset['diasPressure'] == 32000, 'diasPressure'] = np.NaN

dfMonitor_Codesubset.loc[dfMonitor_Codesubset['artheartrate'] == 0, 'artheartrate'] = np.NaN

#then backfill, forwardfill

dfMonitor_Codesubset.fillna(method='ffill', inplace=True)

dfMonitor_Codesubset.fillna(method='bfill', inplace=True)

fig = px.box(dfMonitor_Codesubset, y='diasPressure')

fig.show()



"""Find outliers"""

#https://careerfoundry.com/en/blog/data-analytics/how-to-find-outliers/

import plotly.express as px

fig = px.histogram(dfMonitor_Codesubset, x='heartrate')

fig.show()



"""MERGE MONITOR AND LAB DATA"""

#join on data frame column

dfMonitorLab_joined = dfMonitor_Codesubset.set_index('DateTime').join(dfLab_pivot.set_index('DateTime'), how='outer')
dfMonitorLab_joined

dfMonitorLab_joined.fillna(method='ffill', inplace=True)
dfMonitorLab_joined

dfMonitorLab_joined.fillna(method='bfill', inplace=True)
dfMonitorLab_joined

dfMonitorLab_Final = dfMonitorLab_joined.copy()

#save as csv file
dfMonitorLab_Final.to_csv('dfMonitorLabFinal.csv')



#https://stackoverflow.com/questions/49394737/exporting-data-from-google-colab-to-local-machine
from google.colab import files
files.download("dfMonitorLabFinal.csv")

"""END - This preprocessed file is used further programming

RL
"""

#https://github.com/aniruddhraghu/sepsisrl/blob/master/preprocessing/preprocess_data.ipynb
# add rewards - sparsely for now; reward function shaping comes in a separate script
dfMonitorLab_Final['reward'] = 0
dfMonitorLab_Final

dfMonitorLab_Final

dfMonitorLab_Final.to_csv("LabMonitorReward.csv")

"""START FROM HERE FOR SAVED CSV (LAB and MONITOR DATA with initialised REWARDS)"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dfMonitorLab_fromcsv = pd.read_csv("LabMonitorReward.csv", delimiter= None)
# dfMonitorLab_fromcsv = pd.read_csv("LabMonitorReward.csv", delimiter= '\t', names=col_names, encoding='utf-8')
dfMonitorLab_fromcsv.head()

#no. of columns
dfMonitorLab_fromcsv.shape
#44 columns - that is totally 42 columns for learning(disregarding datetime and reward columns)





dfMonitorLab_fromcsv['DetLabel'] = 0

dfMonitorLab_fromcsv.head()