#%%
import numpy as np 
import pandas as pd 
import warnings
warnings.filterwarnings('ignore')
import gc
import matplotlib.pyplot as plt 
import seaborn as sns 
import os

#%%
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objects as go 
import plotly.offline as offline
offline.init_notebook_mode() 

#%%
%%time
folder = 'input/'
train_df = pd.read_csv(folder + 'train.csv')
weather_train_df = pd.read_csv(folder + 'weather_train.csv')
test_df = pd.read_csv(folder + 'test.csv')
weather_test_df = pd.read_csv(folder + 'weather_test.csv')
building_meta_df = pd.read_csv(folder + 'building_metadata.csv')
sample_submission = pd.read_csv(folder + 'sample_submission.csv')
#%%
# check size of data
print('Size of train_df data', train_df.shape)
print('Size of weather_train_df data', weather_train_df.shape)
print('Size of weather_test_df data', weather_test_df.shape)
print('Size of building_meta_df data', building_meta_df.shape)

#%%
train_df.head()

#%%
train_df.columns.values

#%%
weather_train_df.head()

#%%
weather_train_df.columns.values 

#%%
weather_test_df.head()

#%%
weather_test_df.columns.values

#%%
building_meta_df.head()

#%%
building_meta_df.columns.values

#%%
# for key, d in train_df.groupby('meter_reading'):
#     break
#     d.head()
plt.figure(figsize = (10, 5))
train_df['meter_reading'].plot()

#%%
plt.hist(train_df['meter_reading'], bins=77)
plt.title('Distribution of id_01 variable')

#%%
train_df['meter_reading'].plot(kind='hist', bins=25, figsize=(15, 5), title= 'Distribution of Target Variable (meter_reading)')
plt.show()

#%%
# examine missing values for train data
total = train_df.isnull().sum().sort_values(ascending=False)
percent= (train_df.isnull().sum()/train_df.isnull().count()*100).sort_values(ascending=False)
missing_train_data = pd.concat([total, percent], axis=1, keys=['total', 'Percent'])
missing_train_data

#%%
# examine missing values for weather_train data
total = weather_train_df.isnull().sum().sort_values(ascending=False)
percent = (weather_train_df.isnull().sum()/weather_train_df.isnull().count()*100).sort_values(ascending=False)
missing_weather_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_weather_data

#%%
# examine missing values for weather_test data
total = weather_test_df.isnull().sum().sort_values(ascending=False)
percent = (weather_test_df.isnull().sum()/weather_test_df.isnull().count()*100).sort_values(ascending=False)
missing_weather_test_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_weather_test_data

#%%
# examine missing values for building_meta data
total = building_meta_df.isnull().sum().sort_values(ascending=False)
percent = (building_meta_df.isnull().sum()/building_meta_df.isnull().count()*100).sort_values(ascending=False)
missing_building_meta_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_building_meta_data

#%%
# Number of each type of column
train_df.dtypes.value_counts()

#%%
# Number of unique classes in each object column
train_df.select_dtypes('object').apply(pd.Series.nunique, axis=0)

#%%
# lets find the correlation between the data to find some relevance of that features.
correlations = train_df.corr()['meter_reading'].sort_values()

# Display correlations
print('Most Positive Correlations:\n', correlations.tail(15))
print('\nMost Negative Correlations:\n', correlations.head(15))

#%%
