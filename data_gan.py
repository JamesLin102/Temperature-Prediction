import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

# make the time-series dataset
dataset = pd.read_csv('data/seattle-weather.csv')
temp = dataset['temp_min'] # temp_max or temp_min
save_path = 'data/temp_min/' # temp_max or temp_min

time_steps = 5
x_timeSeries = []
y_timeSeries = []
for i in range(len(temp)-time_steps-1):
    x_set = temp[i:i+time_steps]
    x_timeSeries.append(x_set)
    y_set = temp[i+time_steps:i+time_steps+1]
    y_timeSeries.append(y_set)

x_timeSeries = np.array(x_timeSeries)
y_timeSeries = np.array(y_timeSeries)

# cross validation
n_splits = 10
kf = KFold(n_splits=n_splits)
for i, (train_index, test_index) in enumerate(kf.split(x_timeSeries)):
    
    # split data
    ori_x_train, x_test = x_timeSeries[train_index], x_timeSeries[test_index]
    ori_y_train, y_test = y_timeSeries[train_index], y_timeSeries[test_index]

    # process dataset for CTGAN
    data = np.concatenate((ori_x_train, ori_y_train), axis=1)
    np.save(save_path+'ori_training_data_'+str(i+1)+'.npy', data)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    data = pd.DataFrame(data)
    data.columns = ['temp_1', 'temp_2', 'temp_3', 'temp_4', 'temp_5', 'temp_6']

    # CTGAN
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=data)
    metadata.validate()
    synthesizer = CTGANSynthesizer(metadata, enforce_min_max_values=True, epochs=500, verbose=True)
    synthesizer.fit(data)
    synthetic_data = synthesizer.sample(num_rows=1400)
    synthetic_data = scaler.inverse_transform(synthetic_data)
    np.save(save_path+'synthetic_data_'+str(i+1)+'.npy', synthetic_data)

    # save testing data
    testing_data = np.concatenate((x_test, y_test), axis=1)
    np.save(save_path+'testing_data_'+str(i+1)+'.npy', testing_data)
