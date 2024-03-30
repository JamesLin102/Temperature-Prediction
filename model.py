import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from predict import plot_pred, plot_residuals
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from tcn import TCN

tf.random.set_seed(7) # fix random seed for reproducibility

# parameters
file_path = 'data/temp_max/' # temp_max or temp_min
splits = 10
time_steps = 5
process = {'TCN + CTGAN (processed)':False, # true is execute, false is not
           'TCN + CTGAN':False,
           'TCN':True}

# for evaluation
y_true_all = np.zeros((1,1))
y_pred_all = np.zeros((1,1))

# TCN + CTGAN (processed)
if process['TCN + CTGAN (processed)'] == True:
    for i in range(splits):

        # load the dataset
        ori_training_data = np.load(file_path+'ori_training_data_'+str(i+1)+'.npy')
        gan_training_data = np.load(file_path+'synthetic_data_'+str(i+1)+'.npy')
        testing_data = np.load(file_path+'testing_data_'+str(i+1)+'.npy')

        # move the outlier in gan_dataset
        result = OneClassSVM().fit_predict(gan_training_data)
        index = []
        for j in range(len(result)):
            if result[j] == -1:
                index.append(j)
        gan_training_data = np.delete(gan_training_data, index, axis=0)

        # data preprocessing
        ori_x_train = ori_training_data[:,:5]
        ori_y_train = ori_training_data[:,5].reshape(-1,1)
        gan_x_train = gan_training_data[:,:5]
        gan_y_train = gan_training_data[:,5].reshape(-1,1)
        x_test = testing_data[:,:5]
        y_test = testing_data[:,5].reshape(-1,1)
        y_true = y_test.copy()

        ori_x_scaler = StandardScaler()
        ori_x_train = ori_x_scaler.fit_transform(ori_x_train)
        ori_y_scaler = StandardScaler()
        ori_y_train = ori_y_scaler.fit_transform(ori_y_train)
        gan_x_scaler = StandardScaler()
        gan_x_train = gan_x_scaler.fit_transform(gan_x_train)
        gan_y_scaler = StandardScaler()
        gan_y_train = gan_y_scaler.fit_transform(gan_y_train)
        x_test_scaler = StandardScaler()
        x_test = x_test_scaler.fit_transform(x_test)
        y_test_scaler = StandardScaler()
        y_test = y_test_scaler.fit_transform(y_test)

        ori_x_transformer = PowerTransformer(method='yeo-johnson')
        ori_x_train = ori_x_transformer.fit_transform(ori_x_train)
        ori_y_transformer = PowerTransformer(method='yeo-johnson')
        ori_y_train = ori_y_transformer.fit_transform(ori_y_train)
        gan_x_transformer = PowerTransformer(method='yeo-johnson')
        gan_x_train = gan_x_transformer.fit_transform(gan_x_train)
        gan_y_transformer = PowerTransformer(method='yeo-johnson')
        gan_y_train = gan_y_transformer.fit_transform(gan_y_train)
        x_test_transformer = PowerTransformer(method='yeo-johnson')
        x_test = x_test_transformer.fit_transform(x_test)
        y_test_transformer = PowerTransformer(method='yeo-johnson')
        y_test = y_test_transformer.fit_transform(y_test)

        # training with TCN
        x_train = np.concatenate((ori_x_train, gan_x_train), axis=0)
        y_train = np.concatenate((ori_y_train, gan_y_train), axis=0)
        x_train = np.reshape(x_train, (x_train.shape[0], time_steps, 1))

        model = Sequential()
        model.add(TCN(25))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(x_train, y_train, epochs=10, batch_size=100)
        model.summary()

        # prediction
        y_pred = model.predict(x_test)
        y_pred = y_test_transformer.inverse_transform(y_pred)
        y_pred = y_test_scaler.inverse_transform(y_pred)
        y_pred_all = np.concatenate((y_pred_all, y_pred), axis=0)
        y_true_all = np.concatenate((y_true_all, y_true), axis=0)

    # evaluation
    y_pred_all = y_pred_all[1:,:] # remove first element (0,0)
    y_true_all = y_true_all[1:,:]
    # np.save('data/data_plot/TCP_y_pred_all.npy', y_pred_all)
    # np.save('data/data_plot/TCP_y_true_all.npy', y_true_all)
    plot_pred(y_true_all, y_pred_all, 'TCN + CTGAN (processed)')
    plot_residuals(y_true_all, y_pred_all, 'TCN + CTGAN (processed)')
    
    # Return to zero
    y_true_all = np.zeros((1,1))
    y_pred_all = np.zeros((1,1))

# TCN + CTGAN
if process['TCN + CTGAN'] == True:

    for i in range(splits):

        # load the dataset
        ori_training_data = np.load(file_path+'ori_training_data_'+str(i+1)+'.npy')
        gan_training_data = np.load(file_path+'synthetic_data_'+str(i+1)+'.npy')
        testing_data = np.load(file_path+'testing_data_'+str(i+1)+'.npy')

        # data preprocessing
        ori_x_train = ori_training_data[:,:5]
        ori_y_train = ori_training_data[:,5].reshape(-1,1)
        gan_x_train = gan_training_data[:,:5]
        gan_y_train = gan_training_data[:,5].reshape(-1,1)
        x_test = testing_data[:,:5]
        y_test = testing_data[:,5].reshape(-1,1)
        y_true = y_test.copy()

        ori_x_scaler = MinMaxScaler()
        ori_x_train = ori_x_scaler.fit_transform(ori_x_train)
        ori_y_scaler = MinMaxScaler()
        ori_y_train = ori_y_scaler.fit_transform(ori_y_train)
        gan_x_scaler = MinMaxScaler()
        gan_x_train = gan_x_scaler.fit_transform(gan_x_train)
        gan_y_scaler = MinMaxScaler()
        gan_y_train = gan_y_scaler.fit_transform(gan_y_train)
        x_test_scaler = MinMaxScaler()
        x_test = x_test_scaler.fit_transform(x_test)
        y_test_scaler = MinMaxScaler()
        y_test = y_test_scaler.fit_transform(y_test)
      
        # training with TCN
        x_train = np.concatenate((ori_x_train, gan_x_train), axis=0)
        y_train = np.concatenate((ori_y_train, gan_y_train), axis=0)
        x_train = np.reshape(x_train, (x_train.shape[0], time_steps, 1))

        model = Sequential()
        model.add(TCN(25))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(x_train, y_train, epochs=10, batch_size=100)
        model.summary()

        # prediction
        y_pred = model.predict(x_test)
        y_pred = y_test_scaler.inverse_transform(y_pred)
        y_pred_all = np.concatenate((y_pred_all, y_pred), axis=0)
        y_true_all = np.concatenate((y_true_all, y_true), axis=0)

    # evaluation
    y_pred_all = y_pred_all[1:,:] # remove first element (0,0)
    y_true_all = y_true_all[1:,:]
    # np.save('data/data_plot/TC_y_pred_all.npy', y_pred_all)
    # np.save('data/data_plot/TC_y_true_all.npy', y_true_all)
    plot_pred(y_true_all, y_pred_all, 'TCN + CTGAN')
    plot_residuals(y_true_all, y_pred_all, 'TCN + CTGAN')

    # Return to zero
    y_true_all = np.zeros((1,1))
    y_pred_all = np.zeros((1,1))

# TCN
if process['TCN'] == True:

    for i in range(splits):

        # load the dataset
        ori_training_data = np.load(file_path+'ori_training_data_'+str(i+1)+'.npy')
        testing_data = np.load(file_path+'testing_data_'+str(i+1)+'.npy')

        # data preprocessing
        x_train = ori_training_data[:,:5]
        y_train = ori_training_data[:,5].reshape(-1,1)
        x_test = testing_data[:,:5]
        y_test = testing_data[:,5].reshape(-1,1)
        y_true = y_test.copy()

        x_scaler = MinMaxScaler()
        x_train = x_scaler.fit_transform(x_train)
        y_scaler = MinMaxScaler()
        y_train = y_scaler.fit_transform(y_train)
        x_test_scaler = MinMaxScaler()
        x_test = x_test_scaler.fit_transform(x_test)
        y_test_scaler = MinMaxScaler()
        y_test = y_test_scaler.fit_transform(y_test)
      
        # training with TCN
        x_train = np.reshape(x_train, (x_train.shape[0], time_steps, 1))

        model = Sequential()
        model.add(TCN(25))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(x_train, y_train, epochs=10, batch_size=100)
        model.summary()

        # prediction
        y_pred = model.predict(x_test)
        y_pred = y_test_scaler.inverse_transform(y_pred)
        y_pred_all = np.concatenate((y_pred_all, y_pred), axis=0)
        y_true_all = np.concatenate((y_true_all, y_true), axis=0)

    # evaluation
    y_pred_all = y_pred_all[1:,:] # remove first element (0,0)
    y_true_all = y_true_all[1:,:]
    # np.save('data/data_plot/T_y_pred_all.npy', y_pred_all)
    # np.save('data/data_plot/T_y_true_all.npy', y_true_all)
    plot_pred(y_true_all, y_pred_all, 'TCN')
    plot_residuals(y_true_all, y_pred_all, 'TCN')

    # Return to zero
    y_true_all = np.zeros((1,1))
    y_pred_all = np.zeros((1,1))