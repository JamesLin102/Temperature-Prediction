# -*- coding: utf-8 -*-
import numpy as np
from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from sklearn.ensemble import AdaBoostRegressor
from sklearn import neighbors
from sklearn.linear_model  import LogisticRegression
#from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import csv

#direction = os.path.dirname(os.path.abspath(__file__))
#save_path = direction +'/figure'

def score_calculation(y, y_pred):
    MAE = np.round(mean_absolute_error(y, y_pred), 3)
    RMSE = np.round(np.sqrt(mean_squared_error(y, y_pred)), 3)
    R2_Score = np.round(r2_score(y, y_pred), 3)
    
    print('MAE: ' + str(MAE))
    print('RMSE: ' + str(RMSE))
    print('R2 Score: ' + str(R2_Score))
    
    return MAE, RMSE, R2_Score
    
def plot_pred(y, y_pred, model_name):
    
    residuals = y_pred - y
    res_abs = np.abs(residuals)
    
    th_1 = 2.5   #Define this value in your case
    th_2 = 5   #Define this value in your case
    r1_idx = np.where(res_abs <= th_1)
    r2_idx = np.where((res_abs > th_1) & (res_abs <= th_2))
    r3_idx = np.where(res_abs > th_2)
    
    #Calculate Error
    MAE, RMSE, R2_Score = score_calculation(y, y_pred)
        
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111)
#    plt.scatter(y, y_pred, color='b', alpha=0.15, s=40)
    
    plt.scatter(y[r1_idx], y_pred[r1_idx], c='royalblue', 
                alpha=0.15, s=40, label=r'|R|$\leq$'+str(th_1))
    plt.scatter(y[r2_idx], y_pred[r2_idx], c='yellowgreen', 
                alpha=0.15, s=40, label=str(th_1)+r'<|R|$\leq$'+str(th_2))
    plt.scatter(y[r3_idx], y_pred[r3_idx], c='orange', 
                alpha=0.15, s=40, label='|R|>'+str(th_2))
    
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=1.5)
#    plt.plot([y.min(), y.max()], [y.min()-th_1, y.max()-th_1], 'r--', lw=0.5)
#    plt.plot([y.min(), y.max()], [y.min()+th_1, y.max()+th_1], 'r--', lw=0.5)
#    plt.plot([y.min(), y.max()], [y.min()-th_2, y.max()-th_2], 'r--', lw=0.5)
#    plt.plot([y.min(), y.max()], [y.min()+th_2, y.max()+th_2], 'r--', lw=0.5)
    plt.title(model_name + ' Prediction Results')
    plt.xlabel('Actual Value')
    plt.ylabel('Predicted Value')
    info_show = 'MAE: ' + str(MAE) + '\n'\
                'RMSE: ' + str(RMSE) + '\n'\
                'R2 Score: ' + str(R2_Score) + '\n'
    plt.text(0.8, 0.05, info_show, ha='left', va='center', transform=ax.transAxes)
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()
    #plt.savefig(save_path + '/Pred_' + model_name + '.png')

def plot_residuals(y, y_pred, model_name):
    
    residuals = y_pred - y
    res_abs = np.abs(residuals)
#    print(residuals)
    
    th_1 = 2.5   #Define this value in your case
    th_2 = 5   #Define this value in your case
    r1_idx = np.where(res_abs <= th_1)
    r2_idx = np.where((res_abs > th_1) & (res_abs <= th_2))
    r3_idx = np.where(res_abs > th_2)
    
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111)
    grid = plt.GridSpec(4, 4, wspace=0.5, hspace=0.5)

    main_ax = plt.subplot(grid[0:3,1:4])
#    plt.plot(y_pred, residuals,'ok',markersize=3,alpha=0.2)
    plt.scatter(y_pred[r1_idx], residuals[r1_idx], c='royalblue', 
                alpha=0.15, s=40, label=r'|R|$\leq$'+str(th_1))
    plt.scatter(y_pred[r2_idx], residuals[r2_idx], c='yellowgreen', 
                alpha=0.15, s=40, label=str(th_1)+r'<|R|$\leq$'+str(th_2))
    plt.scatter(y_pred[r3_idx], residuals[r3_idx], c='orange', 
                alpha=0.15, s=40, label='|R|>'+str(th_2))
    plt.plot([y_pred.min(), y_pred.max()], [0, 0], 'r--', lw=1.5)
#    plt.plot([y_pred.min(), y_pred.max()], [th_1, th_1], 'k--', lw=0.5)
#    plt.plot([y_pred.min(), y_pred.max()], [-th_1, -th_1], 'k--', lw=0.5)
#    plt.plot([y_pred.min(), y_pred.max()], [th_2, th_2], 'k--', lw=0.5)
#    plt.plot([y_pred.min(), y_pred.max()], [-th_2, -th_2], 'k--', lw=0.5)
    plt.legend(loc='upper left')
    plt.grid()
    plt.title('Residuals for ' + model_name + ' Model')
    
    
    y_hist = plt.subplot(grid[0:3,0], xticklabels=[], sharey=main_ax)
    plt.hist(residuals,60,orientation='horizontal',color='g')
    y_hist.invert_xaxis()
    plt.ylabel('Residuals')
    plt.grid()
    
    x_hist = plt.subplot(grid[3,1:4],yticklabels=[],sharex=main_ax)
    plt.hist(y_pred,60,orientation='vertical', color='g')
    x_hist.invert_yaxis()
    plt.xlabel('Predicted Value')
    plt.grid()  
    
    MAE, RMSE, R2_Score = score_calculation(y, y_pred) 
    info_show = 'MAE: ' + str(MAE) + '\n'\
                'RMSE: ' + str(RMSE) + '\n'\
                'R2 Score: ' + str(R2_Score) + '\n'
    plt.text(0.0, 0.05, info_show, ha='left', va='center', transform=ax.transAxes)
    plt.show()
    #plt.savefig(save_path + '/Res_' + model_name + '.png')

