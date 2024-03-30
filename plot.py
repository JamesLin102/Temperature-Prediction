import numpy as np
import matplotlib.pyplot as plt
from predict import plot_pred, plot_residuals

file_path = 'data/data_plot/temp_min/' # temp_max oe temp_min

# TCN + CTGAN (processsed)
TCP_y_true_all = np.load(file_path + 'TCP_y_true_all.npy')
TCP_y_pred_all = np.load(file_path + 'TCP_y_pred_all.npy')
plot_pred(TCP_y_true_all, TCP_y_pred_all, 'TCN + CTGAN (processed)')
plot_residuals(TCP_y_true_all, TCP_y_pred_all, 'TCN + CTGAN (processed)')

# TCN + CTGAN 
TC_y_true_all = np.load(file_path + 'TC_y_true_all.npy')
TC_y_pred_all = np.load(file_path + 'TC_y_pred_all.npy')
plot_pred(TC_y_true_all, TC_y_pred_all, 'TCN + CTGAN')
plot_residuals(TC_y_true_all, TC_y_pred_all, 'TCN + CTGAN')

# TCN
T_y_true_all = np.load(file_path + 'T_y_true_all.npy')
T_y_pred_all = np.load(file_path + 'T_y_pred_all.npy')
plot_pred(T_y_true_all, T_y_pred_all, 'TCN')
plot_residuals(T_y_true_all, T_y_pred_all, 'TCN')

# comparison
start_duration = 100 # 100 500
end_duration = 200   # 200 600
plt.figure(figsize=(15,8))
plt.plot(TCP_y_true_all[start_duration:end_duration], label='Real Temperature')
plt.plot(TCP_y_pred_all[start_duration:end_duration], label='Prediction of TCN+CTGAN(processed)')
plt.plot(TC_y_pred_all[start_duration:end_duration], label='Prediction of TCN+CTGAN')
plt.plot(T_y_pred_all[start_duration:end_duration], label='Prediction of TCN')
plt.xlabel('Time (day)', fontsize=18)
plt.ylabel('Temperature (°C)', fontsize=18)
plt.xlim([0,100])
plt.title('Comparison of Algorithms from 2012/4/9~2012/7/18', fontsize=18) # 2012/4/9~2012/7/18 or 2013/5/14~2013/8/22
plt.grid()
plt.legend(fontsize=15)
plt.show()