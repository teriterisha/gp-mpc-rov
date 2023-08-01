from scipy import io
import matplotlib.pyplot as plt
import numpy as np
import math
# 导入数据
# 拷贝用：data/lemniscate/Tv_dis_with_ero/   data/lemniscate/Tv_dis_No_ero/  data/lemniscate/No_dis_No_ero  data/lemniscate/Con_dis_No_ero  
# data/L  data/circle/r=2/  data/circle/r=2.5/  data/circle/r=3/
input_file = 'data/circle/r=3/'
all_data_PID = io.loadmat(input_file + 'PID.mat')
all_data_Nominal = io.loadmat(input_file + 'Nominal.mat')
all_data_OFF = io.loadmat(input_file + 'Offline-GP.mat')
all_data_ON = io.loadmat(input_file + 'Online-GP.mat')

input_file_2 = 'data/height/No_dis/0.75/'
Z_data_PID = io.loadmat(input_file_2 + 'PID.mat')
Z_data_Nominal = io.loadmat(input_file_2 + 'Nominal.mat')
Z_data_OFF = io.loadmat(input_file_2 + 'Offline-GP.mat')
Z_data_ON = io.loadmat(input_file_2 + 'Online-GP.mat')

# 导入参考数据
state_all_ref = all_data_PID['ref']
Z_ref = Z_data_Nominal['ref']
t = all_data_PID['t']
# 分别导入不同算法数据
real_PID = all_data_PID['s_real']
real_Nominal = all_data_Nominal['s_real']
real_OFF = all_data_OFF['s_real']
real_ON = all_data_ON['s_real']

realZ_PID = Z_data_PID['s_real']
realZ_Nominal = Z_data_Nominal['s_real']
realZ_OFF = Z_data_OFF['s_real']
realZ_ON = Z_data_ON['s_real']

all_error = [[real_PID[:, 0] - state_all_ref[:, 0], real_PID[:, 1] - state_all_ref[:, 1], realZ_PID[:, 0] - Z_ref[:, 0]], 
            [real_Nominal[:, 0] - state_all_ref[:, 0], real_Nominal[:, 1] - state_all_ref[:, 1], realZ_Nominal[:, 0] - Z_ref[:, 0]], 
            [real_OFF[:, 0] - state_all_ref[:, 0], real_OFF[:, 1] - state_all_ref[:, 1], realZ_OFF[:, 0] - Z_ref[:, 0]], 
            [real_ON[:, 0] - state_all_ref[:, 0], real_ON[:, 1] - state_all_ref[:, 1], realZ_ON[:, 0] - Z_ref[:, 0]]]
error2_distance = []
for i in range(4):
    error_this = all_error[i][0] ** 2 + all_error[i][1] ** 2 + all_error[i][2] ** 2
    error2_distance.append(error_this)
rmse = [math.sqrt(np.sum(error2_distance[0]) / len(error2_distance[0])), math.sqrt(np.sum(error2_distance[1]) / len(error2_distance[1])), 
        math.sqrt(np.sum(error2_distance[2]) / len(error2_distance[2])), math.sqrt(np.sum(error2_distance[3]) / len(error2_distance[3]))]
me = [math.sqrt(np.max(error2_distance[0])), math.sqrt(np.max(error2_distance[1])), math.sqrt(np.max(error2_distance[2])), math.sqrt(np.max(error2_distance[3]))]
print('PID: RMSE = %.3f ME = %.3f' % (rmse[0], me[0]))
print('Nominal: RMSE = %.3f ME = %.3f' % (rmse[1], me[1]))
print('OFF: RMSE = %.3f ME = %.3f' % (rmse[2], me[2]))
print('ON: RMSE = %.3f ME = %.3f' % (rmse[3], me[3]))