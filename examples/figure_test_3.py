from scipy import io
import matplotlib.pyplot as plt
import numpy as np
# 导入数据
# 拷贝用：data/lemniscate/Tv_dis_with_ero/   data/lemniscate/Tv_dis_No_ero/  data/lemniscate/No_dis_No_ero  data/lemniscate/Con_dis_No_ero  
# data/L  data/circle/r=2/  data/circle/r=2.5/  data/circle/r=3/
input_file = 'data/lemniscate/Tv_dis_with_ero/'
all_data_PID = io.loadmat(input_file + 'PID.mat')
all_data_Nominal = io.loadmat(input_file + 'Nominal.mat')
all_data_OFF = io.loadmat(input_file + 'Offline-GP.mat')
all_data_ON = io.loadmat(input_file + 'Online-GP.mat')

input_file_2 = 'data/height/Tv_error/'
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

rmse_PID = all_data_PID['rmse']
rmse_Nominal = all_data_Nominal['rmse']
rmse_OFF = all_data_OFF['rmse']
rmse_ON = all_data_ON['rmse']
# print(np.size(u,0))
# print(u_PID[: , 0, 0])
plt.figure('fig1', figsize=(15,15))
plt.axis('off')
plt.title("trajectory")
# 参考轨迹
# plt.plot(real_PID[:, 0], real_PID[:, 1], linestyle = '-', c='r', label = 'ref')

# 真实轨迹
ax = plt.axes(projection='3d')
ax.plot3D(real_PID[:, 0], real_PID[:, 1], realZ_PID[:, 0], linestyle = '-', label='PID')
ax.plot3D(real_Nominal[:, 0], real_Nominal[:, 1], realZ_Nominal[:, 0], c='c', linestyle = '-', label='Nominal-MPC')
ax.plot3D(real_OFF[:, 0], real_OFF[:, 1], realZ_OFF[:, 0], c='y', linestyle = '-.', label = 'Offline-GP-MPC')
ax.plot3D(real_ON[:, 0], real_ON[:, 1], realZ_ON[:, 0], c='g', linestyle = ':', label = 'Online-GP-MPC')
ax.scatter3D(state_all_ref[:, 0], state_all_ref[:, 1], Z_ref[:, 0],s= 2, c='r', label = 'Reference')
ax.set(xlim=[-2, 2], ylim=[-1.0, 1.0], zlim=[0, 5])
ax.set_xlabel("x position(m)")
ax.set_ylabel("y position(m)")
ax.set_zlabel("z position(m)")
ax.set_title("trajectory")
plt.legend(loc='upper left')#绘制曲线图例，信息来自类型label
# plt.zlabel("z position(m)")
# plt.plot(real_PID[:, 0], real_PID[:, 1], linestyle = '-', label='PID')
# plt.plot(real_Nominal[:, 0], real_Nominal[:, 1], c='c', linestyle = '-', label='Nominal-MPC')
# plt.plot(real_OFF[:, 0], real_OFF[:, 1], c='y', linestyle = '-.', label = 'Offline-GP-MPC')
# plt.plot(real_ON[:, 0], real_ON[:, 1], c='g', linestyle = ':', label = 'Online-GP-MPC')
# plt.scatter(0, 0, c='k')
# plt.xlabel("x position(m)")
# plt.ylabel("y position(m)")
# plt.zlabel("z position(m)")
# plt.legend(loc='upper left')#c
# plt.scatter(state_all_ref[:, 0], state_all_ref[:, 1], 'r--')
# 各状态对比
y_label = ['x error(m)', 'y error(m)', 'yaw error(rad)']
fig, ax = plt.subplots(4, 1, figsize=(18, 10)) 
for i in range(3):
    plt.subplot(4, 1, i + 1)
    plt.xlim(0, 10.)
    plt.xticks(np.arange(0., 10.1, 0.5))
    plt.xlabel("time (s)")
    plt.ylabel(y_label[i])
    # plt.plot(t[0], state_all_ref[:, i],linestyle = ':', c='r', label = 'ref')
    plt.plot(t[0], real_PID[:, i] - state_all_ref[:, i], linestyle = ':', c='b', label='PID')
    plt.plot(t[0], real_Nominal[:, i] - state_all_ref[:, i], linestyle = '--', c='c', label='Nominal-MPC')
    plt.plot(t[0], real_OFF[:, i]-  state_all_ref[:, i], c='r', linestyle = '-.', label='Offline-GP-MPC')
    plt.plot(t[0], real_ON[:, i] - state_all_ref[:, i], c='k', linestyle = '-', label='Online-GP-MPC')
    # plt.legend(loc='upper left')#绘制曲线图例，信息来自类型label

plt.subplot(4, 1, 4)
i = 0
plt.xlabel("time (s)")
plt.ylabel('z error(m)')
plt.xlim(0, 10)
plt.xticks(np.arange(0., 10.1, 0.5))
plt.plot(t[0], realZ_PID[:, i] - Z_ref[:, i], linestyle = ':', c='b', label='PID')
plt.plot(t[0], realZ_Nominal[:, i] - Z_ref[:, i], linestyle = '--', c='c', label='Nominal-MPC')
plt.plot(t[0], realZ_OFF[:, i]-  Z_ref[:, i], c='r', linestyle = '-.', label='Offline-GP-MPC')
plt.plot(t[0], realZ_ON[:, i] - Z_ref[:, i], c='k', linestyle = '-', label='Online-GP-MPC')

lines, labels = fig.axes[-1].get_legend_handles_labels()
    
fig.legend(lines, labels, loc = 'upper right')
plt.show()