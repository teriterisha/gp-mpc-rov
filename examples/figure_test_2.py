from scipy import io
import matplotlib.pyplot as plt
import numpy as np
# 导入数据
# 拷贝用：data/height/Tv-dis/   data/height/No_dis/0.25/ data/height/Con_dis/ Speed_up
input_file = 'data/height/Speed_up/'
all_data_PID = io.loadmat(input_file + 'PID.mat')
all_data_Nominal = io.loadmat(input_file + 'Nominal.mat')
all_data_OFF = io.loadmat(input_file + 'Offline-GP.mat')
all_data_ON = io.loadmat(input_file + 'Online-GP.mat')
all_data = []
# 导入参考数据
state_all_ref = all_data_PID['ref']
t = all_data_PID['t']
# 分别导入不同算法数据
real_PID = all_data_PID['s_real']
real_Nominal = all_data_Nominal['s_real']
real_OFF = all_data_OFF['s_real']
real_ON = all_data_ON['s_real']

rmse_PID = all_data_PID['rmse']
rmse_Nominal = all_data_Nominal['rmse']
rmse_OFF = all_data_OFF['rmse']
rmse_ON = all_data_ON['rmse']
rmse = {'rmse_PID' : rmse_PID, 'rmse_Nominal': rmse_Nominal, 'rmse_OFF' : rmse_OFF, 'rmse_ON': rmse_ON}
print(rmse)
# # print(np.size(u,0))
# # print(u_PID[: , 0, 0])
# plt.figure('fig1')
# plt.title("trajectory")
# # 参考轨迹
# plt.plot(state_all_ref[:, 0], state_all_ref[:, 1], linestyle = '-', c='r', label = 'ref')

# # 真实轨迹
# plt.plot(real_PID[:, 0], real_PID[:, 1], linestyle = '-', label='PID')
# plt.plot(real_Nominal[:, 0], real_Nominal[:, 1], c='c', linestyle = '-', label='Nominal-MPC')
# plt.plot(real_OFF[:, 0], real_OFF[:, 1], c='y', linestyle = '-.', label = 'Offline-GP-MPC')
# plt.plot(real_ON[:, 0], real_ON[:, 1], c='g', linestyle = ':', label = 'Online-GP-MPC')
# plt.scatter(0, 0, c='k')
# plt.xlabel("x position(m)")
# plt.ylabel("y position(m)")
# plt.legend(loc='upper left')#绘制曲线图例，信息来自类型label
# # plt.scatter(state_all_ref[:, 0], state_all_ref[:, 1], 'r--')
# # 各状态对比
# y_label = ['x position(m)', 'y position(m)', 'yaw(rad)']
# for i in range(3):
#     s = str(i + 2)
#     s = 'fig' + s
#     plt.figure(s)
#     plt.xlim((0, 10))
#     plt.xlabel("time (s)")
#     plt.ylabel(y_label[i])
#     # plt.plot(t[0], state_all_ref[:, i],linestyle = ':', c='r', label = 'ref')
#     plt.plot(t[0], real_PID[:, i] - state_all_ref[:, i], c='b', label='PID')
#     plt.plot(t[0], real_Nominal[:, i] - state_all_ref[:, i], c='c', label='Nominal-MPC')
#     plt.plot(t[0], real_OFF[:, i]-  state_all_ref[:, i], c='y', label='Offline-GP-MPC')
#     plt.plot(t[0], real_ON[:, i] - state_all_ref[:, i], c='g', label='Online-GP-MPC')
#     plt.legend(loc='upper left')#绘制曲线图例，信息来自类型label

plt.figure('fig5')
i = 0
# plt.plot(t[0], state_all_ref[:, i],linestyle = ':', c='r', label = 'ref')
plt.xlim((0, 10))
plt.xlabel("time (s)")
plt.ylabel("Z error(m)")
plt.plot(t[0], real_PID[:, i] - state_all_ref[:, i], c='b', label='PID')
plt.plot(t[0], real_Nominal[:, i] - state_all_ref[:, i], c='c', label='Nominal-MPC')
plt.plot(t[0], real_OFF[:, i]-  state_all_ref[:, i], c='y', label='Offline-GP-MPC')
plt.plot(t[0], real_ON[:, i] - state_all_ref[:, i], c='g', label='Online-GP-MPC')
plt.legend(loc='upper left')#绘制曲线图例，信息来自类型label
plt.show()