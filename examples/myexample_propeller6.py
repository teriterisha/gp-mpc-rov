from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sys import path
path.append(r"./../") # Add gp_mpc pagkage to path

from my_gp_class import MYGP
from dynamic_fun import *
from my_model import *
from data_process import *
from PID import PID

import numpy as np
import math
import casadi as ca
import matplotlib.pyplot as plt
import time
from scipy import io

"""PID控制案例"""
dt = .05                    # Sampling time
Nx = 1                      # Number of states
Nu = 1                      # Number of inputs
# Limits in the training data
ulb = -150.0      # input bound
uub = 150.0        # input bound
xlb = [-1.0]    # state bound
xub = [1.0]      # state bound

# 真实模型扩展+离散化
ode_real_ca = get_ode_ca_distrub_z(z_real_distrub, z_distrub, Nx, Nu)
kine_real_ca = dyna_2_kine_real_z(ode_real_ca, 2 * Nx, Nu)
kine_real_rk4 = my_rk4_fun_distrub(kine_real_ca, dt, 2 * Nx, Nu)

pid_z_pos = PID(10, 1, 10, 8., -8.)

pid_vz = PID(50, 10, 10, 200., -200.,)

# 初始状态
x_init_list = [0.0, 0.0]
x_init = np.array(x_init_list).reshape(-1, 1) # 每一步的状态（未开始仿真前存储初始状态）

x_predict = [] # 存储预测状态
u_c = [] # 存储控制全部计算后的控制指令
t_c = [0.0] # 保存时间
x_real = []  # 存储真实状态
sim_time = 10.0 # 仿真时长
index_t = [] # 存储时间戳，以便计算每一步求解的时间
t0 = 0.0
# 得到参考轨迹参数（函数命名有问题）
state_ref = get_mpc_parameter_z(z_fun, vz_fun, dt, t0, 0)
state_all_ref = state_ref[:, 0].reshape(-1, 1)

### 开始仿真
mpciter = 0 # 迭代计数器
while(mpciter-sim_time / dt <0.0 ): 
    # 获得控制输入
    error_pos = state_ref[:1, 0] - x_init[:1, 0]
    # print(error_pos)
    v_ref = np.array([0.0])
    v_ref[0] = pid_z_pos.update(error_pos[0])

    error_v = v_ref - x_init[1:, 0]
    u_this = np.array([0.0])       # 仅将第一步控制作为真实控制输出
    add1 = pid_vz.update(error_v[0])

    u_this[0] = max(min(add1, uub), ulb)
    # u_this[1] = 0
    print(x_init)
    u = np.array([u_this])
    u_c.append(u)
    # print(u_this)
    # 更新下一时刻状态
    t0 += dt
    t_c.append(t0)
    state_ref = get_mpc_parameter_z(z_fun, vz_fun, dt, t0, 0)
    state_all_ref = np.concatenate((state_all_ref,state_ref[:, 0].reshape(-1,1)),axis=1)

    state_next_real = kine_real_rk4(x_init, u_this, t0)
    x_real.append(state_next_real)

    x_init = state_next_real

    # 计数器+1
    mpciter = mpciter + 1

x_real_np = np.array(x_real)[:,:,0]
a =np.array([x_init_list])
x_real_np = np.concatenate((a,x_real_np),axis=0)    

# 计算误差及相关指标
error_z_axis = x_real_np[:, 0] - state_all_ref[0, :].T
error_distance = error_z_axis ** 2
# print(error_distance)
error_distance_max = math.sqrt(np.max(error_distance))
mse = np.sum(error_distance) / len(error_z_axis)
rmse = math.sqrt(mse)
print(rmse, error_distance_max)

z_data = {'s_real':x_real_np, 'ref': state_all_ref.T, 't' : t_c, 'u' : u_c ,'mse' : mse, 'rmse' : rmse, 'ME' : error_distance_max}
io.savemat('data/height/Con_dis/PID.mat', z_data)
# print(x_real_np)
# print(mpciter)
plt.figure('fig2')
plt.plot(t_c, x_real_np[:, 0]-state_all_ref[0, :].T)
# plt.scatter(tc, state_all_ref[0, :].T, s= 2, c='r')
plt.show()