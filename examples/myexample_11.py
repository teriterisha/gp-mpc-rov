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

"""水平面PID控制
"""

dt = .05                    # Sampling time
Nx = 3                      # Number of states
Nu = 3                      # Number of inputs
# Limits in the training data
ulb = [-100.0, -100.0, -10.0]      # input bound
uub = [100.0, 100.0, 10.0]        # input bound
xlb = [-2., -2., -np.pi / 2]    # state bound
xub = [2., 2., np.pi / 2]      # state bound

# 所有PID类
pid_x_pos = PID(30, 5, 5, 2., -2.)
pid_y_pos = PID(30, 5, 5, 2., -2.)
pid_yaw_pos = PID(10, 0, 0, 10 * math.pi, -10 * math.pi)

pid_vx = PID(200, 0, 10, 100., -100.,)
pid_vy = PID(200, 0, 10, 100., -100.,)
pid_vyaw = PID(10, 0, 1, 10., -10.,)
# 真实模型扩展+离散化
ode_real_ca = get_ode_ca_distrub(ode_real_distrub, x_distrub, y_distrub, 4, 3)
kine_real_ca = dyna_2_kine_real(ode_real_ca, 6, 3)
kine_real_rk4 = my_rk4_fun_distrub(kine_real_ca, dt, 2 * Nx, Nu)
# 初始状态
x_init_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
x_init = np.array(x_init_list).reshape(-1, 1) # 每一步的状态（未开始仿真前存储初始状态）

x_predict = [] # 存储预测状态
u_c = [] # 存储控制全部计算后的控制指令
t_c = [] # 保存时间
x_real = []  # 存储真实状态
sim_time = 10.0 # 仿真时长
index_t = [] # 存储时间戳，以便计算每一步求解的时间
t0 = 0.0
# 得到参考轨迹参数（函数命名有问题）
state_ref = get_mpc_parameter(x_fun, y_fun, yaw_fun, vx_fun, vy_fun, theta_fun, dt, t0, 0)
state_all_ref = state_ref[:, 0].reshape(-1, 1)
# print(x_init, state_ref)
## 开始仿真
mpciter = 0 # 迭代计数器
### 开始仿真
while(mpciter-sim_time / dt <0.0 ): 
    # 获得控制输入
    error_pos = state_ref[:3, 0] - x_init[:3, 0]
    # print(error_pos)
    v_ref = np.array([0.0, 0.0, 0.0])
    v_ref[0] = pid_x_pos.update(error_pos[0])
    v_ref[1] = pid_y_pos.update(error_pos[1])
    v_ref[2] = pid_yaw_pos.update(error_pos[2])
    v_body_ref = word2body(v_ref, x_init[2])
    # print(v_body_ref)
    error_v = v_body_ref - x_init[3:, 0]
    u_this = np.array([0.0, 0.0, 0.0])       # 仅将第一步控制作为真实控制输出
    u_this[0] = pid_vx.update(error_v[0])
    u_this[1] = pid_vy.update(error_v[1])
    u_this[2] = pid_vyaw.update(error_v[2])
    # print(u_this)
    # 更新下一时刻状态
    t0 += dt
    state_ref = get_mpc_parameter(x_fun, y_fun, yaw_fun, vx_fun, vy_fun, theta_fun, t0, t0, 0)
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
error_x_axis = x_real_np[:, 0] - state_all_ref[0, :].T
error_y_axis = x_real_np[:, 1] - state_all_ref[1, :].T
error_distance = error_x_axis ** 2 + error_y_axis ** 2
print(error_distance)
error_distance_max = math.sqrt(np.max(error_distance))
mse = np.sum(error_distance) / len(error_x_axis)
rmse = math.sqrt(mse)
print(rmse, error_distance_max)
# print(x_real_np)
# print(mpciter)
plt.figure()
plt.plot(x_real_np[:, 0], x_real_np[:, 1])
# plt.scatter(x_real_np[:, 0], x_real_np[:, 1], s= 2, c='b')
plt.scatter(state_all_ref[0, :], state_all_ref[1, :], s= 2, c='r')
plt.show()