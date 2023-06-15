from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sys import path
path.append(r"./../") # Add gp_mpc pagkage to path

from my_gp_class import MYGP
from dynamic_fun import *
from my_model import *
from data_process import *

import numpy as np
import math
import casadi as ca
import matplotlib.pyplot as plt
import time

dt = 1e-5                   # Sampling time
Nx = 3                      # Number of states
Nu = 3                      # Number of inputs

dt_for_sim = 0.01           # 用于仿真的时间间隔

ode_real_ca = get_ode_ca(ode_real, Nx, Nu)
kine_real_ca = dyna_2_kine_real(ode_real_ca, Nx, Nu)
kine_real_rk4 = my_rk4_fun(kine_real_ca, dt, 2 * Nx, Nu)

ode_nominal_ca = get_ode_ca(ode_nominal, Nx, Nu)
kine_nominal_ca = dyna_2_kine_nominal(ode_nominal_ca, Nx, Nu)
kine_nominal_rk4 = my_rk4_fun(kine_nominal_ca, dt, 2 * Nx, Nu)

kine_real_rk4_for_sim = my_rk4_fun(kine_real_ca, dt, 2 * Nx, Nu)

x_init = np.array([0.5, 0.0, np.pi / 4, 1.0, 0.0, 0.0]).reshape(-1, 1) # 每一步的状态（未开始仿真前存储初始状态）
u_init = np.array([10., 10.0, 0.1]).reshape(-1, 1)    # nlp问题求解的初值，求解器可以在此基础上优化至最优值

acc_x_pre ,acc_y_pre, acc_yaw_pre = get_acc(kine_nominal_ca, kine_nominal_rk4, dt, x_init, u_init)
acc_x_real ,acc_y_real, acc_yaw_real = get_acc(kine_real_ca, kine_real_rk4, dt, x_init, u_init)

acc_x_error = acc_x_real - acc_x_pre
acc_y_error = acc_y_real - acc_y_pre
acc_yaw_error = acc_yaw_real - acc_yaw_pre

# print(acc_x_pre ,acc_y_pre, acc_yaw_pre, acc_x_real ,acc_y_real, acc_yaw_real)
print(acc_x_error, acc_y_error, acc_yaw_error)