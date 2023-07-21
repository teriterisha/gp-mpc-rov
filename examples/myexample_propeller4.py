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
from scipy import io

"""z轴案例
"""

def get_MPC_solver(state_fun, Nx, Nu, N_predict, Q, Q_end, R):
    """建立MPC求解器，输入为离散状态方程，状态量个数，控制量个数，预测步长，状态惩罚矩阵，终端惩罚矩阵，控制惩罚矩阵
    """
    u_all_horizon = ca.SX.sym('u_all_horizon', Nu, N_predict) # N步内的控制输出
    x_all_horizon = ca.SX.sym('x_all_horizon', Nx, N_predict + 1) # N+1步的系统状态，通常长度比控制多1
    p = ca.SX.sym('P', Nx, N_predict + 2) # 输入量，目前任务为轨迹跟踪，所以输入为初始状态与N+1步内全部的参考状态量，初始状态在前
    x_all_horizon[:, 0] = p[:, 0]
    for i in range(N_predict):
        x_all_horizon[:, i + 1] = state_fun(x_all_horizon[:, i], u_all_horizon[:, i])
    obj = 0
    # 构建损失函数
    for i in range(N_predict):
        obj = obj + ca.mtimes([(x_all_horizon[:, i]-p[:, i + 1]).T, Q, x_all_horizon[:, i]-p[:, i + 1]]) + ca.mtimes([u_all_horizon[:, i].T, R, u_all_horizon[:, i]])
    obj += ca.mtimes([(x_all_horizon[:, N_predict]-p[:, -1]).T, Q_end, x_all_horizon[:, N_predict]-p[:, -1]])
    # 尝试采用single-shoot方法，这种方法在之前的仿真过程中速度要快于multiple-shoot法，但是可能会有稳定性的问题，之后可以尝试更换方法
    g = []
    for i in range(N_predict + 1):
        for j in range(Nx):
            g.append(x_all_horizon[j, i])
    # 定义NLP问题，'f'为目标函数，'x'为需寻找的优化结果（优化目标变量），'p'为系统参数，'g'为等式约束条件
    # 需要注意的是，用SX表达必须将所有表示成标量或者是一维矢量的形式
    nlp_prob = {'f': obj, 'x': ca.reshape(u_all_horizon, -1, 1), 'p':ca.reshape(p, -1, 1), 'g':ca.vertcat(*g)}
    opts_setting = {'ipopt.max_iter':100, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6}
    # 最终目标，获得求解器
    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)
    # print(solver)
    return solver

dt = .05                    # Sampling time
Nx = 1                      # Number of states
Nu = 1                      # Number of inputs
R_n = np.eye(Nx) * 1e-6     # Covariance matrix of added noise
# Limits in the training data
ulb = [-150.0]      # input bound
uub = [150.0]        # input bound
xlb = [-1.0]    # state bound
xub = [1.0]      # state bound

N = 100          # Number of training data
N_test = 40    # Number of test data

"""生成训练集与测试集，通过GP类进行验证
"""
noise_cov = np.eye(Nx) * 1e-3
X, Y = my_generate_training_data(z_error, N, Nu, Nx, uub, ulb, xub, xlb, noise_cov, noise=True)
X_test, Y_test = my_generate_training_data(z_error, N_test, Nu, Nx, uub, ulb, xub, xlb, noise_cov, noise=True)

# np.save('X_test_propeller_1.npy',X_test)
# np.save('Y_test_propeller_1.npy',Y_test)

# X_test = np.load("X_test_propeller_1.npy")
# Y_test = np.load("Y_test_propeller_1.npy")

gp = MYGP(X, Y, X_test, Y_test)

"""通过训练数据与名义模型得到矫正后模型的casadi表达式"""
ode_nominal_ca = get_ode_ca(z_nominal, Nx, Nu)
my_gp_error_fun_ca = gp.get_mean_fun()
my_gp_correct_fun_ca = merge_model(ode_nominal_ca, my_gp_error_fun_ca, Nx, Nu)
"""将得到的矫正后动力学模型拓展为运动学模型，这个例子中目前只涉及到了平面内两个自由度平动，所以位置量可以直接积分得到"""
kine_fun_correct_pre_ca = dyna_2_kine(my_gp_correct_fun_ca, Nx, Nu)
# error_fun = error_fun_test()
# kine_fun_correct_ca = merge_kine_model(kine_fun_correct_pre_ca, error_fun, 6, 3, 2)
"""将矫正后的模型进行离散化，使用RK4方法，得到离散后的casadi表达式
   加入了真实模型的RK4离散化，目的是为了得到真实的下一时刻状态
"""
kine_fun_correct_rk4 = my_rk4_fun(kine_fun_correct_pre_ca, dt, 2 * Nx, Nu)
# 真实模型扩展+离散化
ode_real_ca = get_ode_ca_distrub_z(z_real_distrub, z_distrub, Nx, Nu)
kine_real_ca = dyna_2_kine_real_z(ode_real_ca, 2 * Nx, Nu)
kine_real_rk4 = my_rk4_fun_distrub(kine_real_ca, dt, 2 * Nx, Nu)
# 模型准确但无干扰
ode_real_no_dis_ca = get_ode_ca(z_real, Nx, Nu)
kine_real_no_dis_ca = dyna_2_kine(ode_real_no_dis_ca, Nx, Nu)
kine_real_no_dis_rk4 = my_rk4_fun(kine_real_no_dis_ca, dt, 2 * Nx, Nu)
"""这部分代码是为了验证离散后预测模型的准确性，方法是同样将真实模型离散化，得到真实状态下一时刻的结果，并且与预测结果对比，验证效果还行,之后可以删去"""
# 名义模型扩展+离散化
kine_nominal_ca = dyna_2_kine(ode_nominal_ca, Nx, Nu)
kine_nominal_rk4 = my_rk4_fun(kine_nominal_ca, dt, 2 * Nx, Nu)

"""随后应该根据矫正过的模型进行预测控制，可以使用名义模型进行对比
   已完成，在干扰下可以达到一个较好的控制效果，之后将下面这部分整合成子函数 
"""
Nx = 2 * Nx
N_predict = 5 # 预测布长
# 状态约束（此处的位置约束重定义，速度约束与上文保持相同
state_l = [-np.inf, -1.0] 
state_u = [np.inf, 1.0] 
control_l = ulb  
control_u = uub   

Q = np.eye(Nx)  * 1e-2       # 状态惩罚矩阵
Q[0,0] = 10
Q[1,1] = 0.1
Q_end = 5 * np.eye(Nx)     # 终端惩罚矩阵
R = np.zeros((Nu, Nu))     # 控制量惩罚矩阵
# 复制粘贴用： kine_fun_correct_rk4  kine_nominal_rk4 kine_real_no_dis_rk4
solver = get_MPC_solver(kine_nominal_rk4, Nx, Nu, N_predict, Q, Q_end, R)

lbx = [] # 最低约束条件(nlp问题的求解变量，该问题中为N_predict的控制输出)
ubx = [] # 最高约束条件
lbg = [] # 等式最低约束条件(nlp问题的等式，该问题中为下一时刻状态与此时刻的状态与控制量)
ubg = [] # 等式最高约束条件
# 初始状态
x_init_list = [0.0]
x_init = np.array(x_init_list).reshape(-1, 1) # 每一步的状态（未开始仿真前存储初始状态）
for _ in range(N_predict):
    lbx.append(control_l)

    ubx.append(control_u)
for _ in range(N_predict + 1):
    lbg.append(state_l)
    ubg.append(state_u)
lbg = np.array(lbg).reshape(-1, 1)
ubg = np.array(ubg).reshape(-1, 1)
lbx = np.array(lbx).reshape(-1, 1)
ubx = np.array(ubx).reshape(-1, 1)

# 仿真条件和相关变量
t0 = 0.0 # 仿真时间
x_init_list = [0.0, 0.0]

u_init_list = [0.0] * N_predict
x_init = np.array(x_init_list).reshape(-1, 1) # 每一步的状态（未开始仿真前存储初始状态）
u_init = np.array(u_init_list).reshape(-1, Nu)    # nlp问题求解的初值，求解器可以在此基础上优化至最优值
state_ref = get_mpc_parameter_z(z_fun, vz_fun, dt, t0, N_predict)
state_all_ref = state_ref[:, 0].reshape(-1, 1)

x_predict = [] # 存储预测状态
u_c = [] # 存储控制全部计算后的控制指令
t_c = [0.0] # 保存时间
x_real = []  # 存储真实状态
sim_time = 10.0 # 仿真时长
index_t = [] # 存储时间戳，以便计算每一步求解的时间
t0 = 0.0
## 开始仿真
mpciter = 0 # 迭代计数器
start_time = time.time() # 获取开始仿真时间

### 开始仿真
mpciter = 0 # 迭代计数器
while(mpciter-sim_time / dt <0.0 ): 
    # 初始化优化参数
    c_p = np.concatenate((x_init, state_ref), axis = 1).T.reshape(-1, 1)
    # 初始化优化目标变量
    init_control = ca.reshape(u_init, -1, 1)
    ### 计算结果并且计时
    t_ = time.time()
    res = solver(x0 = init_control, p = c_p, lbg = lbg, lbx = lbx, ubg = ubg, ubx = ubx)
    index_t.append(time.time() - t_)
    # 获得最优控制结果u
    u_sol = ca.reshape(res['x'], Nu, N_predict).T # 将其恢复U的形状定义
    u_this = u_sol[0, :] # 仅将第一步控制作为真实控制输出
    # 对比，如果不动态更新，有一个推进器出故障
    print(x_init)
    print(u_this)
    # u_this[0,0] = 0.0
    # print(u_this)
    # 更新下一时刻状态
    u_c.append(u_this)
    t0 += dt
    state_ref = get_mpc_parameter_z(z_fun, vz_fun, dt, t0, N_predict)
    state_all_ref = np.concatenate((state_all_ref,state_ref[:, 0].reshape(-1,1)),axis=1)

    state_next_predict = kine_nominal_rk4(x_init, u_this)
    state_next_real = kine_real_rk4(x_init, u_this, t0)
    x_predict.append(state_next_predict)
    x_real.append(state_next_real)
    t_c.append(t0)
    x_init = state_next_real
    u_init = u_sol
    # 计数器+1
    mpciter = mpciter + 1

tc = np.array(t_c)
x_real_np = np.array(x_real)[:,:,0]
a =np.array([x_init_list])
x_real_np = np.concatenate((a,x_real_np),axis=0)
x_predict_np = np.array(x_predict)[:,:,0]
# 计算误差及相关指标
error_z_axis = x_real_np[:, 0] - state_all_ref[0, :].T
error_distance = error_z_axis ** 2
error_distance_max = math.sqrt(np.max(error_distance))
mse = np.sum(error_distance) / len(error_z_axis)
rmse = math.sqrt(mse)
print(rmse, error_distance_max)

z_data = {'s_real':x_real_np, 'ref': state_all_ref.T, 't' : t_c, 'u' : u_c ,'mse' : mse, 'rmse' : rmse, 'ME' : error_distance_max}
# 拷贝用 Nominal  Offline-GP 
io.savemat('data/height/Con_dis/Nominal.mat', z_data)
# print(x_real_np)
# print(mpciter)
plt.figure('fig2')
plt.plot(tc, x_real_np[:, 0]-state_all_ref[0, :].T)
# plt.scatter(tc, state_all_ref[0, :].T, s= 2, c='r')
plt.show()