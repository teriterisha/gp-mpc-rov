from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sys import path
path.append(r"./../") # Add gp_mpc pagkage to path

import numpy as np
import math
import casadi as ca
import matplotlib.pyplot as plt
from gp_mpc import Model, GP, MPC, plot_eig, lqr
import pyDOE
import time

"""note:目前拓展了水平面三自由度的运动学，使用修正后模型在某些情况下要优于名义模型，但是不稳定，可能是因为使用single-shoot的方法导致求解存在一些问题
之后可以考虑换成multiple-shoot方法
4-3note：1、点对点跟踪的轨迹效果以及响应速度目前看来并不取决于模型是否准确，可能有轻微影响，至少影响不大，影响因素更大在于控制器的参数设计，比如[0,0,0]~[5,5,0]的
点对点跟踪问题，明显优化曲线应该为doc中图片所示，所以在设计控制器时，应该对cost函数设计作特殊修正，让y轴的位置惩罚最大，指导轨迹效果。
目前问题是使用名义模型响应速度比真实模型还要快，修正模型效果与真实模型类似，不知道什么毛病
2、求解器warning问题：目前尚不知道为什么会出现这种警告，但是似乎不影响结果，将离散间隔时间改为0.1s后不在报这种警告
"""
"""to do:实时更新与代码整理，需要重写gp类
        轨迹跟踪任务
        模型复杂度选择
"""

def ode_nominal(x, u):
    """ROV model
    拓展模型ing，目标是拓展为4自由度模型
    此处定义x[0]为x轴向速度  x[1]为y轴轴向速度   x[2]为yaw角角速度
          u[0]为x轴推力  u[1]为y轴推力  u[2]为z轴转动力矩
    """
    dxdt = [
            (u[0] - 30.0 * x[0] * ca.fabs(x[0]) - 0.0 * x[0]) / 8.4 - x[2] * x[1],
            (u[1] - 80.0 * x[1] * ca.fabs(x[1]) - 0.0 * x[1]) / (8.4 +10.0) + x[2] * x[0],
            (u[2] - 0.05 * x[2] * ca.fabs(x[2]) - 1.0 * x[2]) / 0.2,
    ]
    return  dxdt

def ode_real(x, u):
    """ ROV model
    目前考虑了未知水动力，质量测量不准确，外界恒定干扰三个要素
    """
    dxdt = [
            (u[0] - 34.96 * x[0] * ca.fabs(x[0]) - 0.0 * x[0] - 0.15 * x[1]) / (8.4 + 2.6) - x[2] * x[1],
            (u[1] - 103.25 * x[1] * ca.fabs(x[1]) - 0.26 * x[1] - 0.15 * x[0]) / (8.4 + 18.5) + x[2] * x[0],
            (u[2] - 0.084 * x[2] * ca.fabs(x[2]) - 0.895 * x[2]) / 0.26,
    ]
    return  dxdt

def ode_error(x, u):
    """ 2D ROV model
    """
    dxdt = [
            (u[0] - 34.96 * x[0] * ca.fabs(x[0]) - 0.0 * x[0] - 0.15 * x[1]) / (8.4 + 2.6) - (u[0] - 30.0 * x[0] * ca.fabs(x[0]) - 0.0 * x[0]) / 8.4,
            (u[1] - 103.25 * x[1] * ca.fabs(x[1]) - 0.26 * x[1] - 0.15 * x[0]) / (8.4 + 18.5) - (u[1] - 80.0 * x[1] * ca.fabs(x[1]) - 0.0 * x[1]) / (8.4 +10.0),
            (u[2] - 0.084 * x[2] * ca.fabs(x[2]) - 0.895 * x[2]) / 0.26 - (u[2] - 0.05 * x[2] * ca.fabs(x[2]) - 1.0 * x[2]) / 0.2,
    ]
    return  dxdt

def my_covSEard_np(x, z, ell, sf2):
    """ GP squared exponential kernel
        Copyright (c) 2018, Helge-André Langåker
    """
    dist = np.sum((x - z)**2 / ell**2, 1)
    return sf2 * np.exp(-.5 * dist)

def my_covSEard(x_train, x, u, ell, sf2, N):
    """ GP squared exponential kernel
        Copyright (c) 2018, Helge-André Langåker
    """
    x_predict = ca.repmat(ca.vertcat(x, u).T,N,1)
    ell_rept = ca.repmat(ell,N,1)
    dist = ca.sum2((x_train - x_predict)**2 / ell_rept**2)
    return sf2 * ca.SX.exp(-.5 * dist)

def my_generate_training_data(ode, N, Nu, Nx, uub, ulb, xub, xlb, noise_cov, noise=True):
        """ Generate training data using latin hypercube design

        # Arguments:
            N:   Number of data points to be generated
            uub: Upper input range (Nu,1)
            ulb: Lower input range (Nu,1)
            xub: Upper state range (Ny,1)
            xlb: Lower state range (Ny,1)
            noise_cov: 噪声矩阵(Ny*Ny)
        # Returns:
            Z: Matrix (N, Nx + Nu) with state x and inputs u at each row
            Y: Matrix (N, Nx) where each row is the state x at time t+dt,
                with the input from the same row in Z at time t.
        """
        # Make sure boundry vectors are numpy arrays
        uub = np.array(uub)
        ulb = np.array(ulb)
        xub = np.array(xub)
        xlb = np.array(xlb)

        # Predefine matrix to collect noisy state outputs
        Y = np.zeros((N, Nx))

        # Create control input design using a latin hypecube
        # Latin hypercube design for unit cube [0,1]^Nu
        if Nu > 0:
            U = pyDOE.lhs(Nu, samples=N, criterion='maximin')
             # Scale control inputs to correct range
            for k in range(N):
                U[k, :] = U[k, :] * (uub - ulb) + ulb
        else:
            U = []

        # Create state input design using a latin hypecube
        # Latin hypercube design for unit cube [0,1]^Ny
        X = pyDOE.lhs(Nx, samples=N, criterion='maximin')

        # Scale state inputs to correct range
        for k in range(N):
            X[k, :] = X[k, :] * (xub - xlb) + xlb

        for i in range(N):
            if Nu > 0:
                u_t = U[i, :]    # control input for simulation
            else:
                u_t = []
            x_t = X[i, :]    # state input for simulation

            # Simulate system with x_t and u_t inputs for deltat time
            Y[i, :] = ode(x_t, u_t)

            # Add normal white noise to state outputs
            if noise:
                Y[i, :] += np.random.multivariate_normal(
                                np.zeros((Nx)), noise_cov)

        # Concatenate previous states and inputs to obtain overall input to GP model
        if Nu > 0:
            Z = np.hstack([X, U])
        else:
            Z = X
        return Z, Y

def my_compare_date(gp,X_test,ode_real,ode_nominal):
    y_real = []
    y_nominal = []
    y_predict = []
    # print(X_test[:, :Nx], X_test[:, Nx:])

    cov = np.zeros((Nx + Nu, Nx + Nu))

    # x_t = np.array([1.0,1.0])
    # u_t = np.array([1.0,1.0])
    # y_error_t, cov = gp.predict(x_t,u_t,cov)
    # print(x_t,y_error_t)

    for i in range(N_test):
        x_t = X_test[i, :Nx]
        u_t = X_test[i, Nx:]

        y_real_t = ode_real(x_t,u_t)
        y_real.append(y_real_t)

        y_error_t, _ = gp.predict(x_t,u_t,cov)

        y_nominal_t = ode_nominal(x_t,u_t)
        y_nominal.append(y_nominal_t)

        y_predict_t = y_nominal_t + np.array(y_error_t).flatten()
        y_predict.append(y_predict_t)

    # print(np.array(y_real))
    # print(np.array(y_nominal))
    # print(np.array(y_predict))

    y_real_np = np.array(y_real)
    y_nominal_np = np.array(y_nominal)
    y_predict_np = np.array(y_predict)
    x_x = np.array([-60,60])
    for i in range(Nx):
        plt.figure(i)
        plt.scatter(y_predict_np[:, i], y_real_np[:, i], s= 10, c='r')
        plt.scatter(y_nominal_np[:, i], y_real_np[:, i], s= 10, c='g')
        plt.plot(x_x, x_x)

    # plt.figure(2)
    # plt.scatter(y_real_np[:, 0], y_real_np[:, 1], s= 10, c='r')
    # plt.scatter(y_nominal_np[:, 0], y_nominal_np[:, 1], s= 10, c='g')

    # plt.figure(3)
    # plt.scatter(y_real_np[:, 0], y_real_np[:, 1], s= 10, c='r')
    # plt.scatter(y_predict_np[:, 0], y_predict_np[:, 1], s= 10, c='b')
    plt.show()
    return y_real_np,y_nominal_np,y_predict_np

def get_ode_ca(ode, Nx, Nu):
    """将list形式的ode表达式构建成casadi表达式
    """
    X = ca.SX.sym('x',Nx)
    U = ca.SX.sym('u',Nu)
    ode_ca = ca.Function("ode_ca", [X, U], [ca.vertcat(*ode(X, U))])
    return ode_ca

def get_gp_correct_model(ode_nominal, x_train, y_train, xlb, xub, ulb, uub,X_test):
    """目前的想法是构建一个函数，通过输入标称模型与训练数据，得到一个gp修正过的模型表达式
       输入参数：ode_nomial(标称模型)，x_train(N * (Nx + Nu)), y_train(N * Nx)  
       已完成
    """
    N, Nt = x_train.shape
    Nx = y_train.shape[1]
    Nu = Nt - Nx

    X = ca.SX.sym('x',Nx)
    U = ca.SX.sym('u',Nu)
    Y = ca.SX.sym('y',Nx)
    ell_s = ca.SX.sym('ell', 1, Nt)
    sf2_s = ca.SX.sym('sf2')
    x_train_ca = ca.SX(x_train)
    gp = GP(x_train, y_train, mean_func='zero', normalize=False, xlb=xlb, xub=xub, ulb=ulb,
        uub=uub, optimizer_opts=None)
    # 得到GP类训练完成的超参数以及计算的ivk矩阵，这样带来的问题可能是ivk不能根据采集到的数据实时更新，GP类代码中更新的部分存在一些问题，之后需要修复
    invk = gp.get_invk()
    hyper_gp = gp.get_hyper_parameters()
    alpha_np = np.zeros((N,1))
    covSE = ca.Function('covSE', [X, U, ell_s, sf2_s],
                        [my_covSEard(x_train_ca, X, U, ell_s, sf2_s, N)])
    for i in range(Nx):
        alpha_np = invk[i,:,:] @ y_train[:,i]
        alpha_ca = ca.SX(alpha_np)
        ell_ca = ca.SX(hyper_gp['length_scale'][i,:])
        sf2 = ca.SX(hyper_gp['signal_var'][i])
        Y[i] = ca.mtimes(covSE(X, U, ell_ca, sf2).T, alpha_ca)
    y_predict_fun = ca.Function('y_predict',[X, U], [Y])
    # 此处代码的作用是验证预测函数的正确性，导入的参数X_test是同样的目的，目前已经验证了是正确的，随后可以删去
    # cov = np.zeros((Nt, Nt))
    # gp_fun_predict,_ = gp.predict(X_test[0, :Nx],X_test[0, Nx:],cov)
    # my_gp_predict = y_predict_fun(X_test[0, :Nx],X_test[0, Nx:])
    # print(gp_fun_predict, my_gp_predict)

    # 建立标称模型
    Y = ca.vertcat(*ode_nominal(X, U))
    ode_nominal_ca = ca.Function("ode_nominal", [X, U], [ca.vertcat(*ode_nominal(X, U))])
    # 联合构建修正后的模型
    correct_fun_predict = ca.Function("correct_fun",[X,U],[ode_nominal_ca(X, U) + y_predict_fun(X, U)])
    # print(correct_fun_predict)
    return correct_fun_predict

def my_rk4_fun(ode, dt, Nx, Nu):
    """ Create discrete RK4 model """
    X = ca.SX.sym('x',Nx)
    U = ca.SX.sym('u',Nu)
    ode_casadi = ca.Function("ode", [X, U], [ode(X,U)])
    k1 = ode_casadi(X, U)
    k2 = ode_casadi(X + dt/2*k1, U)
    k3 = ode_casadi(X + dt/2*k2, U)
    k4 = ode_casadi(X + dt*k3, U)
    xrk4 = X + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    rk4 = ca.Function("ode_rk4", [X, U], [xrk4])
    # print(rk4)
    return rk4

def dyna_2_kine(ode_dyna, Nx, Nu):
    """此函数的目的是为了将动力学方程扩展为运动学方程，目前的方法是将状态量扩展为位置+速度，位置微分直接定义为速度，这会导致问题，
    因为旋转角度的存在，在世界坐标系下位置的微分并不等于速度，目前暂且不考虑旋转运动，之后考虑旋转的话这个函数需要重写
    """
    x_extern = ca.SX.sym('x_extern', 2 * Nx)
    dx_extern = ca.SX.sym('dx_extern', 2 * Nx)
    U = ca.SX.sym('u',Nu)
    dx_extern[:Nx] = x_extern[Nx:]
    dx_extern[Nx:] = ode_dyna(x_extern[Nx:], U)
    Kine_fun_ca = ca.Function("kine_fun_ca", [x_extern, U], [dx_extern])
    # print(Kine_fun_ca)
    return Kine_fun_ca

def dyna_2_kine_new(ode_dyna, Nx, Nu):
    """此函数的目的是为了将动力学方程扩展为运动学方程，目前的方法是将状态量扩展为位置+速度，此函数目前仅仅为水平面三自由度运动提供转换
    已完成
    """
    x_extern = ca.SX.sym('x_extern', 2 * Nx)
    dx_extern = ca.SX.sym('dx_extern', 2 * Nx)
    U = ca.SX.sym('u',Nu)
    dx_extern[0] = x_extern[3] * ca.cos(x_extern[2]) - x_extern[4] * ca.sin(x_extern[2])
    dx_extern[1] = x_extern[3] * ca.sin(x_extern[2]) + x_extern[4] * ca.cos(x_extern[2])
    dx_extern[2] = x_extern[5]
    dx_extern[3:] = ode_dyna(x_extern[3:], U)
    Kine_fun_ca = ca.Function("kine_fun_ca", [x_extern, U], [dx_extern])
    # print(Kine_fun_ca)
    return Kine_fun_ca

def get_MPC_solver(state_fun, Nx, Nu, N_pedict, Q, Q_end, R):
    """建立MPC求解器，输入为离散状态方程，状态量个数，控制量个数，预测步长，状态惩罚矩阵，终端惩罚矩阵，控制惩罚矩阵
    """
    u_all_horizon = ca.SX.sym('u_all_horizon', Nu, N_pedict) # N步内的控制输出
    x_all_horizon = ca.SX.sym('x_all_horizon', Nx, N_pedict + 1) # N+1步的系统状态，通常长度比控制多1
    p = ca.SX.sym('P', 2 * Nx) # 输入量，目前任务为点到点运动控制，所以输入量为初始状态与结束状态
    x_all_horizon[:,0] = p[:Nx]
    for i in range(N_pedict):
        x_all_horizon[:, i + 1] = state_fun(x_all_horizon[:, i], u_all_horizon[:, i])
    obj = 0
    # 构建损失函数
    for i in range(N_pedict):
        obj = obj + ca.mtimes([(x_all_horizon[:, i]-p[Nx:]).T, Q, x_all_horizon[:, i]-p[Nx:]]) + ca.mtimes([u_all_horizon[:, i].T, R, u_all_horizon[:, i]])
    obj += ca.mtimes([(x_all_horizon[:, N_pedict]-p[Nx:]).T, Q_end, x_all_horizon[:, N_pedict]-p[Nx:]])
    # 尝试采用single-shoot方法，这种方法在之前的仿真过程中速度要快于multiple-shoot法，但是可能会有稳定性的问题，之后可以尝试更换方法
    g = []
    for i in range(N_pedict + 1):
        for j in range(Nx):
            g.append(x_all_horizon[j, i])
    # 定义NLP问题，'f'为目标函数，'x'为需寻找的优化结果（优化目标变量），'p'为系统参数，'g'为等式约束条件
    # 需要注意的是，用SX表达必须将所有表示成标量或者是一维矢量的形式
    nlp_prob = {'f': obj, 'x': ca.reshape(u_all_horizon, -1, 1), 'p':p, 'g':ca.vertcat(*g)}
    opts_setting = {'ipopt.max_iter':100, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6}
    # 最终目标，获得求解器
    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)
    # print(solver)
    return solver

dt = .1                    # Sampling time
Nx = 3                      # Number of states
Nu = 3                      # Number of inputs
R_n = np.eye(Nx) * 1e-6     # Covariance matrix of added noise
R_n[2,2] *= 1e-3
# Limits in the training data
ulb = [-50.0, -50.0, -1.0]    # input bound
uub = [50.0, 50.0, 1.0]      # input bound
xlb = [-4., -4., -np.pi / 2]    # state bound
xub = [4., 4., np.pi / 2]      # state bound

N = 40          # Number of training data
N_test = 100    # Number of test data

""" Create simulation model and generate training/test data"""
""" 这个model类目前想要弃用，因为直接进行了离散化，如果想要预测加速度误差应得到加速度的关系式"""
# model          = Model(Nx=Nx, Nu=Nu, ode=ode_error, dt=dt, R=R_n, clip_negative=True)
# X, Y           = model.generate_training_data(N, uub, ulb, xub, xlb, noise=True)
# X_test, Y_test = model.generate_training_data(N_test, uub, ulb, xub, xlb, noise=True)

"""生成训练集与测试集，通过GP类进行验证
"""
noise_cov = np.eye(Nx) * 1e-3
X, Y = my_generate_training_data(ode_error, N, Nu, Nx, uub, ulb, xub, xlb, noise_cov, noise=True)
X_test, Y_test = my_generate_training_data(ode_error, N_test, Nu, Nx, uub, ulb, xub, xlb, noise_cov, noise=True)

# print(X, Y)

gp = GP(X, Y, mean_func='zero', normalize=False, xlb=xlb, xub=xub, ulb=ulb,
        uub=uub, optimizer_opts=None)

gp.print_hyper_parameters()
hyper_gp = gp.get_hyper_parameters()


gp.validate(X_test, Y_test)
# my_compare_date(gp,X_test,ode_real,ode_nominal)

"""得到GP的invk矩阵，方便计算mean
    目前可以通过np来计算得到预测均值，下一步应该写成casadi形式"""
# alpha = np.zeros((N,Nx))
# invk = gp.get_invk()
# for i in range(Nx):
#     alpha[:,i] = invk[i,:,:] @ Y[:,i]
# my_predict = [0,0]
# my_predict[0] = my_covSEard_np(X, X_test[0, :], hyper_gp['length_scale'][0,:],hyper_gp['signal_var'][0]).reshape(1,-1) @ alpha[:,0] 
# my_predict[1] = my_covSEard_np(X, X_test[0, :], hyper_gp['length_scale'][1,:],hyper_gp['signal_var'][1]).reshape(1,-1) @ alpha[:,1] 
# print(np.array(my_predict).flatten())
# cov = np.zeros((4,4))
# gp_fun_predict,_ = gp.predict(X_test[0, :Nx],X_test[0, Nx:],cov)
# print(gp_fun_predict)
"""通过训练数据与名义模型得到矫正后模型的casadi表达式"""
my_gp_correct_fun_ca = get_gp_correct_model(ode_nominal, X, Y, xlb, xub, ulb, uub, X_test)
"""验证输出矫正后模型的效果，目前的方法是手动输出对比，随后应该采用类似与GP.validate函数实现，可以删除"""
# for i in range(N_test):
#     X_test_ca = ca.SX(X_test[i, :Nx].reshape(-1,1))
#     U_test_ca = ca.SX(X_test[i, Nx:].reshape(-1,1))
#     my_predict_ca = my_gp_correct_fun_ca(X_test_ca, U_test_ca)
#     y_real_t = ode_real(X_test[i, :Nx],X_test[i, Nx:])
#     y_nominal_t = ode_nominal(X_test[i, :Nx],X_test[i, Nx:])
#     print(y_real_t, y_nominal_t, my_predict_ca)
"""将得到的矫正后动力学模型拓展为运动学模型，这个例子中目前只涉及到了平面内两个自由度平动，所以位置量可以直接积分得到"""
kine_fun_correct_ca = dyna_2_kine_new(my_gp_correct_fun_ca, Nx, Nu)
"""这部分代码是为了验证拓展后模型的正确性,验证结果还行，之后可以删去"""
# ode_real_ca = get_ode_ca(ode_real, Nx, Nu)
# kine_real_ca = dyna_2_kine_new(ode_real_ca, Nx, Nu)
# ode_nominal_ca = get_ode_ca(ode_nominal, Nx, Nu)
# kine_nominal_ca = dyna_2_kine_new(ode_nominal_ca, Nx, Nu)
# # 构建拓展后的状态信息，定义位置为[0, 0]
# x_test_extern = ca.SX(np.vstack([np.zeros((Nx, 1)), X_test[0, :Nx].reshape(-1,1)]))
# u_test = ca.SX(X_test[0, Nx:].reshape(-1,1))
# print(x_test_extern, u_test)
# dx_test_extern_real = kine_real_ca(x_test_extern, u_test)
# dx_test_extern_nominal = kine_nominal_ca(x_test_extern, u_test)
# dx_test_extern_predict = kine_fun_correct_ca(x_test_extern, u_test)
# print(dx_test_extern_real, dx_test_extern_nominal, dx_test_extern_predict)
"""将矫正后的模型进行离散化，使用RK4方法，得到离散后的casadi表达式
   加入了真实模型的RK4离散化，目的是为了得到真实的下一时刻状态
"""
kine_fun_correct_rk4 = my_rk4_fun(kine_fun_correct_ca, dt, 2 * Nx, Nu)
ode_real_ca = get_ode_ca(ode_real, Nx, Nu)
kine_real_ca = dyna_2_kine_new(ode_real_ca, Nx, Nu)
kine_real_rk4 = my_rk4_fun(kine_real_ca, dt, 2 * Nx, Nu)
"""这部分代码是为了验证离散后预测模型的准确性，方法是同样将真实模型离散化，得到真实状态下一时刻的结果，并且与预测结果对比，验证效果还行,之后可以删去"""
# 真实模型扩展+离散化
# ode_real_ca = get_ode_ca(ode_real, Nx, Nu)
# kine_real_ca = dyna_2_kine(ode_real_ca, Nx, Nu)
# kine_real_rk4 = my_rk4_fun(kine_real_ca, dt, 2 * Nx, Nu)
# 名义模型扩展+离散化
ode_nominal_ca = get_ode_ca(ode_nominal, Nx, Nu)
kine_nominal_ca = dyna_2_kine_new(ode_nominal_ca, Nx, Nu)
kine_nominal_rk4 = my_rk4_fun(kine_nominal_ca, dt, 2 * Nx, Nu)
# # 构建拓展后的状态信息，定义位置为[0, 0]
# x_test_extern = ca.SX(np.vstack([np.zeros((Nx, 1)), X_test[0, :Nx].reshape(-1,1)]))
# u_test = ca.SX(X_test[0, Nx:].reshape(-1,1))
# print(x_test_extern, u_test)
# # 根据test数据预测下一时刻状态，对比结果
# state_next_nominal = kine_nominal_rk4(x_test_extern, u_test)
# state_next_real = kine_real_rk4(x_test_extern, u_test)
# state_next_predict = kine_fun_correct_rk4(x_test_extern, u_test)
# print(state_next_real, state_next_nominal, state_next_predict)
# nominal_error = state_next_real - state_next_nominal
# predict_error = state_next_real - state_next_predict
# print(nominal_error, predict_error)
"""随后应该根据矫正过的模型进行预测控制，可以使用名义模型进行对比
   已完成，在干扰下可以达到一个较好的控制效果，之后将下面这部分整合成子函数
"""
Nx = 2 * Nx
N_pedict = 3 # 预测布长
# 状态约束（此处的位置约束重定义，速度约束与上文保持相同
state_l = [-np.inf, -np.inf, -np.pi, -4., -0.01, -np.pi / 2] 
state_u = [np.inf, np.inf, np.pi, 4., 0.01, np.pi / 2] 
control_l = [-50.0, -50.0, -1.0]   
control_u = [50.0, 50.0, 1.0]    

Q = np.eye(Nx) * 1e-1           # 状态惩罚矩阵
Q[0,0] = 1
Q[1,1] = 100
Q_end = 2 * np.eye(Nx)     # 终端惩罚矩阵
R = np.zeros((Nu, Nu))     # 控制量惩罚矩阵
# 复制粘贴用： kine_fun_correct_rk4  kine_nominal_rk4 kine_real_rk4
solver = get_MPC_solver(kine_real_rk4, Nx, Nu, N_pedict, Q, Q_end, R)

lbx = [] # 最低约束条件(nlp问题的求解变量，该问题中为N_predict的控制输出)
ubx = [] # 最高约束条件
lbg = [] # 等式最低约束条件(nlp问题的等式，该问题中为下一时刻状态与此时刻的状态与控制量)
ubg = [] # 等式最高约束条件
for _ in range(N_pedict):
    lbx.append(control_l)
    ubx.append(control_u)
for _ in range(N_pedict + 1):
    lbg.append(state_l)
    ubg.append(state_u)
lbg = np.array(lbg).reshape(-1, 1)
ubg = np.array(ubg).reshape(-1, 1)
lbx = np.array(lbx).reshape(-1, 1)
ubx = np.array(ubx).reshape(-1, 1)
# print(lbg, ubg)
# 仿真条件和相关变量
t0 = 0.0 # 仿真时间
x_init_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
x_end_list = [3.0, 1.5, np.pi / 2, 0.0, 0.0, 0.0]
u_init_list = [0.0, 0.0, 0.0] * N_pedict
x_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(-1, 1) # 每一步的状态（未开始仿真前存储初始状态）
x_end = np.array([5.0, 5.0, 0.0, 0.0, 0.0, 0.0]).reshape(-1, 1)  # 末了状态
u_init = np.array([0.0, 0.0, 0.0] * N_pedict).reshape(-1, Nu)    # nlp问题求解的初值，求解器可以在此基础上优化至最优值

x_predict = [] # 存储预测状态
u_c = [] # 存储控制全部计算后的控制指令
t_c = [] # 保存时间
x_real = []  # 存储真实状态
sim_time = 30.0 # 仿真时长
index_t = [] # 存储时间戳，以便计算每一步求解的时间

## 开始仿真
mpciter = 0 # 迭代计数器
start_time = time.time() # 获取开始仿真时间
### 终止条件为此刻状态和目标状态的欧式距离小于0.01或者仿真超时
while(np.linalg.norm(x_init - x_end) > 1e-2 and mpciter-sim_time / dt <0.0 ): 
    # 初始化优化参数
    c_p = np.concatenate((x_init, x_end))
    # 初始化优化目标变量
    init_control = ca.reshape(u_init, -1, 1)
    ### 计算结果并且计时
    t_ = time.time()
    res = solver(x0 = init_control, p = c_p, lbg = lbg, lbx = lbx, ubg = ubg, ubx = ubx)
    index_t.append(time.time() - t_)
    # 获得最优控制结果u
    u_sol = ca.reshape(res['x'], Nu, N_pedict).T # 将其恢复U的形状定义
    u_this = u_sol[0, :] # 仅将第一步控制作为真实控制输出
    state_next_predict = kine_fun_correct_rk4(x_init, u_this)
    state_next_real = kine_real_rk4(x_init, u_this)
    x_predict.append(state_next_predict)
    x_real.append(state_next_real)
    # 更新下一时刻状态
    x_init = state_next_real
    u_init = u_sol
    # 计数器+1
    mpciter = mpciter + 1
x_real_np = np.array(x_real)[:,:,0]
a =np.array([x_init_list])
x_real_np = np.concatenate((a,x_real_np),axis=0)
x_predict_np = np.array(x_predict)[:,:,0]
print(x_real_np)
print(mpciter)
plt.figure()
plotsca_x = np.array([0.0,5.0])
plt.scatter(plotsca_x, plotsca_x, s= 5, c='b')
plt.plot(x_real_np[:, 0], x_real_np[:, 1])
plt.scatter(x_real_np[:, 0], x_real_np[:, 1], s= 2, c='r')
plt.show()