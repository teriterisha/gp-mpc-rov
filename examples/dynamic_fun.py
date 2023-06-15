import casadi as ca
import math
import numpy as np

def get_ode_ca(ode, Nx, Nu):
    """将list形式的ode表达式构建成casadi表达式
    """
    X = ca.SX.sym('x',Nx)
    U = ca.SX.sym('u',Nu)
    ode_ca = ca.Function("ode_ca", [X, U], [ca.vertcat(*ode(X, U))])
    return ode_ca

def get_ode_ca_distrub(ode, x_dis, y_dis, Nx, Nu):
    """将list形式的ode表达式构建成casadi表达式
    """
    X = ca.SX.sym('x',Nx)
    U = ca.SX.sym('u',Nu)
    T = ca.SX.sym('t')
    ode_ca = ca.Function("ode_ca", [X, U, T], [ca.vertcat(*ode(X, U, x_dis, y_dis, T))])
    return ode_ca 

def x_distrub(t):
    return 75 * ca.sin(math.pi * t)
    # return 0

def y_distrub(t):
    return 75 * ca.cos(math.pi * t)
    # return 0

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

def my_rk4_fun_distrub(ode, dt, Nx, Nu):
    """ 创建带干扰的rk4模型 """
    X = ca.SX.sym('x',Nx)
    U = ca.SX.sym('u',Nu)
    T = ca.SX.sym('t')

    ode_casadi = ca.Function("ode", [X, U, T], [ode(X, U, T)])
    k1 = ode_casadi(X, U, T)
    k2 = ode_casadi(X + dt/2*k1, U, T + dt / 2)
    k3 = ode_casadi(X + dt/2*k2, U, T + dt / 2)
    k4 = ode_casadi(X + dt*k3, U, T + dt)
    xrk4 = X + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    rk4 = ca.Function("ode_rk4", [X, U, T], [xrk4])
    # print(rk4)
    return rk4

###################################################################三自由度###################################################
def dyna_2_kine_nominal(ode_dyna, Nx, Nu):
    """此函数的目的是为了将动力学方程扩展为运动学方程，目前的方法是将状态量扩展为位置+速度，此函数目前仅仅为水平面三自由度运动提供转换(不考虑干扰)
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

def dyna_2_kine_real(ode_dyna, Nx, Nu):
    """此函数的目的是为了将动力学方程扩展为运动学方程，目前的方法是将状态量扩展为位置+速度，此函数目前仅仅为水平面三自由度运动提供转换(考虑干扰，与ode_real_distrub配合使用)
    已完成
    """
    x_extern = ca.SX.sym('x_extern', Nx)
    dx_extern = ca.SX.sym('dx_extern', Nx)
    U = ca.SX.sym('u',Nu)
    T = ca.SX.sym('t')

    dx_extern[0] = x_extern[3] * ca.cos(x_extern[2]) - x_extern[4] * ca.sin(x_extern[2])
    dx_extern[1] = x_extern[3] * ca.sin(x_extern[2]) + x_extern[4] * ca.cos(x_extern[2])
    dx_extern[2:] = ode_dyna(x_extern[2:], U, T)
    Kine_fun_ca = ca.Function("kine_fun_ca", [x_extern, U, T], [dx_extern])
    # print(Kine_fun_ca)
    return Kine_fun_ca

###################################################################四自由度###################################################
def dyna_2_kine_nominal_4(ode_dyna, Nx, Nu):
    """此函数的目的是为了将动力学方程扩展为运动学方程，目前的方法是将状态量扩展为位置+速度，此函数目前为四自由度运动(不考虑干扰)
    """
    x_extern = ca.SX.sym('x_extern', 2 * Nx)
    dx_extern = ca.SX.sym('dx_extern', 2 * Nx)
    U = ca.SX.sym('u',Nu)

    dx_extern[0] = x_extern[4] * ca.cos(x_extern[3]) - x_extern[5] * ca.sin(x_extern[3])
    dx_extern[1] = x_extern[4] * ca.sin(x_extern[3]) + x_extern[5] * ca.cos(x_extern[3])
    dx_extern[2] = x_extern[6]
    dx_extern[3] = x_extern[7]
    dx_extern[4:] = ode_dyna(x_extern[4:], U)
    Kine_fun_ca = ca.Function("kine_fun_ca", [x_extern, U], [dx_extern])
    # print(Kine_fun_ca)
    return Kine_fun_ca

def dyna_2_kine_real_4(ode_dyna, Nx, Nu):
    """此函数的目的是为了将动力学方程扩展为运动学方程，目前的方法是将状态量扩展为位置+速度，此函数目前为四自由度运动(考虑干扰，与ode_real_4_distrub配合使用)
    已完成
    """
    x_extern = ca.SX.sym('x_extern', Nx)
    dx_extern = ca.SX.sym('dx_extern', Nx)
    U = ca.SX.sym('u',Nu)
    T = ca.SX.sym('t')

    dx_extern[0] = x_extern[4] * ca.cos(x_extern[3]) - x_extern[5] * ca.sin(x_extern[3])
    dx_extern[1] = x_extern[4] * ca.sin(x_extern[3]) + x_extern[5] * ca.cos(x_extern[3])
    dx_extern[2] = x_extern[6]
    dx_extern[3:] = ode_dyna(x_extern[3:], U, T)
    Kine_fun_ca = ca.Function("kine_fun_ca", [x_extern, U, T], [dx_extern])
    # print(Kine_fun_ca)
    return Kine_fun_ca

###################################################################轨迹生成###################################################
def x_fun(t):
    """定义参考轨迹，目前为圆形，原点在圆心，半径为2,10s仿真时间，运行一圈结束"""
    # return math.sin(math.pi * t / 5) * 3  #圆
    # return 0.5 * t / 10 * math.sin(math.pi * t / 5) # 放大圆
    # if t < 5:                                    #L转向
    #     return 1.5 * t
    # return 7.5
    return  1.5 * math.cos(math.pi * t / 5) / 1.0  - 0                   #8字形
    # return 0.5 * t                                   #直线

def y_fun(t):
    # return -math.cos(math.pi * t / 5) * 3 + 3 #圆
    # return 0.5 * t / 10 * math.cos(math.pi * t / 5)  # 放大圆
    # if t < 5:                                    #L转向
    #     return 0.0
    # return 1.5 * t - 7.5
    return 1.0 * math.cos(math.pi * t / 5) * math.sin(math.pi * t / 5) / 1.0               #8字形
    # return 0.5 #直线

def z_fun(t):
    """暂定z轴不动"""
    # return 0.0
    return 0.15 * t

def yaw_fun(t):
    # return math.pi * t / 5 #圆 放大圆
    # if t < 5:                                    #L转向
    #     return 0.0
    # return math.pi / 2
    return 0                   #8字形

def vx_fun(t):
    # return math.pi / 5 * 3 #圆
    # return 0.5 * t / 10 * math.pi / 5 # 放大圆
    # return 1.5                    #L转向
    return - math.sin(math.pi * t / 5) * 1.5 * math.pi / 5    #8字形
    # return 0.5               # 直线

def vy_fun(t):
    # return 0.0 # 圆 L转向放 大圆
    # return 0.05  # 放大圆
    return ((math.cos(math.pi * t / 5) + math.sin(math.pi * t / 5)) * (math.cos(math.pi * t / 5) - math.sin(math.pi * t / 5)) / 1.0) * math.pi / 5 #8字形

def vz_fun(t):
    # return 0.0
    return 0.15

def theta_fun(t):
    # return math.pi / 5 # 圆
    return 0.0              #8字形 L转向

def get_mpc_parameter(x_fun, y_fun, yaw_fun, vx_fun, vy_fun, theta_fun, dt, t, N_predict):
    """得到参考轨迹（三自由度）
    """
    state_ref = np.empty((6, N_predict + 1))
    for i in range(N_predict + 1):
        state_ref[:, i] = np.array([x_fun(t + dt * i), y_fun(t + dt * i), yaw_fun(t + dt * i), vx_fun(t + dt * i), vy_fun(t + dt * i), theta_fun(t + dt * i)])
    return state_ref

def get_mpc_parameter_4(x_fun, y_fun, z_fun, yaw_fun, vx_fun, vy_fun, vz_fun ,theta_fun, dt, t, N_predict):
    """得到参考轨迹（四自由度）
    """
    state_ref = np.empty((8, N_predict + 1))
    for i in range(N_predict + 1):
        state_ref[:, i] = np.array([x_fun(t + dt * i), y_fun(t + dt * i),  z_fun(t + dt * i), yaw_fun(t + dt * i), vx_fun(t + dt * i), vy_fun(t + dt * i), vz_fun(t + dt * i),theta_fun(t + dt * i)])
    return state_ref

def merge_model(nominal_fun, error_fun, Nx, Nu):
    """将名义模型与误差模型混合,要求传入的均为casadi形式的表达式"""
    X = ca.SX.sym('x',Nx)
    if Nu != 0:
        U = ca.SX.sym('u',Nu)
        # 联合构建修正后的模型
        correct_fun = ca.Function("correct_fun",[X,U],[nominal_fun(X, U) + error_fun(X, U)])
    else:
        correct_fun = ca.Function("correct_fun",[X],[nominal_fun(X) + error_fun(X)])
    return correct_fun

def merge_kine_model(nominal_kine_fun, error_fun, Nx_kine, Nu_kine, Nx_error):
    """目前想法是将外界干扰通过GPS预测并且将干扰考虑加入模型中"""
    X = ca.SX.sym('x', Nx_kine)   
    U = ca.SX.sym('u', Nu_kine)
    Y = ca.SX.sym('Y', Nx_kine)
    X_e = ca.SX.sym('x_e', Nx_error)
    Y = nominal_kine_fun(X, U)
    Y[:Nx_error] = Y[:Nx_error] + error_fun(X_e)
    correct_fun = ca.Function("correct_fun", [X, U],[Y])
    return correct_fun

def error_fun_test():
    """仅用于测试merge_kine_model函数功能，随后可以删去"""
    X = ca.SX.sym('x', 2) 
    Y = ca.SX.sym('y', 2) 
    error = ca.SX(2)
    Y[0] = -0.0
    Y[1] = -0.0 
    error_fun = ca.Function("error_fun",[X],[Y])
    return error_fun   

def get_acc(kine_fun, rk4_fun, dt, state_now, control_now):
    """用于计算当前时刻的加速度，由于将外界干扰加在世界坐标系中进行处理，需要将其换算到身体坐标系
    rk4方程的dt应与输入dt保持一致"""
    yaw_now = state_now[2]
    ode_state_now = kine_fun(state_now, control_now)
    vx_now = ode_state_now[0]
    vy_now = ode_state_now[1]
    yaw_speed_now = ode_state_now[2]

    vx_body_now, vy_body_now = word2body(vx_now, vy_now, yaw_now)

    state_next = rk4_fun(state_now, control_now)
    yaw_next = state_next[2]
    ode_state_next = kine_fun(state_next, control_now)
    vx_next = ode_state_next[0]
    vy_next = ode_state_next[1]
    yaw_speed_next = ode_state_next[2]

    vx_body_next, vy_body_next = word2body(vx_next, vy_next, yaw_next)

    acc_x = (vx_body_next - vx_body_now) / dt
    acc_y = (vy_body_next - vy_body_now) / dt
    acc_yaw = (yaw_speed_next - yaw_speed_now) / dt
    return acc_x, acc_y, acc_yaw

def word2body(v, yaw):
    """将世界坐标系速度换算到本体坐标系"""
    v_body = [0., 0., 0.]
    v_body[0] = math.cos(yaw) * v[0] + math.sin(yaw) * v[1]
    v_body[1] = math.cos(yaw) * v[1] - math.sin(yaw) * v[0]
    v_body[2] = v[2]
    return v_body