import casadi as ca
import math
import numpy as np
import matplotlib.pyplot as plt

from PID import PID
from dynamic_fun import *
"""微分方程更改为和水流速度相关的方程
"""
c_dis = {}
def x_dis(t):
    return 0.5

def y_dis(t):
    return 0.5

def z_dis(t):
    return 0.0

c_dis['x_dis'] = x_dis
c_dis['y_dis'] = y_dis
c_dis['z_dis'] = z_dis

def my_rk4_fun(ode, dt, Nx, Nu):
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

def get_ode_ca(ode, Nx, Nu):
    """将list形式的ode表达式构建成casadi表达式
    """
    X = ca.SX.sym('x',Nx)
    U = ca.SX.sym('u',Nu)
    T = ca.SX.sym('t')
    ode_ca = ca.Function("ode_ca", [X, U, T], [ca.vertcat(*ode(X, U, T))])
    return ode_ca 

def real_model(x, u, t):
    # 本体系数
    m = 7.31
    Iz = 0.15
    # 附加质量
    Xu_dot = 2.6
    Yv_dot = 18.5
    Zw_dot = 13.3
    Nr_dot = 0.28
    # 一阶阻尼
    Xu = 0.0
    Yv = 0.26
    Zw = 0.19
    Nr = 4.64
    # 二阶阻尼
    Xuu = 34.96
    Yvv = 103.25
    Zww = 74.23
    Nrr = 0.43
    #推进器角度
    sita = math.pi / 4
    p1 = math.cos(sita)
    p2 = math.sin(sita)
    l = 0.1
    dxdt = [
        ca.cos(x[3]) * x[4] - ca.sin(x[3]) * x[5],
        ca.sin(x[3]) * x[4] + ca.cos(x[3]) * x[5],
        x[6],
        x[7],
        ((u[0] + u[1]) * p1 - (u[2] + u[3]) * p1 
            + (x[5] - (ca.cos(x[3]) * c_dis['y_dis'](t) - ca.sin(x[3]) * c_dis['x_dis'](t))) * x[7] * (m + Yv_dot)
            - Xu * (x[4] - (ca.cos(x[3]) * c_dis['x_dis'](t) + ca.sin(x[3]) * c_dis['y_dis'](t)))
            - Xuu * ca.fabs(x[4] - (ca.cos(x[3]) * c_dis['x_dis'](t) + ca.sin(x[3]) * c_dis['y_dis'](t))) * (x[4] - (ca.cos(x[3]) * c_dis['x_dis'](t) + ca.sin(x[3]) * c_dis['y_dis'](t)))) / (m + Xu_dot),
        ((u[0] + u[2]) * p2 - (u[1] + u[3]) * p2
            - (x[4] - (ca.cos(x[3]) * c_dis['x_dis'](t) + ca.sin(x[3]) * c_dis['y_dis'](t))) * x[7] * (m + Xu_dot) 
            - Yv * (x[5] - (ca.cos(x[3]) * c_dis['y_dis'](t) - ca.sin(x[3]) * c_dis['x_dis'](t))) 
            - Yvv * ca.fabs((x[5] - (ca.cos(x[3]) * c_dis['y_dis'](t) - ca.sin(x[3]) * c_dis['x_dis'](t)))) * (x[5] - (ca.cos(x[3]) * c_dis['y_dis'](t) - ca.sin(x[3]) * c_dis['x_dis'](t)))) / (m + Yv_dot),
        ((u[4] + u[5])
            - Zw * (x[6] - c_dis['z_dis'](t)) 
            - Zww * ca.fabs(x[6] - c_dis['z_dis'](t)) * (x[6] - c_dis['z_dis'](t))) / (m + Zw_dot),
        ((u[0] + u[3]) * l - (u[1] + u[2]) * l
            + (Xu_dot - Yv_dot) * (x[4] - (ca.cos(x[3]) * c_dis['x_dis'](t) + ca.sin(x[3]) * c_dis['y_dis'](t))) * (x[5] - (ca.cos(x[3]) * c_dis['y_dis'](t) - ca.sin(x[3]) * c_dis['x_dis'](t))) 
            - Nr * x[7] 
            - Nrr * ca.fabs(x[7]) * x[7]) / (Iz + Nr_dot),
    ]
    return dxdt

Nx, Nu, t0, dt = 8, 6, .0, .05
ulb = -75.0      # input bound
uub = 75.0        # input bound

# 初始状态
x_init_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
x_init = np.array(x_init_list).reshape(-1, 1) # 每一步的状态（未开始仿真前存储初始状态）

x_predict = [] # 存储预测状态
u_c = [] # 存储控制全部计算后的控制指令
t_c = [0.0] # 保存时间
x_real = []  # 存储真实状态
sim_time = 10.0 # 仿真时长
index_t = [] # 存储时间戳，以便计算每一步求解的时间

ode_real_ca = get_ode_ca(real_model, Nx, Nu)
ode_real_rk4 = my_rk4_fun(ode_real_ca, dt, Nx, Nu)

pid_x_pos = PID(50, 0, 10, 8., -8.)
pid_y_pos = PID(50, 0, 10, 8., -8.)
pid_z_pos = PID(50, 0, 10, 8., -8.)
pid_yaw_pos = PID(50, 0, 10, 10 * math.pi, -10 * math.pi)

pid_vx = PID(50, 10, 10, 200., -200.,)
pid_vy = PID(50, 10, 10, 200., -200.,)
pid_vz = PID(50, 10, 10, 200., -200.,)
pid_vyaw = PID(50, 1, 1, 10., -10.,)

state_ref = get_mpc_parameter_4(x_fun, y_fun, z_fun, yaw_fun, vx_fun, vy_fun, vz_fun, theta_fun, dt, t0, 0)
state_all_ref = state_ref[:, 0].reshape(-1, 1)

### 开始仿真
mpciter = 0 # 迭代计数器
while(mpciter-sim_time / dt <0.0 ): 
    # 获得控制输入
    error_pos = state_ref[:4, 0] - x_init[:4, 0]

    v_ref = np.array([0.0, 0.0, 0.0, 0.0])
    v_ref[0] = pid_x_pos.update(error_pos[0])
    v_ref[1] = pid_y_pos.update(error_pos[1])
    v_ref[2] = pid_z_pos.update(error_pos[2])
    v_ref[3] = pid_yaw_pos.update(error_pos[3])

    v_body_ref = word2body(v_ref, x_init[3])
    error_v = v_body_ref - x_init[4:, 0]
    u_this = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])       # 仅将第一步控制作为真实控制输出
    add1 = pid_vx.update(error_v[0])
    add2 = pid_vy.update(error_v[1])
    add3 = pid_vyaw.update(error_v[3])
    add4 = pid_vz.update(error_v[2])

    u_this[0] = max(min(add1 + add2 + add3, uub), ulb)
    u_this[1] = max(min(add1 - add2 - add3, uub), ulb)
    u_this[2] = max(min(-add1 + add2 - add3, uub), ulb)
    u_this[3] = max(min(-add1 - add2 + add3, uub), ulb)
    u_this[4] = max(min(add4, uub), ulb)
    u_this[5] = max(min(add4, uub), ulb)

    # print(x_init)

    u = np.array([u_this])
    u_c.append(u)

    # 更新下一时刻状态
    t0 += dt
    t_c.append(t0)
    state_ref = get_mpc_parameter_4(x_fun, y_fun, z_fun, yaw_fun, vx_fun, vy_fun, vz_fun, theta_fun, dt, t0, 0)
    state_all_ref = np.concatenate((state_all_ref,state_ref[:, 0].reshape(-1,1)),axis=1)

    state_next_real = ode_real_rk4(x_init, u_this, t0)
    x_real.append(state_next_real)

    x_init = state_next_real

    # 计数器+1
    mpciter = mpciter + 1

x_real_np = np.array(x_real)[:,:,0]
a =np.array([x_init_list])
x_real_np = np.concatenate((a,x_real_np),axis=0)  
# print(x_real_np)

# 计算误差及相关指标
error_x_axis = x_real_np[:, 0] - state_all_ref[0, :].T
error_y_axis = x_real_np[:, 1] - state_all_ref[1, :].T
error_z_axis = x_real_np[:, 2] - state_all_ref[2, :].T
error_distance = error_x_axis ** 2 + error_y_axis ** 2 + error_z_axis ** 2
error_distance_max = math.sqrt(np.max(error_distance))
mse = np.sum(error_distance) / len(error_x_axis)
rmse = math.sqrt(mse)
print(rmse, error_distance_max)

ax = plt.axes(projection='3d')
# ax.set_zlim(-1.0, 1.0)
ax.plot3D(x_real_np[:, 0], x_real_np[:, 1], x_real_np[:, 2])
ax.scatter3D(state_all_ref[0, :], state_all_ref[1, :], state_all_ref[2, :],s= 2, c='r')
plt.show()