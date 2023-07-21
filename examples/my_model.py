import casadi as ca
import math
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

def ode_nominal(x, u):
    """ROV model
    此处定义x[0]为x轴向速度  x[1]为y轴轴向速度   x[2]为yaw角角速度
          u[0]为x轴推力  u[1]为y轴推力  u[2]为z轴转动力矩
    """
    # dxdt = [
    #         (u[0] - 30.0 * x[0] * ca.fabs(x[0]) - 0.0 * x[0]) / 8.4 - x[2] * x[1],
    #         (u[1] - 80.0 * x[1] * ca.fabs(x[1]) - 0.0 * x[1]) / (8.4 +10.0) + x[2] * x[0],
    #         (u[2] - 0.05 * x[2] * ca.fabs(x[2]) - 1.0 * x[2]) / 0.2,
    # ]
    dxdt = [
            # (u[0] - 0.0 * x[0] * ca.fabs(x[0]) - 0.0 * x[0]) / (8.4 + 0.0) - x[2] * x[1],
            # (u[1] - 0.0 * x[1] * ca.fabs(x[1]) - 0.0 * x[1]) / (8.4 + 0.0) + x[2] * x[0],
            # (u[2] - 0.0 * x[2] * ca.fabs(x[2]) - 0.0 * x[2]) / 0.2,
            u[0] / m + x[2] * x[1],
            u[1] / m - x[2] * x[0],
            u[2] / Iz,
    ]
    return  dxdt

def ode_real(x, u):
    """ ROV model
    目前考虑了未知水动力，质量测量不准确，外界恒定干扰三个要素
    """
    dxdt = [
            # (u[0] - 34.96 * x[0] * ca.fabs(x[0]) - 0.0 * x[0] - 3.5 * x[1]) / (8.4 + 2.6) - x[2] * x[1],
            # (u[1] - 103.25 * x[1] * ca.fabs(x[1]) - 0.26 * x[1] - 3.5 * x[0]) / (8.4 + 18.5) + x[2] * x[0],
            # (u[2] - 0.084 * x[2] * ca.fabs(x[2]) - 0.895 * x[2]) / 0.26,
            (u[0] + x[2] * x[1] * (m + Yv_dot) - Xu * x[0] - Xuu * ca.fabs(x[0]) * x[0]) / (m + Xu_dot),
            (u[1] - x[2] * x[0] * (m + Xu_dot) - Yv * x[1] - Yvv * ca.fabs(x[1]) * x[1]) / (m + Yv_dot),
            (u[2] + (Xu_dot - Yv_dot) * x[0] * x[1] - Nr * x[2] - Nrr * ca.fabs(x[2]) * x[2]) / (Iz + Nr_dot),
    ]
    return  dxdt

def ode_error(x, u):
    """ 2D ROV model
    """
    dxdt = [
            # (u[0] - 34.96 * x[0] * ca.fabs(x[0]) - 0.0 * x[0] - 3.5* x[1]) / (8.4 + 2.6) - (u[0] - 0.0 * x[0] * ca.fabs(x[0]) - 0.0 * x[0]) / 8.4,
            # (u[1] - 103.25 * x[1] * ca.fabs(x[1]) - 0.26 * x[1] - 3.5 * x[0]) / (8.4 + 18.5) - (u[1] - 0.0 * x[1] * ca.fabs(x[1]) - 0.0 * x[1]) / (8.4 + 0.0),
            # (u[2] - 0.084 * x[2] * ca.fabs(x[2]) - 0.895 * x[2]) / 0.26 - (u[2] - 0.00 * x[2] * ca.fabs(x[2]) - 0.0 * x[2]) / 0.2,
            (u[0] + x[2] * x[1] * (m + Yv_dot) - Xu * x[0] - Xuu * ca.fabs(x[0]) * x[0]) / (m + Xu_dot) - (u[0] / m + x[2] * x[1]),
            (u[1] - x[2] * x[0] * (m + Xu_dot) - Yv * x[1] - Yvv * ca.fabs(x[1]) * x[1]) / (m + Yv_dot) - (u[1] / m - x[2] * x[0]),
            (u[2] + (Xu_dot - Yv_dot) * x[0] * x[1] - Nr * x[2] - Nrr * ca.fabs(x[2]) * x[2]) / (Iz + Nr_dot) - u[2] / Iz,
    ]
    return  dxdt

def ode_real_distrub(x, u, x_dis, y_dis, t):
    """建立带干扰的运动学模型，由于干扰是定义在世界坐标系下的，所以需要引入偏航角修正方向x[0]为yaw角，其余依次为body坐标系下的x速度，y速度，偏航角速度"""
    dxdt = [
            #  x[3],
            # (u[0] - 34.96 * x[1] * ca.fabs(x[1]) - 0.0 * x[1] - 3.5 * x[2] + ca.cos(x[0]) * x_dis(t) + ca.sin(x[0]) * y_dis(t)) / (8.4 + 2.6) - x[3] * x[2],
            # (u[1] - 103.25 * x[2] * ca.fabs(x[2]) - 0.26 * x[2] - 3.5 * x[1] - ca.sin(x[0]) * x_dis(t) + ca.cos(x[0]) * y_dis(t)) / (8.4 + 18.5) + x[3] * x[1],
            # (u[2] - 0.084 * x[3] * ca.fabs(x[3]) - 0.895 * x[3]) / 0.26,
            x[3],
            (u[0] + x[3] * x[2] * (m + Yv_dot) - Xu * x[1] - Xuu * ca.fabs(x[1]) * x[1] + ca.cos(x[0]) * x_dis(t) + ca.sin(x[0]) * y_dis(t)) / (m + Xu_dot),
            (u[1] - x[3] * x[1] * (m + Xu_dot) - Yv * x[2] - Yvv * ca.fabs(x[2]) * x[2] - ca.sin(x[0]) * x_dis(t) + ca.cos(x[0]) * y_dis(t)) / (m + Yv_dot),
            (u[2] + (Xu_dot - Yv_dot) * x[1] * x[2] - Nr * x[3] - Nrr * ca.fabs(x[3]) * x[3]) / (Iz + Nr_dot),
    ]
    return dxdt

def ode_error_distrub(x, u, x_dis, y_dis, t):
    dxdt = [
            # (u[0] - 34.96 * x[1] * ca.fabs(x[1]) - 0.0 * x[1] - 3.5 * x[2] + ca.cos(x[0]) * x_dis(t) + ca.sin(x[0]) * y_dis(t)) / (8.4 + 2.6) - (u[0] - 0.0 * x[1] * ca.fabs(x[1]) - 0.0 * x[1]) / 8.4,
            # (u[1] - 103.25 * x[2] * ca.fabs(x[2]) - 0.26 * x[2] - 3.5 * x[1] - ca.sin(x[0]) * x_dis(t) + ca.cos(x[0]) * y_dis(t)) / (8.4 + 18.5)- (u[1] - 0.0 * x[2] * ca.fabs(x[2]) - 0.0 * x[2]) / (8.4 + 0.0),
            # (u[2] - 0.084 * x[3] * ca.fabs(x[3]) - 0.895 * x[3]) / 0.26 - (u[2] - 0.00 * x[3] * ca.fabs(x[3]) - 0.0 * x[3]) / 0.2,
            (u[0] + x[3] * x[2] * (m + Yv_dot) - Xu * x[1] - Xuu * ca.fabs(x[1]) * x[1] + ca.cos(x[0]) * x_dis(t) + ca.sin(x[0]) * y_dis(t)) / (m + Xu_dot) - (u[0] / m + x[3] * x[2]),
            (u[1] - x[3] * x[1] * (m + Xu_dot) - Yv * x[2] - Yvv * ca.fabs(x[2]) * x[2] - ca.sin(x[0]) * x_dis(t) + ca.cos(x[0]) * y_dis(t)) / (m + Yv_dot) - (u[1] / m - x[3] * x[1]),
            (u[2] + (Xu_dot - Yv_dot) * x[1] * x[2] - Nr * x[3] - Nrr * ca.fabs(x[3]) * x[3]) / (Iz + Nr_dot) - u[2] / Iz,
    ]
    return dxdt
################################################################     四自由度模型    ##########################
def ode_nominal_4(x, u):
    """四自由度的名义动力学模型"""
    dxdt = [
            u[0] / m + x[3] * x[1],
            u[1] / m - x[3] * x[0],
            u[2] / m,
            u[3] / Iz,
    ]
    return  dxdt

def ode_real_4(x, u):
    """无干扰的四自由度真实动力学模型"""
    dxdt = [
            (u[0] + x[3] * x[1] * (m + Yv_dot) - Xu * x[0] - Xuu * ca.fabs(x[0]) * x[0]) / (m + Xu_dot),
            (u[1] - x[3] * x[0] * (m + Xu_dot) - Yv * x[1] - Yvv * ca.fabs(x[1]) * x[1]) / (m + Yv_dot),
            (u[2] - Zw * x[2] - Zww * ca.fabs(x[2]) * x[2]) / (m + Zw_dot),
            (u[3] + (Xu_dot - Yv_dot) * x[0] * x[1] - Nr * x[3] - Nrr * ca.fabs(x[3]) * x[3]) / (Iz + Nr_dot),
    ]
    return  dxdt

def ode_error_4(x, u):
    """无干扰的四自由度误差公式"""
    dxdt = [
            (u[0] + x[3] * x[1] * (m + Yv_dot) - Xu * x[0] - Xuu * ca.fabs(x[0]) * x[0]) / (m + Xu_dot) - (u[0] / m + x[3] * x[1]),
            (u[1] - x[3] * x[0] * (m + Xu_dot) - Yv * x[1] - Yvv * ca.fabs(x[1]) * x[1]) / (m + Yv_dot) - (u[1] / m - x[3] * x[0]),
            (u[2] - Zw * x[2] - Zww * ca.fabs(x[2]) * x[2]) / (m + Zw_dot) - u[2] / m,
            (u[3] + (Xu_dot - Yv_dot) * x[0] * x[1] - Nr * x[3] - Nrr * ca.fabs(x[3]) * x[3]) / (Iz + Nr_dot) - u[3] / Iz,
    ]
    return dxdt

def ode_real_4_distrub(x, u, x_dis, y_dis, t):
    """建立带干扰的四自由度运动学模型，由于干扰是定义在世界坐标系下的，所以需要引入偏航角修正方向x[0]为yaw角，其余依次为body坐标系下的x速度，y速度，z轴速度，偏航角速度"""
    dxdt = [
             x[4],
            (u[0] + x[4] * x[2] * (m + Yv_dot) - Xu * x[1] - Xuu * ca.fabs(x[1]) * x[1] + ca.cos(x[0]) * x_dis(t) + ca.sin(x[0]) * y_dis(t)) / (m + Xu_dot),
            (u[1] - x[4] * x[1] * (m + Xu_dot) - Yv * x[2] - Yvv * ca.fabs(x[2]) * x[2] - ca.sin(x[0]) * x_dis(t) + ca.cos(x[0]) * y_dis(t)) / (m + Yv_dot),
            (u[2] - Zw * x[3] - Zww * ca.fabs(x[3]) * x[3]) / (m + Zw_dot),
            (u[3] + (Xu_dot - Yv_dot) * x[1] * x[2] - Nr * x[4] - Nrr * ca.fabs(x[4]) * x[4]) / (Iz + Nr_dot),
    ]
    return dxdt

def ode_error_4_distrub(x, u, x_dis, y_dis, t):
    """有干扰情况下的误差方程"""
    dxdt = [
            (u[0] + x[4] * x[2] * (m + Yv_dot) - Xu * x[1] - Xuu * ca.fabs(x[1]) * x[1] + ca.cos(x[0]) * x_dis(t) + ca.sin(x[0]) * y_dis(t)) / (m + Xu_dot) - (u[0] / m + x[4] * x[2]),
            (u[1] - x[4] * x[1] * (m + Xu_dot) - Yv * x[2] - Yvv * ca.fabs(x[2]) * x[2] - ca.sin(x[0]) * x_dis(t) + ca.cos(x[0]) * y_dis(t)) / (m + Yv_dot) - (u[1] / m - x[4] * x[1]),
            (u[2] - Zw * x[3] - Zww * ca.fabs(x[3]) * x[3]) / (m + Zw_dot) - u[2] / m,
            (u[3] + (Xu_dot - Yv_dot) * x[1] * x[2] - Nr * x[4] - Nrr * ca.fabs(x[4]) * x[4]) / (Iz + Nr_dot) - u[3] / Iz,
    ]
    return dxdt
################################################################     三自由度模型（细化到推进器）    ##########################
def ode_nominal_propeller(x, u):
    """ROV model
    此处定义x[0]为x轴向速度  x[1]为y轴轴向速度   x[2]为yaw角角速度
          u为水平面四个推进器
    """
    dxdt = [
            ((u[0] + u[1]) * p1 - (u[2] + u[3]) * p1) / m + x[2] * x[1],
            ((u[0] + u[2]) * p2 - (u[1] + u[3]) * p2) / m - x[2] * x[0],
            ((u[0] + u[3]) * l - (u[1] + u[2]) * l) / Iz,
    ]
    return  dxdt

def ode_real_propeller(x, u):
    """ ROV model
    目前考虑了未知水动力，质量测量不准确，外界恒定干扰三个要素
    """
    dxdt = [
            # (u[0] - 34.96 * x[0] * ca.fabs(x[0]) - 0.0 * x[0] - 3.5 * x[1]) / (8.4 + 2.6) - x[2] * x[1],
            # (u[1] - 103.25 * x[1] * ca.fabs(x[1]) - 0.26 * x[1] - 3.5 * x[0]) / (8.4 + 18.5) + x[2] * x[0],
            # (u[2] - 0.084 * x[2] * ca.fabs(x[2]) - 0.895 * x[2]) / 0.26,
            (((u[0] + u[1]) * p1 - (u[2] + u[3]) * p1) + x[2] * x[1] * (m + Yv_dot) - Xu * x[0] - Xuu * ca.fabs(x[0]) * x[0]) / (m + Xu_dot),
            (((u[0] + u[2]) * p2 - (u[1] + u[3]) * p2) - x[2] * x[0] * (m + Xu_dot) - Yv * x[1] - Yvv * ca.fabs(x[1]) * x[1]) / (m + Yv_dot),
            (((u[0] + u[3]) * l - (u[1] + u[2]) * l) + (Xu_dot - Yv_dot) * x[0] * x[1] - Nr * x[2] - Nrr * ca.fabs(x[2]) * x[2]) / (Iz + Nr_dot),
    ]
    return  dxdt

def ode_error_propeller(x, u):
    """ 2D ROV model
    """
    dxdt = [
            # (u[0] - 34.96 * x[0] * ca.fabs(x[0]) - 0.0 * x[0] - 3.5* x[1]) / (8.4 + 2.6) - (u[0] - 0.0 * x[0] * ca.fabs(x[0]) - 0.0 * x[0]) / 8.4,
            # (u[1] - 103.25 * x[1] * ca.fabs(x[1]) - 0.26 * x[1] - 3.5 * x[0]) / (8.4 + 18.5) - (u[1] - 0.0 * x[1] * ca.fabs(x[1]) - 0.0 * x[1]) / (8.4 + 0.0),
            # (u[2] - 0.084 * x[2] * ca.fabs(x[2]) - 0.895 * x[2]) / 0.26 - (u[2] - 0.00 * x[2] * ca.fabs(x[2]) - 0.0 * x[2]) / 0.2,
            (((u[0] + u[1]) * p1 - (u[2] + u[3]) * p1) + x[2] * x[1] * (m + Yv_dot) - Xu * x[0] - Xuu * ca.fabs(x[0]) * x[0]) / (m + Xu_dot) - (((u[0] + u[1]) * p1 - (u[2] + u[3]) * p1) / m + x[2] * x[1]),
            (((u[0] + u[2]) * p2 - (u[1] + u[3]) * p2) - x[2] * x[0] * (m + Xu_dot) - Yv * x[1] - Yvv * ca.fabs(x[1]) * x[1]) / (m + Yv_dot) - (((u[0] + u[2]) * p2 - (u[1] + u[3]) * p2) / m - x[2] * x[0]),
            (((u[0] + u[3]) * l - (u[1] + u[2]) * l) + (Xu_dot - Yv_dot) * x[0] * x[1] - Nr * x[2] - Nrr * ca.fabs(x[2]) * x[2]) / (Iz + Nr_dot) - ((u[0] + u[3]) * l - (u[1] + u[2]) * l) / Iz,
    ]
    return  dxdt

def ode_real_distrub_propeller(x, u, x_dis, y_dis, t):
    """建立带干扰的运动学模型，由于干扰是定义在世界坐标系下的，所以需要引入偏航角修正方向x[0]为yaw角，其余依次为body坐标系下的x速度，y速度，偏航角速度"""
    dxdt = [
            #  x[3],
            # (u[0] - 34.96 * x[1] * ca.fabs(x[1]) - 0.0 * x[1] - 3.5 * x[2] + ca.cos(x[0]) * x_dis(t) + ca.sin(x[0]) * y_dis(t)) / (8.4 + 2.6) - x[3] * x[2],
            # (u[1] - 103.25 * x[2] * ca.fabs(x[2]) - 0.26 * x[2] - 3.5 * x[1] - ca.sin(x[0]) * x_dis(t) + ca.cos(x[0]) * y_dis(t)) / (8.4 + 18.5) + x[3] * x[1],
            # (u[2] - 0.084 * x[3] * ca.fabs(x[3]) - 0.895 * x[3]) / 0.26,
            x[3],
            (((u[0] + u[1]) * p1 - (u[2] + u[3]) * p1) + x[3] * x[2] * (m + Yv_dot) - Xu * x[1] - Xuu * ca.fabs(x[1]) * x[1] + ca.cos(x[0]) * x_dis(t) + ca.sin(x[0]) * y_dis(t)) / (m + Xu_dot),
            (((u[0] + u[2]) * p2 - (u[1] + u[3]) * p2) - x[3] * x[1] * (m + Xu_dot) - Yv * x[2] - Yvv * ca.fabs(x[2]) * x[2] - ca.sin(x[0]) * x_dis(t) + ca.cos(x[0]) * y_dis(t)) / (m + Yv_dot),
            (((u[0] + u[3]) * l - (u[1] + u[2]) * l) + (Xu_dot - Yv_dot) * x[1] * x[2] - Nr * x[3] - Nrr * ca.fabs(x[3]) * x[3]) / (Iz + Nr_dot),
    ]
    return dxdt

def ode_error_distrub_propeller(x, u, x_dis, y_dis, t):
    dxdt = [
            # (u[0] - 34.96 * x[1] * ca.fabs(x[1]) - 0.0 * x[1] - 3.5 * x[2] + ca.cos(x[0]) * x_dis(t) + ca.sin(x[0]) * y_dis(t)) / (8.4 + 2.6) - (u[0] - 0.0 * x[1] * ca.fabs(x[1]) - 0.0 * x[1]) / 8.4,
            # (u[1] - 103.25 * x[2] * ca.fabs(x[2]) - 0.26 * x[2] - 3.5 * x[1] - ca.sin(x[0]) * x_dis(t) + ca.cos(x[0]) * y_dis(t)) / (8.4 + 18.5)- (u[1] - 0.0 * x[2] * ca.fabs(x[2]) - 0.0 * x[2]) / (8.4 + 0.0),
            # (u[2] - 0.084 * x[3] * ca.fabs(x[3]) - 0.895 * x[3]) / 0.26 - (u[2] - 0.00 * x[3] * ca.fabs(x[3]) - 0.0 * x[3]) / 0.2,
            (((u[0] + u[1]) * p1 - (u[2] + u[3]) * p1) + x[3] * x[2] * (m + Yv_dot) - Xu * x[1] - Xuu * ca.fabs(x[1]) * x[1] + ca.cos(x[0]) * x_dis(t) + ca.sin(x[0]) * y_dis(t)) / (m + Xu_dot) - (((u[0] + u[1]) * p1 - (u[2] + u[3]) * p1) / m + x[3] * x[2]),
            (((u[0] + u[2]) * p2 - (u[1] + u[3]) * p2) - x[3] * x[1] * (m + Xu_dot) - Yv * x[2] - Yvv * ca.fabs(x[2]) * x[2] - ca.sin(x[0]) * x_dis(t) + ca.cos(x[0]) * y_dis(t)) / (m + Yv_dot) - (((u[0] + u[2]) * p2 - (u[1] + u[3]) * p2) / m - x[3] * x[1]),
            (((u[0] + u[3]) * l - (u[1] + u[2]) * l) + (Xu_dot - Yv_dot) * x[1] * x[2] - Nr * x[3] - Nrr * ca.fabs(x[3]) * x[3]) / (Iz + Nr_dot) - ((u[0] + u[3]) * l - (u[1] + u[2]) * l) / Iz,
    ]
    return dxdt

################################################################     深度控制模型    ##########################
def z_nominal(x, u):
    """深度的名义动力学模型"""
    dxdt = [
            u[0] / m,
    ]
    return  dxdt

def z_real(x, u):
    """深度的真实动力学模型"""
    dxdt = [
            (u[0] - Zw * x[0] - Zww * ca.fabs(x[0]) * x[0]) / (m + Zw_dot),
    ]
    return  dxdt

def z_error(x, u):
    """深度的误差动力学模型"""
    dxdt = [
            (u[0] - Zw * x[0] - Zww * ca.fabs(x[0]) * x[0]) / (m + Zw_dot) - u[0] / m,
    ]
    return dxdt

def z_real_distrub(x, u, z_dis, t):
    """带干扰深度的误差动力学模型"""
    dxdt = [
            (u[0] - Zw * x[0] - Zww * ca.fabs(x[0]) * x[0] + z_dis(t)) / (m + Zw_dot),
    ]
    return dxdt

def z_error_distrub(x, u, z_dis, t):
    dxdt = [
            (u[0] - Zw * x[0] - Zww * ca.fabs(x[0]) * x[0] + z_dis(t)) / (m + Zw_dot) - u[0] / m,
    ]
    return dxdt