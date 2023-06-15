from my_gp_class import MYGP

import pyDOE
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

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

def ode_real_new(x, u):
    """ ROV model
    目前考虑了未知水动力，质量测量不准确，外界恒定干扰三个要素
    """
    dxdt = [
            (10 + u[0] - 34.96 * x[0] * ca.fabs(x[0]) - 0.0 * x[0] - 0.15 * x[1]) / (8.4 + 2.6) - x[2] * x[1],
            (20 + u[1] - 103.25 * x[1] * ca.fabs(x[1]) - 0.26 * x[1] - 0.15 * x[0]) / (8.4 + 18.5) + x[2] * x[0],
            (0.5 + u[2] - 0.084 * x[2] * ca.fabs(x[2]) - 0.895 * x[2]) / 0.26,
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

def ode_error_new(x, u):
    """ 2D ROV model
    """
    dxdt = [
            (10 + u[0] - 34.96 * x[0] * ca.fabs(x[0]) - 0.0 * x[0] - 0.15 * x[1]) / (8.4 + 2.6) - (u[0] - 30.0 * x[0] * ca.fabs(x[0]) - 0.0 * x[0]) / 8.4,
            (20 + u[1] - 103.25 * x[1] * ca.fabs(x[1]) - 0.26 * x[1] - 0.15 * x[0]) / (8.4 + 18.5) - (u[1] - 80.0 * x[1] * ca.fabs(x[1]) - 0.0 * x[1]) / (8.4 +10.0),
            (0.5 + u[2] - 0.084 * x[2] * ca.fabs(x[2]) - 0.895 * x[2]) / 0.26 - (u[2] - 0.05 * x[2] * ca.fabs(x[2]) - 1.0 * x[2]) / 0.2,
    ]
    return  dxdt

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
"""生成训练集与测试集，通过GP类进行验证
"""
noise_cov = np.eye(Nx) * 1e-3
X, Y = my_generate_training_data(ode_error, N, Nu, Nx, uub, ulb, xub, xlb, noise_cov, noise=True)

X_test, Y_test = my_generate_training_data(ode_error_new, N_test, Nu, Nx, uub, ulb, xub, xlb, noise_cov, noise=True)

X_test_new, Y_test_new = my_generate_training_data(ode_error_new, N_test, Nu, Nx, uub, ulb, xub, xlb, noise_cov, noise=True)

test_gp = MYGP(X_test, Y_test, X, Y, ode_nominal)

my_predict_fun_old = test_gp.get_correct_fun()
my_predict_fun_new = test_gp.get_correct_fun()

y_real = []
y_nominal = []
y_predict = []
for i in range(N_test):
    X_test_ca = ca.SX(X_test[i, :Nx].reshape(-1,1))
    U_test_ca = ca.SX(X_test[i, Nx:].reshape(-1,1))
    my_predict_fun_new = test_gp.correct_fun_update(X_test[i, :], Y_test[i, :])
    my_predict_ca = my_predict_fun_new(X_test[i, :Nx],X_test[i, Nx:])

    y_real_t = ode_real(X_test[i, :Nx],X_test[i, Nx:])
    y_real.append(y_real_t)

    y_nominal_t = ode_nominal(X_test[i, :Nx],X_test[i, Nx:])
    y_nominal.append(y_nominal_t)

    y_predict.append(my_predict_ca)

y_real = []
y_nominal = []
y_predict = []
for i in range(N_test):
    X_test_ca = ca.SX(X_test_new[i, :Nx].reshape(-1,1))
    U_test_ca = ca.SX(X_test_new[i, Nx:].reshape(-1,1))
    my_predict_ca = my_predict_fun_new(X_test_new[i, :Nx],X_test_new[i, Nx:])

    y_real_t = ode_real_new(X_test_new[i, :Nx],X_test_new[i, Nx:])
    y_real.append(y_real_t)

    y_nominal_t = ode_nominal(X_test_new[i, :Nx],X_test_new[i, Nx:])
    y_nominal.append(y_nominal_t)

    y_predict.append(my_predict_ca)

y_real_np = np.array(y_real)
y_nominal_np = np.array(y_nominal)
y_predict_np = np.array(y_predict)
x_x = np.array([-60,60])
for i in range(Nx):
    plt.figure(i)
    plt.scatter(y_predict_np[:, i], y_real_np[:, i], s= 10, c='r')
    plt.scatter(y_nominal_np[:, i], y_real_np[:, i], s= 10, c='g')
    plt.plot(x_x, x_x)
plt.show()