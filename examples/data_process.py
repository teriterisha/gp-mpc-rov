import pyDOE
import numpy as np
import matplotlib.pyplot as plt

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

def generate_data_during_sim(ode, x_int, u_int):
    """此段代码的功能是在仿真过程中产生训练数据，用于动态更新"""
    # x_int.reshape((1, -1))
    # u_int.reshape((1, -1))
    y = ode(x_int, u_int)
    return y

def my_compare_date(gp,X_test,ode_real,ode_nominal,Nx,Nu,N_test):
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

    plt.show()
    return y_real_np,y_nominal_np,y_predict_np