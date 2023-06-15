import numpy as np
import casadi as ca
from scipy.optimize import minimize
import time

def my_covSEard(x_train, x, u, ell, sf2, N):
    """ 计算训练数据与预测输入的协方差矩阵
    """
    x_predict = ca.repmat(ca.vertcat(x, u).T,N,1)
    ell_rept = ca.repmat(ell,N,1)
    dist = ca.sum2((x_train - x_predict)**2 / ell_rept**2)
    return sf2 * ca.SX.exp(-.5 * dist)

def calc_cov_matrix(X, ell, sf2):
    """ Calculate covariance matrix K
        使用了optimize.py中的函数，计算输入训练数据的协方差矩阵
        Squared Exponential ARD covariance kernel

    # Arguments:
        X: Training data matrix with inputs of size (N x Nx).
        ell: Vector with length scales of size Nx.
        sf2: Signal variance (scalar)
    """
    dist = 0
    n, D = X.shape
    for i in range(D):
        x = X[:, i].reshape(n, 1)
        dist = (np.sum(x**2, 1).reshape(-1, 1) + np.sum(x**2, 1) -
                2 * np.dot(x, x.T)) / ell[i]**2 + dist
    return sf2 * np.exp(-.5 * dist)

def calc_NLL_numpy(hyper, X, Y):
    """ Objective function
    使用了optimize.py中的函数，计算超参数优化中的最大似然函数
    Calculate the negative log likelihood function.

    # Arguments:
        hyper: Array with hyperparameters [ell_1 .. ell_Nx sf sn], where Nx is the
                number of inputs to the GP.
        X: Training data matrix with inputs of size (N x Nx).
        Y: Training data matrix with outpyts of size (N x Ny), with Ny number of outputs.

    # Returns:
        NLL: The negative log likelihood function (scalar)
    """

    n, D = X.shape
    ell = hyper[:D]
    sf2 = hyper[D]**2
    lik = hyper[D + 1]**2
    #m   = hyper[D + 2]
    K = calc_cov_matrix(X, ell, sf2)
    K = K + lik * np.eye(n)
    K = (K + K.T) * 0.5   # Make sure matrix is symmentric
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        print("K is not positive definit, adding jitter!")
        K = K + np.eye(n) * 1e-8
        L = np.linalg.cholesky(K)

    logK = 2 * np.sum(np.log(np.abs(np.diag(L))))
    invLy = np.linalg.solve(L, Y)
    alpha = np.linalg.solve(L.T, invLy)
    NLL = 0.5 * np.dot(Y.T, alpha) + 0.5 * logK
    return NLL

class MYGP:
    def __init__(self, X_hyper_opt, Y_hyper_opt, x_train, y_train):
        """目前的想法是重写GP类，输入参数为两组训练参数，一组用于初始化时优化超参数，一组用于构建预测方程。超参数优化的数组可以相对大，预测方程用相对小
             
        """
        X_hyper_opt = np.array(X_hyper_opt).copy()
        Y_hyper_opt = np.array(Y_hyper_opt).copy()
        x_train = np.array(x_train).copy()
        y_train = np.array(y_train).copy()

        self.__X_hyper_opt = X_hyper_opt
        self.__Y_hyper_opt = Y_hyper_opt
        self.__Ny = Y_hyper_opt.shape[1]
        self.__Nx = X_hyper_opt.shape[1]
        self.__N = X_hyper_opt.shape[0]
        self.__Nu = self.__Nx - self.__Ny

        self.x_train = x_train
        self.y_train = y_train

        self.train_maxsize = 20
        self.train_originsize = x_train.shape[0]
        # 优化超参数
        self.optimize()
        # 计算构建预测函数所需的矩阵
        self.get_predict_parameter()

    def optimize(self):
        """优化超参数，使用训练超参数的数据（X_hyper_opt, Y_hyper_opt）"""
        # 单个Y输出的超参数个数
        h_ell   = self.__Nx     # Number of length scales parameters
        h_sf    = 1     # Standard deviation function
        h_sn    = 1     # Standard deviation noise
        num_hyp = h_ell + h_sf + h_sn
        options = {'disp': True, 'maxiter': 10000}
        self.hyp_opt = np.zeros((self.__Ny, num_hyp))
        print('\n________________________________________')
        print('# Optimizing hyperparameters (N=%d)' % self.__N )
        print('----------------------------------------')
        for output in range(self.__Ny):
            lb        = -np.inf * np.ones(num_hyp)
            ub        = np.inf * np.ones(num_hyp)
            lb[:self.__Nx]    = 1-2
            ub[:self.__Nx]    = 2e2
            lb[self.__Nx]     = 1e-8
            ub[self.__Nx]     = 1e2
            lb[self.__Nx + 1] = 10**-10
            ub[self.__Nx + 1] = 10**-2
            bounds = np.hstack((lb.reshape(num_hyp, 1), ub.reshape(num_hyp, 1)))
            hyp_init = np.zeros((num_hyp))
            hyp_init[:self.__Nx] = np.std(self.__X_hyper_opt, 0)
            hyp_init[self.__Nx] = np.std(self.__Y_hyper_opt[:, output])
            hyp_init[self.__Nx + 1] = 1e-5
            # 删去了multistart  这里不使用多次优化的方法来达到更优解
            obj = 0
            hyp_opt_loc = np.zeros((1, num_hyp))
            solve_time = -time.time()
            res = minimize(calc_NLL_numpy, hyp_init, args=(self.__X_hyper_opt, self.__Y_hyper_opt[:, output]),
                    method='SLSQP', options=options, bounds=bounds, tol=1e-12)
            obj = res.fun
            hyp_opt_loc[0, :] = res.x
            solve_time += time.time()
            print("* State %d:  %f s" % (output, solve_time))

            # With multistart, get solution with lowest decision function value
            self.hyp_opt[output, :]   = hyp_opt_loc[0, :]
            # ell = hyp_opt[output, :self.__Nx]
            # sf2 = hyp_opt[output, self.__Nx]**2
            # sn2 = hyp_opt[output, self.__Nx + 1]**2
        print('----------------------------------------')

    def get_predict_parameter(self):
        """计算所有与预测均值相关的矩阵，这里使用的是用于预测方程构建的数据（x_train, y_train)"""
        # Calculate the inverse covariance matrix
        N = self.x_train.shape[0]
        self.invK = np.zeros((self.__Ny, N, N))
        self.alpha = np.zeros((self.__Ny, N))
        self.chol = np.zeros((self.__Ny, N, N))
        for output in range(self.__Ny):
            ell = self.hyp_opt[output, :self.__Nx]
            sf2 = self.hyp_opt[output, self.__Nx]**2
            sn2 = self.hyp_opt[output, self.__Nx + 1]**2
            K = calc_cov_matrix(self.x_train, ell, sf2)
            K = K + sn2 * np.eye(N)
            K = (K + K.T) * 0.5   # Make sure matrix is symmentric
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                print("K matrix is not positive definit, adding jitter!")
                K = K + np.eye(N) * 1e-8
                L = np.linalg.cholesky(K)
            invL = np.linalg.solve(L, np.eye(N))
            self.invK[output, :, :] = np.linalg.solve(L.T, invL)
            self.chol[output] = L
            self.alpha[output] = np.linalg.solve(L.T, np.linalg.solve(L, self.y_train[:, output]))

    def get_mean_fun(self):
        """目前的想法是构建一个函数，通过输入标称模型与训练数据，得到一个gp修正过的模型表达式
        输入参数：ode_nomial(标称模型)，x_train(N * (Nx + Nu)), y_train(N * Nx)  
        已完成
        """
        N = self.x_train.shape[0]
        # print(N)
        X = ca.SX.sym('x',self.__Ny)
        U = ca.SX.sym('u',self.__Nu)
        Y = ca.SX.sym('y',self.__Ny)
        ell_s = ca.SX.sym('ell', 1, self.__Nx)
        sf2_s = ca.SX.sym('sf2')
        x_train_ca = ca.SX(self.x_train)

        covSE = ca.Function('covSE', [X, U, ell_s, sf2_s],
                            [my_covSEard(x_train_ca, X, U, ell_s, sf2_s, N)])
        for i in range(self.__Ny):
            # alpha_np = self.invk[i,:,:] @ self.y_train[:,i]
            # alpha_ca = ca.SX(alpha_np)
            ell_ca = ca.SX(self.hyp_opt[i,:self.__Nx])
            sf2 = ca.SX(self.hyp_opt[i, self.__Nx] ** 2)
            Y[i] = ca.mtimes(covSE(X, U, ell_ca, sf2).T, self.alpha[i])
        y_predict_fun = ca.Function('y_predict',[X, U], [Y])

        # 建立标称模型
        # Y = ca.vertcat(*self.ode_nominal(X, U))
        # ode_nominal_ca = ca.Function("ode_nominal", [X, U], [ca.vertcat(*self.ode_nominal(X, U))])
        # # 联合构建修正后的模型
        # correct_fun_predict = ca.Function("correct_fun",[X,U],[ode_nominal_ca(X, U) + y_predict_fun(X, U)])

        return y_predict_fun

    def data_update(self, x_in, y_in):
        """输入一组更新数据，更新用于构建方程的数据"""
        N = self.x_train.shape[0]   

        x_in = np.array(x_in).reshape((1, -1))
        y_in = np.array(y_in).reshape((1, -1)) 

        self.x_train[:N - 1, :] = self.x_train[1:N, :]
        self.x_train[-1, :] = x_in

        self.y_train[:N - 1, :] = self.y_train[1:N, :]
        self.y_train[-1, :] = y_in
        
    def data_update_new(self, x_in, y_in):

        x_in = np.array(x_in).reshape((1, -1))
        y_in = np.array(y_in).reshape((1, -1)) 

        N = self.x_train.shape[0]
        if N < self.train_maxsize:
            self.x_train = np.r_[self.x_train,x_in]
            self.y_train = np.r_[self.y_train,y_in]
        else:
            self.x_train[self.train_originsize:N - 1, :] = self.x_train[self.train_originsize + 1:N, :]
            self.x_train[-1, :] = x_in

            self.y_train[self.train_originsize:N - 1, :] = self.y_train[self.train_originsize + 1:N, :]
            self.y_train[-1, :] = y_in
        # print(self.x_train)  
        # print("----------",N,"-------------------------")  
        
    def mean_fun_update(self):
        self.get_predict_parameter()
        mean_fun_predict = self.get_mean_fun()
        return mean_fun_predict
    
    def get_all_train_data(self):
        return self.x_train, self.y_train