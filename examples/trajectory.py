import math
import numpy as np

class trace:
    def __init__(self, x_fun, y_fun) -> None:
        """输入x位置方程与y位置方程"""
        self.x_fun = x_fun
        self.y_fun = y_fun
        self.dt = 1e-3

    def get_v1(self):
        """机器人朝向不动，得到三轴速度"""
        def yaw_fun(t):
            return 0

        def theta_fun(t):
            return 0

        def vx_fun(t):
            dx = self.x_fun(t + self.dt) - self.x_fun(t)
            return dx/self.dt

        def vy_fun(t):
            dy = self.y_fun(t + self.dt) - self.y_fun(t)
            return dy/self.dt
        
        return vx_fun, vy_fun, theta_fun, yaw_fun

    def get_v2(self):
        """y轴速度为0,仅靠转向来实现"""
        def vy_fun(t):
            return 0

        def yaw_fun(t):
            dx = self.x_fun(t + self.dt) - self.x_fun(t)
            dy = self.y_fun(t + self.dt) - self.y_fun(t)
            yaw = math.atan(dy/dx)