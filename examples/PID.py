#this code refer to CSDN and do some minor change. 
import time

class PID:
    def __init__(self, P, I, D, up_limit, low_limit):
        self.Kp = P
        self.Ki = I
        self.Kd = D
        self.up_limit = up_limit
        self.low_limit = low_limit
        self.clear()
    def clear(self):
        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0
        self.int_error = 0.0
        self.output = 0.0
    def update(self, error):
        delta_error = error - self.last_error
        self.PTerm = self.Kp * error       #比例
        self.ITerm += self.Ki * error      #积分
        self.DTerm = self.Kd * delta_error #微分
        self.last_error = error
        self.output = self.PTerm + self.ITerm + self.DTerm
        if self.output > self.up_limit:
            self.output = self.up_limit
        if self.output < self.low_limit:
            self.output = self.low_limit
        return self.output