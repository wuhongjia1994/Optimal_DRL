import cvxpy as cp
import numpy as np
from cvxpy import Problem, Maximize
# from game_optimal import Get_args_setting
import random


class MU_response(object):
    afa = 0.04  # 数据量和周期的转换系数
    eta = 0.01  # 系统系数
    xi = 0.9  # 系统系数
    T = 100  # 整个FL的时间限制，一般在半个小时或者一个小时；here 60分钟
    T_req = 100

    def __init__(self, args_setting):
        self.user_num = args_setting['user_num']
        self.MC_num = args_setting['MC_num']
        self.T = 100  # 整个FL的时间限制，一般在半个小时或者一个小时；here 60分钟
        self.r = np.zeros(self.MC_num)
        self.s = np.zeros((self.MC_num, self.user_num), 'float32')  # 即V
        self.w = args_setting['w']  # 预测贡献度
        self.b = args_setting['b']  # 上传参数大小
        self.channel = args_setting['channel']  # 通信
        self.tau = args_setting['tau']  # MC的deadline要求
        self.theta = args_setting['theta']  # 用户的local准确度
        self.x = args_setting['x']  # 用户的单位时间采样数据量
        self.D = args_setting['D']  # MC的沉浸感要求底线
        self.B = np.zeros((self.user_num, self.MC_num), 'float32')  # 用户决策的带宽分配
        self.f = np.zeros((self.user_num, self.MC_num), 'float32')  # 用户决策计算资源分配
        self.c_f = args_setting['c_f']  # 用户的计算成本系数
        self.c_B = args_setting['c_B']  # 用户的带宽成本系数
        self.I = np.zeros((self.user_num, self.MC_num), 'float32')  # 预测系数
        self.f_max = args_setting['f_max']  # 用户的最大计算资源
        self.B_max = args_setting['B_max']  # 用户的最大带宽资源
        self.S = args_setting['S']  # 用于享受元宇宙服务以及其他服务需要的cycles
        self.u = args_setting['u']  # MC的效用函数中的利益系数

    def caluation(self, price):
        value = [[0] * self.user_num for _ in range(self.MC_num)]
        eva_reward = [[0] * self.MC_num for _ in range(self.user_num)]
        f = [[0] * self.MC_num for _ in range(self.user_num)]
        B = [[0] * self.MC_num for _ in range(self.user_num)]
        thet = np.zeros(self.user_num)
        reward = np.zeros(self.user_num)
        for m in range(self.user_num):
            thet[m] = np.log(1 / self.theta[m])
            g = np.zeros(self.MC_num)
            for n in range(self.MC_num):
                g[n] = np.log(1 + self.eta * (self.T // self.tau[n]) * self.x[m] * self.tau[n])
                self.I[m][n] = (self.w[m][n] * self.xi * g[n] / self.theta[m])
            # begin CVX

            f1, f2, B1, B2 = cp.Variable(), cp.Variable(), cp.Variable(), cp.Variable()
            # 定义⽬标函数

            A1 = price[0][m] * self.I[m][0] * (
                        self.D[0] - thet[m] * self.afa * self.x[m] * cp.inv_pos(f1) - self.b[m][0] * cp.inv_pos(
                    B1) * cp.inv_pos(self.channel[m][0])) - thet[m] * f1 * self.c_f[m] - B1 * self.c_B[m]
            A2 = price[1][m] * self.I[m][1] * (
                        self.D[1] - thet[m] * self.afa * self.x[m] * cp.inv_pos(f2) - self.b[m][1] * cp.inv_pos(
                    B2) * cp.inv_pos(self.channel[m][1])) - thet[m] * f2 * self.c_f[m] - B2 * self.c_B[m]
            A = A1 + A2
            obj = cp.Maximize(A)
            # 定义约束条件
            constraints = [f1 >= 0.1, B1 >= 0.1, f1 + f2 <= self.f_max[m],
                           thet[m] * self.afa * self.x[m] * self.tau[0] * cp.inv_pos(f1) + self.b[m][0] * cp.inv_pos(
                               B1) * cp.inv_pos(self.channel[m][0]) <= self.tau[0],
                           f2 >= 0.1, B2 >= 0.1, B1 + B2 <= self.B_max[m],
                           thet[m] * self.afa * self.x[m] * self.tau[1] * cp.inv_pos(f2) + self.b[m][1] * cp.inv_pos(
                               B2) * cp.inv_pos(self.channel[m][1]) <= self.tau[1]]
            # constraints = [f1 >= 0.1, B1 >= 0.1, f1 <= self.f_max[m], B1 <= self.B_max[m],
            #                    cp.log(cp.inv_pos(self.theta[m]))*self.afa*self.x[m]*self.tau[0]*cp.inv_pos(f1)+self.b[m][0]*cp.inv_pos(B1)*cp.inv_pos(self.channel[m][0]) <= self.tau[0],
            #                    f2 >= 0.1, B2 >= 0.1, f2 <= self.f_max[m], B2 <= self.B_max[m],
            #                    cp.log(cp.inv_pos(self.theta[m]))*self.afa * self.x[m] * self.tau[1] * cp.inv_pos(f2) + self.b[m][1] * cp.inv_pos(B2) * cp.inv_pos(self.channel[m][1]) <= self.tau[1]]
            #     # 求解

            prob = cp.Problem(obj, constraints)
            prob.solve()
            # print('status: ', prob.status)
            reward[m] = prob.value
            value[0][m] = self.I[m][0] * (
                        self.D[0] - thet[m] * self.afa * self.x[m] * cp.inv_pos(f1.value) - self.b[m][0] * cp.inv_pos(
                    B1.value) * cp.inv_pos(self.channel[m][0]))
            value[1][m] = self.I[m][1] * (
                        self.D[1] - thet[m] * self.afa * self.x[m] * cp.inv_pos(f2.value) - self.b[m][1] * cp.inv_pos(
                    B2.value) * cp.inv_pos(self.channel[m][1]))
            # print('reward', reward[m])
            # print(f1.value, f2.value, B1.value, B2.value)
            if (reward[m] < 0):
                f[m] = []
                B[m] = []
                value[0][m] = 0
                value[1][m] = 0
                reward[m] = 0
            else:
                # print(f1.value)
                # print(B1.value)
                # A = [f1.value,B1.value, f2.value,B2.value]
                f[m] = [f1.value, f2.value]
                B[m] = [B1.value, B2.value]
                value[0][m] = self.I[m][0] * (
                            self.D[0] - thet[m] * self.afa * self.x[m] * self.tau[0] / f[m][0] - self.b[m][0] / B[m][
                        0] / self.channel[m][0])
                value[1][m] = self.I[m][1] * (
                            self.D[1] - thet[m] * self.afa * self.x[m] * self.tau[1] / f[m][1] - self.b[m][1] / B[m][
                        1] / self.channel[m][1])

        # print(value)
        return reward, f, B, value


if __name__ == "__main__":
    args_setting = Get_args_setting()
    price = [[15, 7], [6, 6]]
    reward, f, B, value = MU_response(args_setting).caluation(price)
    print('reward:', reward)
    print('f:', f)
    print('B:', B)
