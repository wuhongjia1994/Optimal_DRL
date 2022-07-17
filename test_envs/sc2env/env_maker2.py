from abc import ABC
import gym
import cvxpy as cp
import numpy as np
from cvxpy import Problem, Maximize
from gym.spaces import Box
import math
import os
import yaml
from types import SimpleNamespace as SN
#from cvx_MU import MU_response

def get_alg_config(name):
    with open('C:/Users\jill\Desktop/no_state_version\mappo_123456\mappo_123456\config\mappo.yaml', "r") as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)
    return SN(**config_dict)


class Toyenv(gym.Env, ABC):


    def __init__(self, args, ob_with_id=True, state_with_id=True, noise_as_id=True,
                 state_type='state_and_ob', ):
        args_setting = get_args_setting(args.env_seed)
        self.user_num = args_setting['user_num']
        self.MC_num = args_setting['MC_num']
        self.xi = args_setting['xi']
        self.afa = args_setting['afa']
        self.eta = args_setting['eta']
        self.T_req = args_setting['T_req']
        self.T = 400  # 整个FL的时间限制，一般在半个小时或者一个小时；here 60分钟
        # self.user_num = user_num  # 用户数量
        # self.MC_num = MC_num  # MC数量
        self.w = args_setting['w']  # 预测贡献度
        self.b = args_setting['b']  # 上传参数大小
        self.channel = args_setting['channel']  # 通信
        self.tau = args_setting['tau']  # MC的deadline要求
        self.theta = args_setting['theta']  # 用户的local准确度
        self.x = args_setting['x']  # 用户的单位时间采样数据量
        self.D = args_setting['D']  # MC的沉浸感要求底线
        self.c_f = args_setting['c_f']  # 用户的计算成本系数
        self.c_B = args_setting['c_B']  # 用户的带宽成本系数
        self.f_max = args_setting['f_max']  # 用户的最大计算资源
        self.B_max = args_setting['B_max']  # 用户的最大带宽资源
        self.S = args_setting['S']  # 用于享受元宇宙服务以及其他服务需要的cycles
        self.u = args_setting['u']  # MC的效用函数中的利益系数
        self.s = np.zeros((self.MC_num, self.user_num), 'float32')  # 即V

        self.ob_with_id = ob_with_id  # if True, observation will be [ob, agent_index]
        self.state_with_id = state_with_id  # if True, state is [state, agent_index]
        self.state_type = state_type

        self.nagent = self.n_agents = self.MC_num
        self.agent_index = np.eye(self.nagent)

        self.episode_timestep = None
        self.max_episode_length = 128

        # you can also use noise as the agent index by uncommenting the following two statements.
        if noise_as_id:
            np.random.seed(10)
            self.agent_index = np.random.normal(0, 1, [self.nagent, self.nagent])
            np.random.seed(None)

        self.ob_dim = self.user_num  # 随便设的
        self.state_dim = self.user_num * self.MC_num  # 随便设的
        self.action_dim = self.user_num  # continous action dim
        self._squash = getattr(args, 'squash', False)

        if self.ob_with_id:
            ob_dim = self.ob_dim + self.nagent
        else:
            ob_dim = self.ob_dim
        self.observation_space = Box(low=-np.ones(ob_dim), high=np.ones(ob_dim), dtype=np.float64)

        if self.state_type == 'state_and_ob':  # [state, self_ob]
            state_dim = self.state_dim + self.ob_dim
        elif self.state_type == 'state_and_allobs':  # [state, ob1,ob2,... ,obn]
            state_dim = self.state_dim + self.ob_dim * self.nagent
        else:  # else state is state
            state_dim = self.state_dim
        if self.state_with_id:  # concat state with agent_index
            state_dim += self.nagent

        self.state_space = Box(low=-np.ones(state_dim), high=np.ones(state_dim), dtype=np.float64)
        self.action_space = Box(low=-np.ones(self.action_dim), high=np.ones(self.action_dim), dtype=np.float64)

    def _init_params(self):
        self.B = np.zeros((self.user_num, self.MC_num), 'float32')  # 用户决策的带宽分配
        self.f = np.zeros((self.user_num, self.MC_num), 'float32')  # 用户决策计算资源分配
        self.I = np.zeros((self.user_num, self.MC_num), 'float32')  # 预测系数
        self.r = np.zeros(self.MC_num)
        self.s1 = np.zeros((self.user_num, self.MC_num), 'float32')  # 求用户的reward用的比较方便
        self.r_user = np.zeros((self.user_num, self.MC_num), 'float32')
        self.reward_user = np.zeros(self.user_num)
        self.r_MC = np.zeros(self.MC_num)
        self.s = np.zeros((self.MC_num, self.user_num), 'float32')  # 即V

    def reset(self):
        self.episode_timestep = 0
        self._init_params()
        obs = np.random.rand(self.nagent, self.ob_dim)
        state = np.random.rand(self.state_dim)
        obs, states = self.format_obs_state(obs, state)
        self.s_buf = []
        self.user_r_buf = []
        obs = np.ones_like(obs)
        states = np.ones_like(states)
        return obs, states

    def step(self, actions):
        if self._squash:
            actions = 0.3 * (actions + 1) / 2
        else:
            actions = 0.6 * (np.tanh(actions) + 1) / 2
       # actions = np.array([[0.028]])
        self._init_params()

        #-----------用cvx解用户最优----------------------------------------#

        f = [[0] * self.MC_num for _ in range(self.user_num)]
        B = [[0] * self.MC_num for _ in range(self.user_num)]
        thet = np.zeros(self.user_num)

        for m in range(self.user_num):
            thet[m] = np.log(1 / self.theta[m])
            g = np.zeros(self.MC_num)
            for n in range(self.MC_num):
                g[n] = np.log(1 + self.eta * (self.T // self.tau[n]) * self.x[m] * self.tau[n])
                self.I[m][n] = (self.w[m][n] * self.xi * g[n] / self.theta[m])
            # begin CVX

            f1, B1, f2, B2, f3, B3 = cp.Variable(), cp.Variable(), cp.Variable(), cp.Variable(), cp.Variable(), cp.Variable()
            # 定义⽬标函数

            A1 = actions[0][m] * self.I[m][0] * (self.D[0] - thet[m] * self.x[m] * self.tau[0] * cp.inv_pos(f1) - self.b[m][0] * cp.inv_pos(B1) * cp.inv_pos(self.channel[m][0])) - thet[m] * f1 * self.c_f[m] - B1 * self.c_B[m]
            A2 = actions[1][m] * self.I[m][1] * (self.D[1] - thet[m] * self.x[m] * self.tau[1] * cp.inv_pos(f2) - self.b[m][1] * cp.inv_pos(B2) * cp.inv_pos(self.channel[m][1])) - thet[m] * f2 * self.c_f[m] - B2 * self.c_B[m]
            A3 = actions[2][m] * self.I[m][2] * (self.D[2] - thet[m] * self.x[m] * self.tau[2] * cp.inv_pos(f3) - self.b[m][2] * cp.inv_pos(B3) * cp.inv_pos(self.channel[m][2])) - thet[m] * f3 * self.c_f[m] - B3 * self.c_B[m]
            A = A1 + A2 + A3
            obj = cp.Maximize(A)
            # 定义约束条件
            constraints = [f1 >= 0.001, B1 >= 0.001,
                           f1 + f2 + f3 <= ((self.T_req * self.f_max[m] - self.S[m]) / self.T_req),
                           thet[m] * self.x[m] * self.tau[0] * cp.inv_pos(f1) + self.b[m][0] * cp.inv_pos(B1) * cp.inv_pos(self.channel[m][0]) <= self.tau[0],
                           f2 >= 0.001, B2 >= 0.001, B1 + B2 + B3 <= self.B_max[m],
                           thet[m] * self.x[m] * self.tau[1] * cp.inv_pos(f2) + self.b[m][1] * cp.inv_pos(B2) * cp.inv_pos(self.channel[m][1]) <= self.tau[1],
                           f3 >= 0.001, B3 >= 0.001,
                           thet[m] * self.x[m] * self.tau[2] * cp.inv_pos(f3) + self.b[m][2] * cp.inv_pos(B3) * cp.inv_pos(self.channel[m][2]) <= self.tau[2]]


            prob = cp.Problem(obj, constraints)
            prob.solve()
            # print('status: ', prob.status)
            self.reward_user[m] = prob.value
            self.s[0][m] = self.I[m][0] * (self.D[0] - thet[m] * self.x[m] * self.tau[0] / f1.value - self.b[m][0] / B1.value /self.channel[m][0])
            self.s[1][m] = self.I[m][1] * (self.D[1] - thet[m] * self.x[m] * self.tau[1] / f2.value - self.b[m][1] / B2.value /self.channel[m][1])
            self.s[2][m] = self.I[m][2] * (self.D[2] - thet[m] * self.x[m] * self.tau[2] / f3.value - self.b[m][2] / B3.value /self.channel[m][2])
            # print('真实reward', reward)
            # print('真实value', value)
            f[m] = [f1.value, f2.value, f3.value]
            B[m] = [B1.value, B2.value, B3.value]
            # print('f', f)
            #
            # print('B', B)

        for n in range(self.MC_num):
            for m in range(self.user_num):
                if(self.reward_user[m]<0):
                    self.reward_user[m] = 0
                    self.r_MC[n] = 0
                    self.s[n][m] = 0
        for n in range(self.MC_num):
            Q = np.zeros(self.MC_num)
            for m in range(self.user_num):

                    Q[n] += self.s[n][m] * actions[n][m]
            if (sum(self.s[n])<=0):
                self.r_MC[n] = 0
            else:
                self.r_MC[n] = self.u[n] * np.log(1 + sum(self.s[n])) - Q[n]
        #print('user', self.reward_user)
        #------------用cvx解用户最优-------------------------#
        self.s_buf.append(self.s)
        self.user_r_buf.append(self.reward_user)
        # print('user', self.reward_user)
        # print(self.s)
        # ------------------用户的response-----------------------------#
        obs = self.s
        reward = self.r_MC
        # print('env里的reward',reward)
        state = np.array(self.s).flatten()
        obs, states = self.format_obs_state(obs, state)

        done = False
        info = {}
        self.episode_timestep += 1
        if self.episode_timestep >= self.max_episode_length:
            done = True  # fake done, the episode is not terminated, it is justs a reset indicator.
            info = {'actions': actions,
                    'VOI': np.mean(self.s_buf, 0),
                    'user_reward': np.mean(self.user_r_buf,0)
            }
        dones = np.array([done] * self.nagent)
        obs = np.ones_like(obs)
        states = np.ones_like(states)
        return obs, reward, dones, states, info

    def format_obs_state(self, obs, state):
        if self.state_type == 'state_and_ob':
            states = np.concatenate([np.repeat(state[None], self.nagent, axis=0), obs], axis=-1)
        elif self.state_type == 'state_and_allobs':
            states = np.concatenate([state, np.ravel(obs)], axis=-1)[None].repeac(self.nagent, axis=0)
        else:
            states = np.repeat(state[None], self.nagent, axis=0)
        if self.state_with_id:
            states = np.concatenate([states, self.agent_index], axis=-1)

        if self.ob_with_id:
            obs = np.concatenate([np.asarray(obs), self.agent_index], axis=-1)
        else:
            obs = np.asarray(obs)
        return obs, states


def get_args_setting(seed):
    import random
    random.seed(seed)
    #####################  hyper parameters  ####################
    user_num = 5
    MC_num = 3
    afa = 0.4  # 数据量和周期的转换系数
    eta = 1000  # 系统系数
    xi = 1  # 系统系数
    T = 400  # 整个FL的时间限制，一般在半个小时或者一个小时；s
    T_req = 3
    w = [[0.21, 0.5, 1.05], [0.6, 0.4, 1.18], [0.29, 0.7, 1.11], [0.4, 0.6, 1.3], [0.25, 0.6, 1.9]]
    #     最优的时候用 模型预测值，用户的每个任务具有不同的针对不同的MC;
    # w = [[0.21, 0.19, 0.35], [0.34, 0.21, 0.28], [0.23, 0.21, 0.19], [0.4, 0.29, 0.31], [0.25, 0.3, 0.29]]
    # 模型预测值，用户的每个任务具有不同的针对不同的MC;
    b = [[6.83, 7.2, 8], [6.83, 7.2, 8], [6.83, 7.2, 8], [6.83, 7.2, 8],
         [6.83, 7.2, 8]]  # MB
    channel = [[7, 6, 5], [6, 7, 5], [6, 7, 5], [6, 5, 7], [6, 6, 5]]
    tau = [10, 8, 5]  # 每个MC具有自己的单轮截止时间 1分钟，但是以s为单位
    theta = [0.2, 0.2, 0.25, 0.28, 0.2]
    # 每个用户具有自己的本地精度
    x = [0.015, 0.019, 0.012, 0.014, 0.015]  # 每个用户设备单位时间内的采样量  #GHz/s=GFLOTS
    c_f = [0.8, 0.7, 0.35, 0.2, 0.5]  # 每个用户的计算成本系数
    f_max = [3, 4, 3.8, 3.5, 4.5]  # 每个用户拥有的计算资源总量, 即3,4,3,5,3.5 Ghz
    c_B = [2, 3, 2.2, 1.2, 4]  # 每个用户的通信成本系数
    B_max = [3.2, 3, 4, 3.2, 3, 4]  # 每个用户拥有的带宽总量   MB
    S = [0.01, 0.07, 0.02, 0.05, 0.03]  # 每个用户享受服务时需要处理的Ghz
    D = [10, 8, 5]  # 每个MC的AR服务对沉浸感要求 s 待定调整参数
    u = [350, 400, 450]
    #####################  hyper parameters  #########################
    args_setting = {
        'user_num': user_num,
        'MC_num': MC_num,
        'afa': afa,
        'eta': eta,
        'xi': xi,
        'T': T,
        'T_req': T_req,
        'w': w,
        'b': b,
        'channel': channel,
        'tau': tau,
        'theta': theta,
        'x': x,
        'c_f': c_f,
        'f_max': f_max,
        'B_max': B_max,
        'c_B': c_B,
        'S': S,
        'D': D,
        'u': u,
    }
    return args_setting

if __name__ == "__main__":
    args = get_alg_config('mappo')
    env = Toyenv(args, ob_with_id=True, state_with_id=True, noise_as_id=True,
                 state_type='state_and_ob', )
    env.reset()
    actions = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
    obs, reward, dones, states, _ = env.step(actions)
    print('reward', reward)
