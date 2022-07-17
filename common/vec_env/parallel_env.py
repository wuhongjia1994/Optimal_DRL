import numpy as np
from multiprocessing import Process, Pipe
from . import VecEnv, CloudpickleWrapper


# refers to openai baselines, create parallel envs

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()

        if cmd == 'step':
            actions = data
            ob, reward, done, state,  info = env.step(actions)
            ob_ = ob[:]
            state_ = state[:]
            if np.all(done):
                ob, state = env.reset()
            remote.send((ob, reward, done, state, info, ob_, state_))

        elif cmd == 'reset':
            obs, state = env.reset()
            remote.send((obs, state))

        elif cmd == 'close':
            remote.close()
            break

        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space, env.state_space))

        elif cmd == 'get_agents_num':
            remote.send(env.n_agents)

        elif cmd == 'is_with_id':
            remote.send((env.ob_with_id, env.state_with_id))
        else:
            raise NotImplementedError


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns):

        self.waiting = False
        self.closed = False
        self.nenvs = len(env_fns)

        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.nenvs)])

        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()

        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space, state_space = self.remotes[0].recv()

        self.remotes[0].send(('get_agents_num', None))
        self.nagent = self.remotes[0].recv()

        self.remotes[0].send(('is_with_id', None))
        self.ob_with_id, self.state_with_id = self.remotes[0].recv()

        VecEnv.__init__(self, len(env_fns), observation_space, action_space, state_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        obs, rewards, dones, states, infos = [], [], [], [], []
        obs_, states_ = [],[]
        for remote in self.remotes:
            ob, reward, done, state, info, ob_,state_ = remote.recv()
            obs.append(ob)
            rewards.append(reward)
            dones.append(done)
            states.append(state)
            infos.append(info)
            obs_.append(ob_)
            states_.append(state_)
        self.waiting = False

        obs = np.array(obs)  # [nenv, nagent, ndim]
        rewards = np.asarray(rewards)
        dones = np.asarray(dones)
        states = np.array(states)  # [nenv, n_state_dim]
        states_ = np.asarray(states_)
        obs_ = np.asarray(obs_)
        return obs, rewards, dones, states, infos, obs_, states_

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs, states= [], []
        for remote in self.remotes:
            ob, state = remote.recv()
            obs.append(ob)
            states.append(state)
        obs = np.array(obs)  # [nenv, nagent, nob_dim]
        states = np.array(states)  # [nenv, nagent, n_state_dim]
        return obs, states

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True
