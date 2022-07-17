from . import VecEnvWrapper
from .running_mean_std import RunningMeanStd
import numpy as np
from common.utils import is_pickleable


class VecNormalize(VecEnvWrapper):
    def __init__(self, venv, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        VecEnvWrapper.__init__(self, venv)
        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.state_rms = RunningMeanStd(shape=self.state_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros((self.num_envs,self.nagent))
        self.gamma = gamma
        self.epsilon = epsilon
        self.nagent = self.venv.nagent

    def step_wait(self):

        obs, rews, dones, states, infos, obs_,states_ = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        states = self._statefilt(states)
        obs_ = self._obfilt(obs_, False)
        states_ = self._statefilt(states_, False)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        return obs, rews, dones, states, infos, obs_,states_

    def _obfilt(self, obs, update=True):
        if self.ob_rms and update:
            self.ob_rms.update(obs)
            if self.ob_with_id:
                obs[..., :-self.nagent] = np.clip((obs[..., :-self.nagent] - self.ob_rms.mean[..., :-self.nagent])
                                                  / np.sqrt(self.ob_rms.var[..., :-self.nagent] + self.epsilon),
                                                  -self.clipob, self.clipob)
            else:
                obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob,
                              self.clipob)
        return obs

    def _statefilt(self, states, update=True):
        if self.state_rms and update:
            self.state_rms.update(states)
            if self.state_with_id:
                states[..., :-self.nagent] = np.clip(
                    (states[..., :-self.nagent] - self.state_rms.mean[..., :-self.nagent]) /
                    np.sqrt(self.state_rms.var[..., :-self.nagent] + self.epsilon), -self.clipob, self.clipob)
            else:
                states = np.clip((states - self.state_rms.mean) / np.sqrt(self.state_rms.var + self.epsilon),
                                 -self.clipob, self.clipob)
        return states

    def reset(self):
        obs, states = self.venv.reset()
        return self._obfilt(obs), self._statefilt(states)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.venv, name)

    def __getstate__(self):
        kwargs = {}
        for k, v in self.__dict__.items():
            if is_pickleable(v):
                kwargs[k] = v
        return kwargs

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)
