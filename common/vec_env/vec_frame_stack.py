from gym import spaces
from . import VecEnvWrapper
import numpy as np

# modified from openai baselines

class VecFrameStack(VecEnvWrapper):
    """
    Vectorized environment base class
    """
    def __init__(self, venv, nstack_ob=1, nstack_state=1):
        """
        :param venv:  env
        :param nstack_ob: the number of stacked frames for the observation
        :param nstack_state: the number of stacked frames for the state
        """
        self.venv = venv
        self.nstack_ob = nstack_ob
        self.nstack_state = nstack_state
        self.nagent = venv.nagent
        # obs
        wos = venv.observation_space # wrapped ob space
        low = np.repeat(wos.low, self.nstack_ob, axis=-1)
        high = np.repeat(wos.high, self.nstack_ob, axis=-1)
        self.stackedobs = np.zeros((venv.num_envs,self.nagent)+low.shape, low.dtype)
        observation_space = spaces.Box(low=low, high=high, dtype=venv.observation_space.dtype)

        #state
        slow = np.repeat(venv.state_space.low, self.nstack_state, axis=-1)
        shigh = np.repeat(venv.state_space.high, self.nstack_state, axis=-1)
        self.stackedstates = np.zeros((venv.num_envs,self.nagent) + slow.shape, slow.dtype)
        state_space = spaces.Box(low=slow, high=shigh, dtype=venv.state_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space,state_space=state_space)

    def step_wait(self):
        obs, rews, news, state, infos,obs_,state_ = self.venv.step_wait()
        self.stackedobs = np.roll(self.stackedobs, shift=-obs.shape[-1], axis=-1)
        self.stackedstates = np.roll(self.stackedstates,shift=-state.shape[-1],axis=-1)
        temp_state_ = self.stackedstates.copy()
        temp_state_[..., -state.shape[-1]:] = state_
        temp_ob_ = self.stackedobs.copy()
        temp_ob_ = temp_ob_[..., -obs.shape[-1]:] = obs_
        for (i, new) in enumerate(news):
            if np.all(new):
                self.stackedobs[i] = 0
                self.stackedstates[i] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        self.stackedstates[..., -state.shape[-1]:] = state
        return self.stackedobs, rews, news, self.stackedstates, infos, obs_,state_

    def reset(self):
        """
        Reset all environments
        """
        obs, state = self.venv.reset()
        self.stackedobs[...] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        self.stackedstates[...] = 0
        self.stackedstates[...,-state.shape[-1]:] = state
        return self.stackedobs, self.stackedstates

    def close(self):
        self.venv.close()

    def restore(self, env):
        # this is used for restore the env from pickle
        prop = env.__dict__
        for k, v in prop.items():
            if k == 'venv':
                self.venv.__setstate__(prop['venv'].__dict__)
            else:
                setattr(self, k, v)
        
        
        
