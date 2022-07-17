import numpy as np
from components.buffer import Buffer


class OnpolicyRunner:

    def __init__(self, env, model, args):
        self.env = env
        self.agent = model.act_model
        self.use_rnn = args.use_rnn
        self.value_normalizer = model.value_normalizer
        self.runner_steps = args.runner_steps
        self.gamma = args.gamma
        self.lam = args.lam
        self.buffer = Buffer(args.runner_steps)

        self.obs, self.cent_obs = self.env.reset()
        self.dones = np.zeros(shape=[self.env.num_envs, self.env.nagent])

        self.state_a = np.zeros(self.agent.rnn_states_a.shape) if self.use_rnn else None
        self.state_c = np.zeros(self.agent.rnn_states_c.shape) if self.use_rnn else None

    def run(self):
        epinfos = []
        self.buffer.clear()
        next_values = {}
        for t in range(self.runner_steps):
            data = {'state_a': self.state_a, 'state_c': self.state_c, 'dones': self.dones[:],
                    'dones_agent': self.dones[:]}
            actions,raw_actions, values, neglogpas, self.state_a, self.state_c = self.agent.step(self.obs,
                                                                                     self.cent_obs,
                                                                                     self.state_a,
                                                                                     self.state_c,
                                                                                     self.dones)

            data.update({'actions': actions,
                         'raw_actions': raw_actions,
                         'values': values,
                         'neglogpacs': neglogpas,
                         'obs': self.obs[:],
                         'cent_obs': self.cent_obs[:],
                         'dones': self.dones[:]})

            # transistion
            self.obs, rewards, self.dones, self.cent_obs, infos, obs_, states_ = self.env.step(actions)
            if np.all(self.dones):
                next_value = self.agent.step(obs_, states_, self.state_a,
                                             self.state_c,
                                             self.dones)[2]
                next_values[t+1] = next_value

            data.update({'rewards': rewards})
            self.buffer.store(data)

            for info in infos:
                maybeeinfo = info.get('episode')
                if maybeeinfo:
                    epinfos.append(maybeeinfo)

        last_dones = self.dones[:]
        last_values = self.agent.step(self.obs, self.cent_obs, self.state_a,
                                      self.state_c, self.dones)[2]

        batch = self.buffer.last_batch(self.runner_steps)
        batch['state_a'] = batch['state_a'].squeeze()
        batch['state_c'] = batch['state_c'].squeeze()
        returns = self.compute_returns(batch, last_dones, last_values, next_values, self.gamma, self.lam)
        batch['returns'] = returns
        return batch, epinfos

    def compute_returns(self, batch, last_dones, last_values,next_values, gamma=0.99, lam=0.95):
        values = np.append(batch['values'], last_values[None], axis=0)
        values = self.value_normalizer.denormalize(values)
        dones = np.append(batch['dones'], last_dones[None], axis=0)
        rewards = batch['rewards']
        advs = np.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(rewards.shape[0])):
            nextnonterminal = 1.0 - dones[t + 1]
            if t+1 in next_values.keys():
                nextvalue = next_values[t+1]
            else:
                nextvalue = values[t + 1]
            delta = rewards[t] + gamma * nextnonterminal * nextvalue - values[t]
            advs[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam

        return advs + values[:-1]
