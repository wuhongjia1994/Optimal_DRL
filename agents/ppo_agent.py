import numpy as np
import tensorflow as tf

from agents.actor_critic import Actor, Critic

class PpoAgent:
    def __init__(self, env, args, nsteps, reuse=False, name='ppo_agent'):
        self.args = args
        self.nsteps = nsteps
        self.nenv = env.num_envs
        self.nagent = env.nagent
        self.ob_dim = env.observation_space.shape[0]
        self.cent_ob_dim = env.state_space.shape[0]
        self.ac_space = env.action_space
        self._hidden_size = args.hidden_size
        self._use_rnn = args.use_rnn
        self._use_ippo = getattr(args, 'ippo', False)
        self._squash = getattr(args, 'squash', False)

        self._init_placeholders()
        self.actor = Actor(args, env)
        self.critic = Critic(args, env)

        self.name = name
        with tf.compat.v1.variable_scope(self.name, reuse):
            self.raw_acts, self.states_a_out, self.pd = self.actor.forward(self.obs, self.rnn_states_a,
                                                                                       self.masks)
            def neglogp(raw_acts):
                neglogpas = self.pd.neglogp(raw_acts)
                if self._squash:
                    neglogpas += self.squash_correction(raw_acts)
                return neglogpas
            self.neglogpas = neglogp(self.raw_acts)
            self.acts = tf.tanh(self.raw_acts) if self._squash else self.raw_acts
            self.values, self.states_c_out = self.critic.forward(self.cent_obs, self.rnn_states_c, self.masks)
        self.neglogp = neglogp
        self.sess = tf.get_default_session()

    def _init_placeholders(self):
        self.obs = tf.compat.v1.placeholder(tf.float32, [self.nsteps, self.nenv, self.nagent, self.ob_dim])
        self.cent_obs = tf.compat.v1.placeholder(tf.float32, [self.nsteps, self.nenv, self.nagent, self.cent_ob_dim])
        if self._use_ippo:
            self.cent_obs = self.obs
        self.rnn_states_a, self.rnn_states_c, self.masks = None, None, None
        if self._use_rnn:
            self.rnn_states_a = tf.compat.v1.placeholder(dtype=tf.float32,
                                               shape=[self.nsteps, self.nenv * self.nagent, self._hidden_size])
            self.rnn_states_c = tf.compat.v1.placeholder(dtype=tf.float32,
                                               shape=[self.nsteps, self.nenv * self.nagent, self._hidden_size])
            self.masks = tf.compat.v1.placeholder(tf.float32, [self.nsteps, self.nenv, self.nagent])

    def squash_correction(self, actions):
        if not self._squash: return 0
        return tf.reduce_sum(tf.log(1 - tf.tanh(actions) ** 2 + 1e-6), axis=-1)

    def step(self, obs, cent_obs, rnn_states_a=None, rnn_states_c=None, masks=None):
        obs = obs[np.newaxis]
        cent_obs = cent_obs[np.newaxis]
        if self._use_ippo:
            cent_obs = obs
        run_op = [self.acts,self.raw_acts, self.values, self.neglogpas]
        feed_dict = {self.obs: obs, self.cent_obs: cent_obs}

        if self._use_rnn:
            feed_dict[self.rnn_states_a] = rnn_states_a
            feed_dict[self.rnn_states_c] = rnn_states_c
            feed_dict[self.masks] = masks[np.newaxis]
            run_op.append(self.states_a_out)
            run_op.append(self.states_c_out)
            acts,raw_acts, values, neglogpas, rnn_states_a, rnn_states_c = self.sess.run(run_op, feed_dict)
        else:
            acts, raw_acts, values, neglogpas = self.sess.run(run_op, feed_dict)
        return acts[0],raw_acts[0], values[0], neglogpas[0], rnn_states_a, rnn_states_c

    @property
    def parameters(self):
        scope = tf.get_variable_scope().name
        scope += '/' + self.name + '/' if len(scope) else self.name + '/'
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
