import tensorflow as tf

from common.distributions import make_pdtype
from .utils import fc, mlp, ln_gru

class Actor:
    def __init__(self, args, env, name='actor'):
        self.args = args
        self.env = env
        self.name = name
        self.pdtype = make_pdtype(env.action_space)
        self._hidden_size = args.hidden_size
        self._use_rnn = args.use_rnn
        self._rnn_chunk = args.rnn_chunk
        self._layer_n = args.layer_n
        self._feat_norm = args.feat_norm

    def forward(self, obs, rnn_states=None, masks=None):
        with tf.compat.v1.variable_scope(self.name, reuse=tf.compat.v1.AUTO_REUSE):  # actor network
            x = mlp(obs, self._hidden_size, self._layer_n, tf.nn.relu, self._feat_norm)
            if self._use_rnn:
                actor_features, rnn_states = ln_gru(x, rnn_states, masks, self._hidden_size, self._rnn_chunk)
            else:
                actor_features = fc(x, units=self._hidden_size, activation=tf.nn.relu)
            logits = fc(actor_features, units=self.pdtype.param_shape()[0], init_scale=0.01)

        pd = self.pdtype.pdfromflat(logits)
        acts = tf.stop_gradient(pd.sample())
  #      neglogpas = pd.neglogp(acts)
        return acts, rnn_states, pd


class Critic:
    def __init__(self, args, env, name='critic'):
        self.args = args
        self.env = env
        self.name = name
        self.pdtype = make_pdtype(env.action_space)
        self._hidden_size = args.hidden_size
        self._use_rnn = args.use_rnn
        self._rnn_chunk = args.rnn_chunk
        self._layer_n = args.layer_n
        self._feat_norm = args.feat_norm
        self._use_popart = args.use_popart

    def forward(self, cent_obs, rnn_states=None, masks=None):
        with tf.compat.v1.variable_scope(self.name, reuse=tf.compat.v1.AUTO_REUSE):
            x = mlp(cent_obs, self._hidden_size, self._layer_n, tf.nn.relu, self._feat_norm)
            if self._use_rnn:
                critic_features, rnn_states = ln_gru(x, rnn_states, masks, self._hidden_size, self._rnn_chunk)
            else:
                critic_features = fc(x, units=self._hidden_size, activation=tf.nn.relu)
            values = fc(critic_features, 1, name='values', init_scale=1)[..., 0]

            if self._use_popart and  len(tf.get_collection('popart_wandb')) == 0:
                scope = tf.get_variable_scope().name+ '/values'
                popart_wandb = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
                tf.add_to_collection('popart_wandb', popart_wandb)
                self.popart_init = True
        return values, rnn_states
