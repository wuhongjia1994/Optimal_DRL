import joblib
import numpy as np
import tensorflow as tf
from agents.ppo_agent import PpoAgent
from common.valuenorm import ValueNorm, FakeNorm
from types import SimpleNamespace as SN


class MappoModel:
    def __init__(self, env, args):
        self.env = env
        self.nenvs = env.num_envs
        self.nagent = env.nagent
        self.huber_delta = float(args.huber_delta)
        self.ent_coef = args.ent_coef
        self.vf_coef = args.vf_coef
        self.max_grad_norm = args.max_grad_norm
        self.use_rnn = args.use_rnn
        self.value_norm = args.value_norm
        self.use_popart = args.use_popart
        self.use_ippo = getattr(args, 'ippo', False)

        self.act_model = PpoAgent(env, args, 1, reuse=False)
        self.step = self.act_model.step
        nbatch = args.runner_steps // args.nminibatch
        self.train_model = PpoAgent(env, args, nbatch, reuse=True)

        if self.value_norm:
            self.value_normalizer = ValueNorm(popart=self.use_popart)
        else:
            self.value_normalizer = FakeNorm()

        self._init_placeholders()
        self._init_pg_loss()
        self._init_value_loss()
        self._init_train_op()

        self.sess = tf.get_default_session()
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def _init_placeholders(self):

        self.Adv =tf.compat.v1.placeholder(tf.float32, shape=self.train_model.values.shape, name='advantage')

        self.Raw_Act = tf.compat.v1.placeholder(tf.float32, shape=self.train_model.acts.shape, name='action')

        self.Return = tf.compat.v1.placeholder(tf.float32, shape=self.train_model.values.shape, name='return')

        self.OldNegLogPac = tf.compat.v1.placeholder(tf.float32, shape=self.train_model.neglogpas.shape, name='old_neg_logpac')

        self.OldVpred = tf.compat.v1.placeholder(tf.float32, self.train_model.values.shape, name='value')

        self.LR = tf.compat.v1.placeholder(tf.float32, [], name='learning_rate')

        self.ClipRange = tf.compat.v1.placeholder(tf.float32, [], name='clip_range')


    def _init_value_loss(self):

        # V-function loss
        vpred = self.train_model.values
        vpredclipped = self.OldVpred + tf.clip_by_value(
            self.train_model.values - self.OldVpred, -self.ClipRange, self.ClipRange)
        vf_losses1 = tf.compat.v1.losses.huber_loss(self.Return, vpred, delta=self.huber_delta)
        vf_losses2 = tf.compat.v1.losses.huber_loss(self.Return, vpredclipped, self.huber_delta)
        vf_loss = tf.maximum(vf_losses1, vf_losses2)

        self.vf_loss = 0.5 * tf.reduce_mean(vf_loss)

    def _init_pg_loss(self):

        # # Policy gradients
        neglogpac = self.train_model.neglogp(self.Raw_Act)
        ratio = tf.exp(self.OldNegLogPac - neglogpac)  # policy ratio

        pg_losses1 = -self.Adv * ratio
        pg_losses2 = -self.Adv * tf.clip_by_value(ratio, 1.0 - self.ClipRange, 1.0 + self.ClipRange)

        self.pg_loss = tf.reduce_mean(tf.maximum(pg_losses1, pg_losses2))
        self.entropy = tf.reduce_mean(self.train_model.pd.entropy())

    def _init_train_op(self):
        # total loss
        loss = self.pg_loss - self.ent_coef * self.entropy + self.vf_coef * self.vf_loss
        params = tf.compat.v1.trainable_variables()
        grads = tf.gradients(loss, params)
        if self.max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)  # clip gradients
        grads = list(zip(grads, params))
        trainer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.LR, epsilon=1e-5)
        self.train_op = trainer.apply_gradients(grads)

    def train(self, lr, cliprange, data):
        data = SN(**data)
        returns = data.returns
        # update value normalizer
        self.value_normalizer.update(returns)
        returns_normlized = self.value_normalizer.normalize(returns)

        advs = data.returns - self.value_normalizer.denormalize(data.values)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        feed_dict = {self.ClipRange: cliprange,
                     self.LR: lr,
                     self.Raw_Act: data.raw_actions,
                     self.Adv: advs,
                     self.Return: returns_normlized,
                     self.OldVpred: data.values,
                     self.OldNegLogPac: data.neglogpacs,
                     self.train_model.obs: data.obs,
                     self.train_model.cent_obs: data.cent_obs}
        if self.use_ippo:
            feed_dict[self.train_model.cent_obs] = data.obs
        if self.use_rnn:
            feed_dict[self.train_model.rnn_states_a] = data.state_a
            feed_dict[self.train_model.rnn_states_c] = data.state_c
            feed_dict[self.train_model.masks] = data.dones

        return self.sess.run([self.pg_loss, self.vf_loss, self.entropy, self.train_op], feed_dict)[:-1]

    def save(self, save_path):
        params = tf.compat.v1.trainable_variables()
        ps = tf.get_default_session().run(params)
        joblib.dump((ps, self.env, self.value_normalizer), save_path)

    def load(self, load_path):
        if load_path is not None:
            loaded_params, env, value_normalizer = joblib.load(load_path)
            self.env.restore(env)
            self.value_normalizer.restore(value_normalizer)
            params = tf.compat.v1.trainable_variables()
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            tf.get_default_session().run(restores)
