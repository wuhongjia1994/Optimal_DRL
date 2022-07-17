import tensorflow as tf
import numpy as np
from common.utils import is_pickleable

class FakeNorm:
    # A fake normalizer, return what input.

    def update(self,x):
        pass
    def normalize(self, x,*args,**kwargs):
        return x
    def denormalize(self, x):
        return x
    def restore(self,*args,**kwargs):
        pass

class ValueNorm:

    def __init__(self,
                 input_shape=(),
                 per_element_update=False,
                 popart=False,
                 beta=0.99999,
                 epsilon=1e-5):
        # if popart is True, this is a popart normalizer

        self.input_shape = input_shape
        self.per_element_update = per_element_update
        self.popart = popart
        self.beta = beta
        self.epsilon = epsilon

        self.mean = np.zeros(input_shape)
        self.mean_sq = np.zeros(input_shape)
        self.debiasing_term = np.array(0)

        if self.popart:
            popart_wandb = tf.get_collection('popart_wandb')[0]
            self.w, self.b = popart_wandb[0], popart_wandb[1]
            self._init_update_wb_op()

    def _init_update_wb_op(self):
        self.old_mean_ph = tf.placeholder(dtype=tf.float32, shape=self.input_shape, name='old_mean')
        self.new_mean_ph = tf.placeholder(dtype=tf.float32, shape=self.input_shape, name='new_mean')
        self.old_stddev_ph = tf.placeholder(dtype=tf.float32, shape=self.input_shape, name='old_stddev')
        self.new_stddev_ph = tf.placeholder(dtype=tf.float32, shape=self.input_shape, name='new_stddev')

        update_w_op = tf.assign(self.w, self.w * self.old_stddev_ph / self.new_stddev_ph)
        update_b_op = tf.assign(self.b,
                                (self.old_stddev_ph * self.b + self.old_mean_ph - self.new_mean_ph) / self.new_stddev_ph)
        self.update_wb_op = [update_w_op, update_b_op]
        self.sess = tf.get_default_session()

    def update(self, input_vector):

        old_mean, old_var = self.debiased_mean_var()
        old_stddev = np.sqrt(old_var)

        input_vector = input_vector.reshape((-1,)+self.input_shape)
        batch_mean = input_vector.mean(0)
        batch_sq_mean = (input_vector ** 2).mean(0)

        if self.per_element_update:
            batch_size = input_vector.shape[0]
            weight = self.beta ** batch_size
        else:
            weight = self.beta

        self.mean = self.mean * weight + batch_mean * (1.0 - weight)
        self.mean_sq = self.mean_sq * weight + batch_sq_mean * (1.0 - weight)
        self.debiasing_term = weight * self.debiasing_term + (1.0 - weight)

        new_mean, new_var = self.debiased_mean_var()
        new_stddev = np.sqrt(new_var)
        if self.popart:
            feed_dict = {self.old_stddev_ph: old_stddev, self.new_stddev_ph: new_stddev,
                         self.new_mean_ph: new_mean, self.old_mean_ph: old_mean}

            self.sess.run(self.update_wb_op, feed_dict)

    def debiased_mean_var(self):
        debiased_mean = self.mean / self.debiasing_term.clip(min=self.epsilon)
        debiased_mean_sq = self.mean_sq / self.debiasing_term.clip(min=self.epsilon)
        debiased_var = (debiased_mean_sq - debiased_mean ** 2).clip(min=self.epsilon)
        return debiased_mean, debiased_var

    def normalize(self, x):
        x_shape = x.shape
        x = x.reshape((-1,)+self.input_shape)
        mean, var = self.debiased_mean_var()
        x_norm = (x - mean) / np.sqrt(var)
        x_norm = x_norm.reshape(x_shape)
        return x_norm

    def denormalize(self, x):
        x_shape = x.shape
        x = x.reshape((-1,)+self.input_shape)
        mean, var = self.debiased_mean_var()
        x_denorm = x * np.sqrt(var) + mean
        x_denorm = x_denorm.reshape(x_shape)
        return x_denorm

    def __getstate__(self):
        kwargs = {}
        for k, v in self.__dict__.items():
            if is_pickleable(v):
                kwargs[k] = v
        return kwargs

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)

    def restore(self, v):
        prop = v.__dict__
        for k, v in prop.items():
            setattr(self, k, v)

