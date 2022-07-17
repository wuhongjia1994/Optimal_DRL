import tensorflow as tf
import numpy as np
from gym import error
import datetime as dt
import os
import multiprocessing
import random

def is_pickleable(x):
    import pickle
    try:
        pickle.dumps(x)
        return True
    except:
        return False

def get_token(unique=False):
    # get the time as the unique token
    if not unique:
        return ''
    t = dt.datetime.now()
    unique_token = '_{}_{}_{}_{}_{}_{}'.format(t.year, t.month, t.day, t.hour, t.minute, t.second)
    return unique_token

def decay(x: float, y: float, mul=False):
    if not mul:
        y = 1
    return x*y

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


def set_global_seed(seed):
    if seed is not None and not (isinstance(seed, int) and 0 <= seed):
        raise error.Error('Seed must be a non-negative integer or omitted, not {}'.format(seed))
    np.random.seed(seed)
    random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    return [seed]


def constfn(val):
    def f(_):
        return val

    return f


def huber_loss(e, d):
    a = tf.cast(tf.abs(e) <= d, tf.float32)
    b = tf.cast(e > d, tf.float32)
    return a * e * e / 2 + b * d * (tf.abs(e) - d / 2)

def make_session(num_cpu=None, make_default=True):
    """Returns a session that will use <num_cpu> CPU's only"""
    if num_cpu is None:
        num_cpu = int(os.getenv('RCALL_NUM_CPU', multiprocessing.cpu_count()))
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu,
        log_device_placement=False)
    tf_config.gpu_options.allocator_type = 'BFC'
  #  tf_config.gpu_options.per_process_gpu_memory_fraction = allocation
    tf_config.gpu_options.allow_growth = True
    if make_default:
        return tf.InteractiveSession(config=tf_config)
    else:
        return tf.Session(config=tf_config)


def GAE(rewards, values, dones, gamma, lam):
    nsteps = rewards.shape[0]
    if dones.shape != values.shape:
        rewards = rewards[..., None].repeat(values.shape[-1], axis=-1)
        dones = dones[..., None].repeat(values.shape[-1], axis=-1)
    assert (rewards.shape == values[:-1].shape) and (rewards.shape == dones[:-1].shape)
    advs = np.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(nsteps)):
        nextnonterminal = 1.0 - dones[t + 1]
        nextvalue = values[t + 1]
        delta = rewards[t] + gamma * nextnonterminal * nextvalue - values[t]
        advs[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
    return advs + values[:-1]


def fc(x, units, activation=None, name=None, reuse=False, init_scale=1.0, init_bias=0.0):
    return tf.layers.dense(x, units=units, activation=activation, name=name, reuse=reuse,
                           kernel_initializer=tf.orthogonal_initializer(init_scale),
                           bias_initializer=tf.constant_initializer(init_bias))


def gru(inputs, state, dones, units):
    cell = tf.nn.rnn_cell.GRUCell(units, kernel_initializer=tf.orthogonal_initializer)
    xs = tf.reshape(inputs, shape=(inputs.shape[0], -1, inputs.shape[-1]))  # [time, batch, dim]
    state = tf.reshape(state, shape=(-1, units))  # [batch, dim]
    dones = tf.reshape(dones, shape=xs.shape.as_list()[:-1] + [-1, ])  # [time, batch, 1]

    y = []
    for t in range(xs.shape[0]):
        state = state * (1 - dones[t])
        x = xs[t][None]
        y_t, state = tf.nn.dynamic_rnn(cell, x, initial_state=state, dtype=tf.float32, time_major=True)
        y.append(y_t)
    y = tf.reshape(tf.concat(y, axis=0), shape=inputs.shape[:-1].as_list() + [units, ])
    return y, state
