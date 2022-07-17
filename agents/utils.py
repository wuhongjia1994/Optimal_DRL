import tensorflow as tf


def fc(x, units, activation=None, name=None, reuse=False, init_scale=1.0, init_bias=0.0):
    return tf.layers.dense(x, units=units, activation=activation, name=name, reuse=reuse,
                           kernel_initializer=tf.orthogonal_initializer(init_scale),
                           bias_initializer=tf.constant_initializer(init_bias))


def mlp(x, units=64, layer_n=1, activ=tf.nn.relu, feat_norm=True):
    if feat_norm:
        x = tf.contrib.layers.layer_norm(x, begin_norm_axis=-1, begin_params_axis=-1)
    for _ in range(layer_n):
        x = fc(x, units=units, activation=activ)
    return x



def ln_gru(inputs, input_states, masks, units, chunk=None):
    cell = tf.nn.rnn_cell.GRUCell(units, kernel_initializer=tf.orthogonal_initializer(1),
                                  bias_initializer=tf.constant_initializer(0))
    xs = tf.reshape(inputs, shape=(inputs.shape[0], -1, inputs.shape[-1]))  # [time, batch, dim]
    masks = tf.reshape(masks, shape=(masks.shape[0], -1, 1))  # [time, batch, 1]

    y = []
    states = []
    state = input_states[0]
    for t in range(xs.shape[0]):
        state_masked = state * (1 - masks[t])
        if chunk is not None and t % chunk == 0:
            state_masked = input_states[t] * (1 - masks[t])
        y_t, state = tf.nn.dynamic_rnn(cell, xs[t][None], initial_state=state_masked, dtype=tf.float32, time_major=True)
        y.append(y_t)
        states.append(state)
    y = tf.reshape(tf.concat(y, axis=0), shape=inputs.shape[:-1].as_list() + [units, ])
    y = tf.contrib.layers.layer_norm(y, begin_norm_axis=-1, begin_params_axis=-1)    # layer norm
    states = tf.reshape(tf.concat(states, axis=0), shape=input_states.shape)
    return y, states
