import tensorflow as tf


def conv_relu_pool3x3(input, kernel, s, name, pooling=True):
    w = tf.get_variable(name + 'conv_w', kernel,
                        dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    conv = tf.nn.conv2d(input, w, [1, s, s, 1], padding='SAME', name=name+'conv_c')
    b = tf.get_variable(name + 'conv_b', [kernel[3]], dtype=tf.float32, initializer=tf.constant_initializer(0))
    act = tf.nn.relu(conv + b, name=None)
    if pooling:
        return tf.nn.max_pool(act, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID')
    else:
        return act


def main():

    batch_size = 10
    timesteps = 4

    input_buffer = tf.placeholder(tf.float32, shape=(batch_size, 227, 227, 3))
    l1 = conv_relu_pool3x3(input_buffer, [11, 11, 3, 96], 4, 'l1_')
    l2 = conv_relu_pool3x3(l1, [5, 5, 96, 256], 1, 'l2_')
    l3 = conv_relu_pool3x3(l2, [3, 3, 256, 384], 1, 'l3_', False)
    l4 = conv_relu_pool3x3(l3, [3, 3, 384, 384], 1, 'l4_', False)
    l5 = conv_relu_pool3x3(l4, [3, 3, 384, 256], 1, 'l5_')
    l5f = tf.reshape(l5, [batch_size, 9216])
    l6 = tf.layers.dense(l5f, 4096, activation=tf.nn.relu)
    rnncell = tf.nn.rnn_cell.BasicRNNCell(4096, activation=tf.nn.relu)
    l7 = tf.nn.static_rnn(rnncell, [l6]*timesteps, initial_state=tf.zeros([batch_size, 4096]))
    l8 = tf.layers.dense(l7[0][timesteps-1], 1000, activation=tf.nn.sigmoid)


if __name__ == "__main__":
    main()
