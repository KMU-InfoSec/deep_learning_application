import configparser
import tensorflow as tf
from tensorflow.contrib import rnn, layers

# config parameters
config = configparser.ConfigParser()
config.read('config.ini')

# define parameters
input_layer_size = int(config.get('CLASSIFIER', 'INPUT_SIZE'))
output_layer_size = int(config.get('CLASSIFIER', 'OUTPUT_SIZE'))

filter_list = [int(x) for x in config.get('CLASSIFIER', 'CONV_FILTER').split(',')]
kernel_size_list = [int(x) for x in config.get('CLASSIFIER', 'CONV_KERNEL').split(',')]

hidden_node_list = [int(x) for x in config.get('CLASSIFIER', 'HIDDEN_NODE').split(',')]


def inference_ANN(x, prob, train_flag=False):
    # classifier
    dense_layer = x
    for hidden_node in hidden_node_list:
        dense_layer = tf.layers.dense(inputs=dense_layer, units=hidden_node, activation=tf.nn.relu)
        dense_layer = tf.layers.batch_normalization(dense_layer, axis=-1)
        # if train_flag:
        #     dense_layer = tf.nn.dropout(dense_layer, prob)

    y_ = tf.layers.dense(inputs=dense_layer, units=output_layer_size)

    return y_, dense_layer


def inference_CNN(x, prob, L2_REGULARIZATION_SCALE, train_flag=False):
    # convolution part
    conv_layer = tf.reshape(x, [-1, input_layer_size, 1])  # tensor : [N(batch #), W(width), C(Channel)]
    for filter, kernel_size in zip(filter_list, kernel_size_list):

        conv_layer = tf.layers.conv1d(inputs=conv_layer, filters=filter, kernel_size=kernel_size,
                                      padding='same', data_format='channels_last',
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REGULARIZATION_SCALE))
        conv_layer = tf.layers.batch_normalization(conv_layer, axis=-1)
        conv_layer = tf.nn.leaky_relu(conv_layer)  # tf.nn.relu(conv_layer)
        conv_layer = tf.layers.max_pooling1d(inputs=conv_layer, pool_size=2,
                                             padding='valid', strides=2)
        # conv_layer = tf.nn.dropout(conv_layer, keep_prob=prob)

    flatten_layer = tf.layers.flatten(conv_layer)

    # classifier
    dense_layer = flatten_layer
    for hidden_node in hidden_node_list:
        dense_layer = tf.layers.dense(inputs=dense_layer, units=hidden_node, activation=tf.nn.leaky_relu,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REGULARIZATION_SCALE))
        dense_layer = tf.layers.dropout(dense_layer, prob, training=train_flag)

    y_ = tf.layers.dense(inputs=dense_layer, units=output_layer_size)

    return y_, flatten_layer


def inference_CNN2(x, prob, train_flag=False):
    # convolution part
    conv_layer = tf.reshape(x, [-1, input_layer_size, 2, 1])  # tensor : [N(batch #), W(width), C(Channel)]
    for filter, kernel_size in zip(filter_list, kernel_size_list):
        conv_layer = tf.layers.conv2d(inputs=conv_layer, filters=filter, kernel_size=kernel_size,
                                      padding='same', data_format='channels_last')
        conv_layer = tf.layers.batch_normalization(conv_layer, axis=-1)
        conv_layer = tf.nn.leaky_relu(conv_layer)  # tf.nn.relu(conv_layer)
        conv_layer = tf.layers.max_pooling2d(inputs=conv_layer, pool_size=(2,1), padding='valid', strides=(2,1))
        # conv_layer = tf.nn.dropout(conv_layer, keep_prob=prob)

    flatten_layer = tf.layers.flatten(conv_layer)

    # classifier
    dense_layer = flatten_layer
    for hidden_node in hidden_node_list:
        dense_layer = tf.layers.dense(inputs=dense_layer, units=hidden_node, activation=tf.nn.leaky_relu)
        dense_layer = tf.layers.dropout(dense_layer, prob, training=train_flag)

    y_ = tf.layers.dense(inputs=dense_layer, units=output_layer_size)

    return y_, flatten_layer


def inference_CNN_2D(x, prob, train_flag=False):
    _IMAGE_SIZE_DICT = {256: [16, 16], 512: [16, 32], 768: [24, 32]}
    layer_size = max(_IMAGE_SIZE_DICT[input_layer_size])

    # convolution part
    conv_layer = tf.reshape(x, [-1, layer_size, layer_size, 1])  # tensor : [N(batch #), W(width), C(Channel)]

    # conv
    conv_layer = tf.layers.conv2d(inputs=conv_layer, filters=32, kernel_size=2,
                                      padding='same', data_format='channels_last')
    conv_layer = tf.nn.tanh(conv_layer)
    # conv
    conv_layer = tf.layers.conv2d(inputs=conv_layer, filters=32, kernel_size=2,
                                  padding='same', data_format='channels_last')
    conv_layer = tf.nn.tanh(conv_layer)
    # pool
    conv_layer = tf.layers.max_pooling2d(inputs=conv_layer, pool_size=2,
                                         padding='valid', strides=2)
    conv_layer = tf.layers.dropout(conv_layer, rate=prob, training=train_flag)
    # conv
    conv_layer = tf.layers.conv2d(inputs=conv_layer, filters=32, kernel_size=2,
                                  padding='same', data_format='channels_last')
    conv_layer = tf.nn.tanh(conv_layer)
    # pool
    conv_layer = tf.layers.max_pooling2d(inputs=conv_layer, pool_size=2,
                                         padding='valid', strides=2)
    conv_layer = tf.layers.dropout(conv_layer, rate=prob, training=train_flag)
    # flatten
    flatten_layer = tf.layers.flatten(conv_layer)
    # fc
    dense_layer = flatten_layer
    dense_layer = tf.layers.dense(inputs=dense_layer, units=512, activation=tf.nn.tanh)
    dense_layer = tf.layers.dense(inputs=dense_layer, units=256, activation=tf.nn.tanh)
    y_ = tf.layers.dense(inputs=dense_layer, units=output_layer_size)

    return y_, flatten_layer


def inference_LSTM(x, prob=1.0, train_flag=False):
    # variable initialization
    hidden_size = int(config.get('CLASSIFIER', 'HIDDEN_SIZE'))
    max_seq_length = int(config.get('CLASSIFIER', 'MAX_SEQ_LENGTH'))
    no_rnn_layers = int(config.get('CLASSIFIER', 'NO_RNN_LAYERS'))
    output_size = int(config.get('CLASSIFIER', 'OUTPUT_SIZE'))

    cell = rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True, activation=tf.tanh)
    cells = [cell for _ in range(no_rnn_layers)]
    cells = rnn.MultiRNNCell(cells)
    # cell = rnn.DropoutWrapper(cell=cell, output_keep_prob=prob)

    outputs, state = tf.nn.dynamic_rnn(cell=cells, inputs=x, dtype=tf.float32)  # state: usually use final values

    y_reshape = tf.reshape(outputs, [-1, max_seq_length * hidden_size])  # 3278
    # y_reshape = tf.reshape(state, [-1, 6*hidden_size])  # why 6?

    y_fc = layers.fully_connected(inputs=y_reshape,
                                  num_outputs=output_size,
                                  activation_fn=None)

    return y_fc