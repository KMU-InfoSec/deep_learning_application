import configparser
import tensorflow as tf

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
        # if train_flag:
        #     dense_layer = tf.nn.dropout(dense_layer, prob)

    y_ = tf.layers.dense(inputs=dense_layer, units=output_layer_size)

    return y_


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
        # if train_flag:
        #     conv_layer = tf.nn.dropout(conv_layer, keep_prob=prob)

    # classifier
    dense_layer = tf.layers.flatten(conv_layer)
    for hidden_node in hidden_node_list:
        dense_layer = tf.layers.dense(inputs=dense_layer, units=hidden_node, activation=tf.nn.leaky_relu,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REGULARIZATION_SCALE))
        dense_layer = tf.layers.dropout(dense_layer, prob, training=train_flag)

    y_ = tf.layers.dense(inputs=dense_layer, units=output_layer_size)

    return y_
