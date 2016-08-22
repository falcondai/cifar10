import tensorflow as tf
import numpy as np

def build_model(n_classes=10):
    img_ph = tf.placeholder('float', [None, 32, 32, 3], name='image')
    keep_prob_ph = tf.placeholder('float', name='keep_prob')
    tf.add_to_collection('inputs', img_ph)
    tf.add_to_collection('inputs', keep_prob_ph)

    with tf.variable_scope('model'):
        conv1 = tf.contrib.layers.convolution2d(
            inputs=tf.nn.dropout(img_ph, keep_prob_ph),
            num_outputs=64,
            kernel_size=(7, 7),
            activation_fn=tf.nn.relu,
            biases_initializer=tf.zeros,
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            scope='conv1'
        )

        conv2 = tf.contrib.layers.convolution2d(
            inputs=tf.nn.dropout(conv1, keep_prob_ph),
            num_outputs=128,
            kernel_size=(5, 5),
            # stride=(2, 2),
            activation_fn=tf.nn.relu,
            biases_initializer=tf.zeros,
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            scope='conv2'
        )

        conv3 = tf.contrib.layers.convolution2d(
            inputs=tf.nn.dropout(conv2, keep_prob_ph),
            num_outputs=128,
            kernel_size=(5, 5),
            # stride=(2, 2),
            activation_fn=tf.nn.relu,
            biases_initializer=tf.zeros,
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            scope='conv3'
        )

        conv4 = tf.contrib.layers.convolution2d(
            inputs=tf.nn.dropout(conv3, keep_prob_ph),
            num_outputs=128,
            kernel_size=(5, 5),
            stride=(2, 2),
            activation_fn=tf.nn.relu,
            biases_initializer=tf.zeros,
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            scope='conv4'
        )

        conv5 = tf.contrib.layers.convolution2d(
            inputs=tf.nn.dropout(conv4, keep_prob_ph),
            num_outputs=64,
            kernel_size=(3, 3),
            activation_fn=tf.nn.relu,
            biases_initializer=tf.zeros,
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            scope='conv5'
        )

        conv6 = tf.contrib.layers.convolution2d(
            inputs=tf.nn.dropout(conv5, keep_prob_ph),
            num_outputs=64,
            kernel_size=(3, 3),
            stride=(2, 2),
            activation_fn=tf.nn.relu,
            biases_initializer=tf.zeros,
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            scope='conv6'
        )

        conv_output = conv6

        conv_shape = conv_output.get_shape().as_list()
        flat_dim = np.product(conv_shape[1:])
        print conv_shape, flat_dim

        fc1 = tf.contrib.layers.fully_connected(
            inputs=tf.contrib.layers.flatten(conv_output),
            num_outputs=128,
            activation_fn=tf.nn.relu,
            biases_initializer=tf.zeros,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            scope='fc1'
        )

        fc2 = tf.contrib.layers.fully_connected(
            inputs=fc1,
            num_outputs=n_classes,
            biases_initializer=tf.zeros,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            scope='fc2'
        )

        logits = fc2
        probs = tf.nn.softmax(logits)
        tf.add_to_collection('outputs', logits)
        tf.add_to_collection('outputs', probs)

    return img_ph, keep_prob_ph, logits, probs
