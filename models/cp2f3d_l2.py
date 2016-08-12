import tensorflow as tf
import numpy as np

def build_model(width=32, n_classes=10, batch=None):
    img_ph = tf.placeholder('float', [batch, width, width, 3], name='image')
    keep_prob_ph = tf.placeholder('float', name='keep_prob')
    tf.add_to_collection('inputs', img_ph)
    tf.add_to_collection('inputs', keep_prob_ph)

    with tf.variable_scope('model'):
        conv1 = tf.contrib.layers.convolution2d(
            inputs=img_ph,
            num_outputs=64,
            kernel_size=(5, 5),
            activation_fn=tf.nn.relu,
            biases_initializer=tf.zeros_initializer,
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            scope='conv1'
        )

        pool1 = tf.contrib.layers.max_pool2d(inputs=conv1, kernel_size=[3, 3], stride=[2, 2], scope='pool1')

        conv2 = tf.contrib.layers.convolution2d(
            inputs=pool1,
            num_outputs=64,
            kernel_size=(5, 5),
            activation_fn=tf.nn.relu,
            biases_initializer=tf.zeros_initializer,
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            scope='conv2'
        )

        pool2 = tf.contrib.layers.max_pool2d(inputs=conv2, kernel_size=[3, 3], stride=[2, 2], scope='pool2')

        conv_output = pool2

        fc1 = tf.contrib.layers.fully_connected(
            inputs=tf.contrib.layers.flatten(conv_output),
            num_outputs=384,
            activation_fn=tf.nn.relu,
            biases_initializer=tf.zeros_initializer,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            weights_regularizer=tf.contrib.layers.l2_regularizer(1.),
            scope='fc1'
        )

        fc2 = tf.contrib.layers.fully_connected(
            inputs=tf.nn.dropout(fc1, keep_prob_ph),
            num_outputs=192,
            activation_fn=tf.nn.relu,
            biases_initializer=tf.zeros_initializer,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            weights_regularizer=tf.contrib.layers.l2_regularizer(1.),
            scope='fc2'
        )

        fc3 = tf.contrib.layers.fully_connected(
            inputs=tf.nn.dropout(fc2, keep_prob_ph),
            num_outputs=n_classes,
            biases_initializer=tf.zeros_initializer,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            scope='fc3'
        )

        logits = fc3
        probs = tf.nn.softmax(logits, name='probs')
        tf.add_to_collection('outputs', logits)
        tf.add_to_collection('outputs', probs)

    return img_ph, keep_prob_ph, logits, probs
