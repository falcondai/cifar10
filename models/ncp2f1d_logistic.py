import tensorflow as tf
import numpy as np

def build_model(n_classes=10, batch=None, is_training=True):
    img_ph = tf.placeholder('float', [batch, 32, 32, 3], name='image')
    keep_prob_ph = tf.placeholder('float', name='keep_prob')
    tf.add_to_collection('inputs', img_ph)
    tf.add_to_collection('inputs', keep_prob_ph)

    with tf.variable_scope('model'):
        norm1 = tf.contrib.layers.batch_norm(
            inputs=img_ph,
            is_training=is_training,
            scope='norm1',
        )
        conv1 = tf.contrib.layers.convolution2d(
            inputs=norm1,
            num_outputs=64,
            kernel_size=(5, 5),
            activation_fn=tf.sigmoid,
            biases_initializer=tf.zeros_initializer,
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            scope='conv1'
        )
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, conv1)

        pool1 = tf.nn.max_pool(conv1, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME', name='pool1')

        norm2 = tf.contrib.layers.batch_norm(
            inputs=pool1,
            is_training=is_training,
            scope='norm2',
        )

        conv2 = tf.contrib.layers.convolution2d(
            inputs=norm2,
            num_outputs=64,
            kernel_size=(5, 5),
            activation_fn=tf.sigmoid,
            biases_initializer=tf.zeros_initializer,
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            scope='conv2'
        )
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, conv2)

        pool2 = tf.nn.max_pool(conv2, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME', name='pool2')

        conv_output = pool2

        fc1 = tf.contrib.layers.fully_connected(
            inputs=tf.nn.dropout(tf.contrib.layers.flatten(conv_output), keep_prob_ph),
            num_outputs=n_classes,
            biases_initializer=tf.zeros_initializer,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            activation_fn=None,
            scope='fc1'
        )
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, fc1)

        logits = fc1
        probs = tf.nn.softmax(logits, name='probs')
        tf.add_to_collection('outputs', logits)
        tf.add_to_collection('outputs', probs)

    return img_ph, keep_prob_ph, logits, probs
