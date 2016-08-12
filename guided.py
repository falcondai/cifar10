import tensorflow as tf
import numpy as np
import os, sys, cPickle, time, glob, itertools
import tqdm
import argparse
from guided_relu_op import *
from pylab import *

def restore_vars(sess, checkpoint_path, latest, meta_path=None):
    """ Restore saved net, global score and step, and epsilons OR
    create checkpoint directory for later storage. """
    if latest:
        # use the latest checkpoint
        checkpoint_dir = checkpoint_path
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)

    if meta_path == None:
        meta_path = checkpoint_path + '.meta'

    saver = tf.train.import_meta_graph(meta_path)
    saver.restore(sess, checkpoint_path)

    print '* restoring from %s' % checkpoint_path
    print '* using metagraph from %s' % meta_path
    saver.restore(sess, checkpoint_path)
    return True

def main():
    # arguments
    parse = argparse.ArgumentParser()
    parse.add_argument('--checkpoint_path', required=True)
    parse.add_argument('--meta_path')
    parse.add_argument('--latest', action='store_true')

    args = parse.parse_args()

    # load data
    print '* loading data from'
    x = []
    y = []
    for path in glob.glob('data/cifar-10-batches-py/test_batch'):
        print path
        with open(path, 'rb') as f:
            d = cPickle.load(f)
        x.append(d['data'].reshape((-1, 3, 32, 32)).astype('float').transpose([0, 2, 3, 1]) / 255.)
        y.append(np.asarray(d['labels'], dtype='int64'))
    x = np.vstack(x)[:100]
    y = np.hstack(y)[:100]
    print x.shape, y.shape

    # load label array
    with open('data/cifar-10-batches-py/batches.meta', 'rb') as f:
        labels = cPickle.load(f)['label_names']
    print labels

    with tf.Session() as sess:
        restore_vars(sess, args.checkpoint_path, args.latest, args.meta_path)
        # model
        img_ph, keep_prob_ph = tf.get_collection('inputs')
        label_ph = tf.placeholder('int64', name='label')
        logits, probs = tf.get_collection('outputs')

        # loss
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, label_ph), name='loss')

        # evaluation
        predicted_labels = tf.argmax(logits, 1)
        correct_prediction = tf.equal(tf.argmax(logits, 1), label_ph)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        confusion_matrix = np.zeros((10, 10), dtype='int32')

        max_logit = tf.reduce_max(logits)

        n_samples = len(x)
        total_accuracy = 0.
        for i in tqdm.tqdm(xrange(n_samples)):
            val_feed = {
                img_ph: x[[i]],
                label_ph: y[[i]],
                keep_prob_ph: 1.0,
            }
            acc_val, pred_val = sess.run([accuracy, predicted_labels], feed_dict=val_feed)
            total_accuracy += acc_val
            confusion_matrix[y[i], pred_val[0]] += 1

            # guided back propagation
            with tf.get_default_graph().gradient_override_map({'Relu': 'GuidedRelu'}):
                # gs = tf.gradients(-loss, img_ph)[0].eval(feed_dict=val_feed)
                gs = tf.gradients(max_logit, img_ph)[0].eval(feed_dict=val_feed)
                suptitle('gt: %s pred: %s' % (labels[y[i]], labels[pred_val[0]]))
                subplot(321)
                imshow(x[i], interpolation='none')
                subplot(322)
                imshow(np.abs(gs[0]).max(axis=-1), interpolation='none', cmap='gray')
                subplot(323)
                imshow(np.maximum(0, gs[0]) / gs[0].max(), interpolation='none')
                subplot(324)
                imshow(np.maximum(0, -gs[0]) / (-gs[0]).max(), interpolation='none')
                subplot(325)
                imshow(x[i] + 0.1 * gs[0], interpolation='none')
                subplot(326)
                imshow(x[i] * np.expand_dims(np.abs(gs[0]).max(axis=-1), 2) / np.abs(gs[0]).max(), interpolation='none')
                savefig('guided/%i.png' % i)

        print 'total accuracy', total_accuracy / n_samples
        print confusion_matrix

if __name__ == '__main__':
    main()
