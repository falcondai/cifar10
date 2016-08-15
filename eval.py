#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import os, sys, cPickle, time, glob, itertools
import tqdm
import argparse

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
    parse.add_argument('--batch_size', type=int, default=32)

    args = parse.parse_args()
    n_classes = 10

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
    x = np.vstack(x)
    y = np.hstack(y)
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
        confusion_matrix = np.zeros((n_classes, n_classes), dtype='int32')

        n_samples = len(x)
        n_steps = np.ceil(n_samples * 1. / args.batch_size)
        total_accuracy = 0.
        for i in tqdm.tqdm(xrange(int(n_steps))):
            start = i * args.batch_size
            end = min(start + args.batch_size, n_samples)
            val_feed = {
                img_ph: x[start:end],
                label_ph: y[start:end],
                keep_prob_ph: 1.0,
            }
            acc_val, pred_val = sess.run([accuracy, predicted_labels], feed_dict=val_feed)
            total_accuracy += acc_val * (end - start)
            for gt, pred in zip(y[start:end], pred_val):
                confusion_matrix[gt, pred] += 1
        print 'total accuracy', total_accuracy / n_samples
        print confusion_matrix
        # per class accuracy
        for k in xrange(n_classes):
            count = confusion_matrix[k].sum()
            precision = confusion_matrix[k, k] * 1. / confusion_matrix[:, k].sum()
            recall = confusion_matrix[k, k] * 1. / confusion_matrix[k, :].sum()
            print 'class %i %s: count %i precision %g recall %g' % (k, labels[k], count, precision, recall)

if __name__ == '__main__':
    main()
