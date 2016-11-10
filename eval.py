#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import os, sys, cPickle, time, glob, itertools, csv
import tqdm
import argparse

def restore_vars(sess, checkpoint_path, meta_path):
    """ Restore saved net, global score and step, and epsilons OR
    create checkpoint directory for later storage. """
    saver = tf.train.import_meta_graph(meta_path)
    saver.restore(sess, checkpoint_path)

    print '* restoring from %s' % checkpoint_path
    print '* using metagraph from %s' % meta_path
    saver.restore(sess, checkpoint_path)
    return True

def load_test_data(data_dir='data/cifar-10-batches-py'):
    # load data
    print '* loading data from'
    x = []
    y = []
    for path in glob.glob('%s/test_batch' % data_dir):
        print path
        with open(path, 'rb') as f:
            d = cPickle.load(f)
        x.append(d['data'].reshape((-1, 3, 32, 32)).astype('float').transpose([0, 2, 3, 1]) / 255.)
        y.append(np.asarray(d['labels'], dtype='int64'))
    x = np.vstack(x)
    y = np.hstack(y)
    print '* dataset shapes:', x.shape, y.shape
    return x, y

def evaluate(x, y, labels, checkpoint_path, meta_path, batch_size):
    n_classes = 10
    with tf.Graph().as_default() as g:
        with tf.Session() as sess:
            restore_vars(sess, checkpoint_path, meta_path)

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
            print confusion_matrix

            # per class accuracy
            for k in xrange(n_classes):
                count = confusion_matrix[k].sum()
                precision = confusion_matrix[k, k] * 1. / confusion_matrix[:, k].sum()
                recall = confusion_matrix[k, k] * 1. / confusion_matrix[k, :].sum()
                print 'class {} {:>16}:\tcount {}\tprecision {:.2%}\trecall {:.2%}'.format(k, labels[k], count, precision, recall)

            total_accuracy /= n_samples
            print 'total accuracy', total_accuracy
            return total_accuracy

if __name__ == '__main__':
    # arguments
    parse = argparse.ArgumentParser()
    parse.add_argument('--checkpoint_path', required=True)
    parse.add_argument('--meta_path')
    parse.add_argument('--latest', action='store_true')
    parse.add_argument('--batch_size', type=int, default=32)
    parse.add_argument('--scan', action='store_true')
    parse.add_argument('--report_path')

    args = parse.parse_args()

    # load label array
    with open('data/cifar-10-batches-py/batches.meta', 'rb') as f:
        labels = cPickle.load(f)['label_names']
    print '* classes', labels

    x, y = load_test_data()
    if args.scan:
        if args.report_path != None:
            f = open(args.report_path, 'wb')
            writer = csv.DictWriter(f, ['checkpoint_path', 'accuracy'])
            writer.writeheader()
        for path in sorted(glob.glob('%s/*' % args.checkpoint_path)):
            try:
                checkpoint_path = tf.train.latest_checkpoint(path)
                meta_path = checkpoint_path + '.meta'
                accuracy = evaluate(x, y, labels, checkpoint_path, meta_path, args.batch_size)
            except:
                accuracy = 'NA'
            if args.report_path != None:
                writer.writerow({'checkpoint_path': checkpoint_path, 'accuracy': accuracy})
        if args.report_path != None:
            f.close()
    else:
        if args.latest:
            # use the latest checkpoint in the folder
            checkpoint_path = tf.train.latest_checkpoint(args.checkpoint_path)
        else:
            checkpoint_path = args.checkpoint_path

        if args.meta_path == None:
            meta_path = checkpoint_path + '.meta'
        else:
            meta_path = args.meta_path

        evaluate(x, y, labels, checkpoint_path, meta_path, args.batch_size)
