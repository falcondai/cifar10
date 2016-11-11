#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import os, sys, cPickle, time, glob, itertools, json
import tqdm
import argparse
import importlib

from eval import load_test_data

def get_current_run_id(checkpoint_dir):
    paths = glob.glob('%s/hyperparameters.*.json' % checkpoint_dir)
    if len(paths) == 0:
        return 0
    return sorted(map(lambda p: int(p.split('.')[-2]), paths))[-1] + 1

def restore_vars(saver, sess, checkpoint_dir, restart=False):
    """ Restore saved net, global score and step, and epsilons OR
    create checkpoint directory for later storage. """
    sess.run(tf.initialize_all_variables())

    if not restart:
        path = tf.train.latest_checkpoint(checkpoint_dir)
        if path is None:
            print '* no existing checkpoint found'
            return False
        else:
            print '* restoring from %s' % path
            saver.restore(sess, path)
            return True
    print '* overwriting checkpoints at %s' % checkpoint_dir
    return False

def load_train_data(data_dir='data/cifar-10-batches-py'):
    # load data
    print '* loading data from'
    x = []
    y = []
    for path in sorted(glob.glob('%s/data_batch_*' % data_dir)):
        print path
        with open(path, 'rb') as f:
            d = cPickle.load(f)
        x.append(d['data'].reshape((-1, 3, 32, 32)).astype('float').transpose([0, 2, 3, 1]) / 255.)
        y.append(np.asarray(d['labels'], dtype='int64'))
    x = np.vstack(x)
    y = np.hstack(y)
    print '* dataset shapes:', x.shape, y.shape
    return x, y

def train(x, y, args, build_model, test_x=None, test_y=None):
    summary_dir = 'tf-log/%s%d-%s' % (args['summary_prefix'], time.time(), os.path.basename(args['checkpoint_dir']))

    # set seeds
    np.random.seed(args['np_seed'])
    tf.set_random_seed(args['tf_seed'])

    # create checkpoint dirs
    if not os.path.exists(args['checkpoint_dir']):
        try:
            os.makedirs(args['checkpoint_dir'])
        except OSError:
            pass

    print '* training hyperparameters:'
    for k in sorted(args.keys()):
        print k, args[k]
    n_run = get_current_run_id(args['checkpoint_dir'])
    with open('%s/hyperparameters.%i.json' % (args['checkpoint_dir'], n_run), 'wb') as hpf:
        json.dump(args, hpf)

    with tf.Graph().as_default() as g:
        # model
        print '* building model %s' % args['model']
        img_ph, keep_prob_ph, logits, probs = build_model()
        label_ph = tf.placeholder('int64', name='label')

        # loss
        regularizer = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        class_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, label_ph), name='class_loss')
        loss = class_loss + args['reg_coeff'] * regularizer

        # optimization
        global_step = tf.Variable(0, trainable=False, name='global_step')
        learning_rate = tf.train.exponential_decay(args['initial_learning_rate'], global_step, args['n_decay_steps'], args['decay_rate'], staircase=not args['no_decay_staircase'])

        if args['optimizer'] == 'adam':
            train_op = tf.train.AdamOptimizer(learning_rate, args['adam_beta1'], args['adam_beta2'], args['adam_epsilon']).minimize(loss, global_step=global_step)
        if args['optimizer'] == 'ag':
            train_op = tf.train.MomentumOptimizer(learning_rate, args['momentum'], use_nesterov=True).minimize(loss, global_step=global_step)
        else:
            train_op = tf.train.MomentumOptimizer(learning_rate, args['momentum']).minimize(loss, global_step=global_step)

        # evaluation
        correct_prediction = tf.equal(tf.argmax(logits, 1), label_ph)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # summary
        if not args['no_summary']:
            tf.scalar_summary('train/learning_rate', learning_rate)
            tf.scalar_summary('train/class_loss', class_loss)
            tf.scalar_summary('train/regularizer', regularizer)
            tf.scalar_summary('train/total_loss', loss)
            tf.scalar_summary('train/accuracy', accuracy)

            print '* extra summary'
            for v in tf.get_collection(tf.GraphKeys.ACTIVATIONS):
                tf.histogram_summary('activations/%s' % v.name, v)
                print 'activations/%s' % v.name
                tf.scalar_summary('sparsity/%s' % v.name, tf.nn.zero_fraction(v))
                print 'sparsity/%s' % v.name

            for v in tf.trainable_variables():
                tf.histogram_summary('gradients/%s' % v.name, tf.gradients(loss, v))
                print 'gradients/%s' % v.name

            summary_op = tf.merge_all_summaries()

            if args['test_model']:
                test_class_loss = tf.placeholder('float')
                test_accuracy = tf.placeholder('float')
                test_summary_op = tf.merge_summary([
                    tf.scalar_summary('test/class_loss', test_class_loss),
                    tf.scalar_summary('test/accuracy', test_accuracy),
                ])

        saver = tf.train.Saver(max_to_keep=2, keep_checkpoint_every_n_hours=1)
        with tf.Session() as sess:
            if not args['no_summary']:
                writer = tf.train.SummaryWriter(summary_dir, sess.graph, flush_secs=60)
            restore_vars(saver, sess, args['checkpoint_dir'], args['restart'])

            print '* regularized parameters:'
            for v in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES):
                print v.name

            n_samples = len(x)
            for i in tqdm.tqdm(xrange(args['n_train_steps'])):
                start = np.random.randint(0, n_samples - args['batch_size'])
                batch_x = x[start:start + args['batch_size']]
                batch_y = y[start:start + args['batch_size']]
                if i % args['n_eval_interval'] == 0:
                    val_feed = {
                        img_ph: batch_x,
                        label_ph: batch_y,
                        keep_prob_ph: 1.0,
                    }
                    if not args['no_summary']:
                        writer.add_summary(sess.run(summary_op, feed_dict=val_feed), global_step.eval())

                        if args['test_model']:
                            test_class_loss_val = 0.
                            test_accuracy_val = 0.
                            test_sample_size = len(test_x)
                            for ti in xrange(0, test_sample_size, args['batch_size']):
                                batch_tx = test_x[ti:ti + args['batch_size']]
                                batch_ty = test_y[ti:ti + args['batch_size']]
                                batch_n = len(batch_tx)
                                test_feed = {
                                    img_ph: batch_tx,
                                    label_ph: batch_ty,
                                    keep_prob_ph: 1.0,
                                }
                                batch_loss, batch_acc = sess.run([class_loss, accuracy], feed_dict=test_feed)
                                test_class_loss_val += batch_n * batch_loss
                                test_accuracy_val += batch_n * batch_acc
                            writer.add_summary(sess.run(test_summary_op, feed_dict={
                                test_class_loss: test_class_loss_val / test_sample_size,
                                test_accuracy: test_accuracy_val / test_sample_size,
                            }), global_step.eval())

                train_feed = {
                    img_ph: batch_x,
                    label_ph: batch_y,
                    keep_prob_ph: 1. - args['dropout_rate'],
                }
                train_op.run(feed_dict=train_feed)

                if i % args['n_save_interval'] == 0:
                    saver.save(sess, args['checkpoint_dir'] + '/model', global_step=global_step.eval())

            # save again at the end
            saver.save(sess, args['checkpoint_dir'] + '/model', global_step=global_step.eval())

def build_argparser():
    parse = argparse.ArgumentParser()
    parse.add_argument('--checkpoint_dir', required=True)
    parse.add_argument('--batch_size', type=int, default=32)
    parse.add_argument('--test_model', action='store_true')
    parse.add_argument('--n_eval_interval', type=int, default=8)
    parse.add_argument('--n_save_interval', type=int, default=16)
    parse.add_argument('--n_train_steps', type=int, default=1024)
    parse.add_argument('--model', required=True)
    parse.add_argument('--optimizer', choices=['adam', 'momentum', 'ag'], default='momentum')
    parse.add_argument('--initial_learning_rate', type=float, default=0.01)
    parse.add_argument('--n_decay_steps', type=int, default=512)
    parse.add_argument('--no_decay_staircase', action='store_true')
    parse.add_argument('--decay_rate', type=float, default=0.8)
    parse.add_argument('--dropout_rate', type=float, default=0.2)
    parse.add_argument('--reg_coeff', type=float, default=0.0001)
    parse.add_argument('--momentum', type=float, default=0.8)
    parse.add_argument('--adam_beta1', type=float, default=0.9)
    parse.add_argument('--adam_beta2', type=float, default=0.999)
    parse.add_argument('--adam_epsilon', type=float, default=1e-8)
    parse.add_argument('--np_seed', type=int, default=123)
    parse.add_argument('--tf_seed', type=int, default=1234)
    parse.add_argument('--restart', action='store_true')
    parse.add_argument('--no_summary', action='store_true')
    parse.add_argument('--summary_prefix', default='')

    return parse


if __name__ == '__main__':
    # arguments
    parse = build_argparser()
    args = parse.parse_args()

    # model
    model = importlib.import_module('models.%s' % args['model'])

    # load data
    x, y = load_train_data()
    if args.test_model:
        tx, ty = load_test_data()
        # train model
        train(x, y, vars(args), model.build_model, tx, ty)
    else:
        train(x, y, vars(args), model.build_model)
