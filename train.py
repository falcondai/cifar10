import tensorflow as tf
import numpy as np
import os, sys, cPickle, time, glob, itertools
import tqdm
import argparse
import importlib

def restore_vars(saver, sess, checkpoint_dir, restart=False):
    """ Restore saved net, global score and step, and epsilons OR
    create checkpoint directory for later storage. """
    sess.run(tf.initialize_all_variables())

    if not os.path.exists(checkpoint_dir):
        try:
            os.makedirs(checkpoint_dir)
        except OSError:
            pass

    if not restart:
        path = tf.train.latest_checkpoint(checkpoint_dir)
        if path is None:
            print '* no existing checkpoint found'
            return False
        else:
            print '* restoring from %s' % path
            # meta_path = path + '.meta'
            # old_saver = tf.train.import_meta_graph(meta_path)
            # old_saver.restore(sess, path)
            saver.restore(sess, path)
            return True

def main():
    # arguments
    parse = argparse.ArgumentParser()
    parse.add_argument('--checkpoint_dir', required=True)
    parse.add_argument('--batch_size', type=int, default=32)
    parse.add_argument('--n_eval_interval', type=int, default=8)
    parse.add_argument('--n_save_interval', type=int, default=16)
    parse.add_argument('--n_train_steps', type=int, default=1024)
    parse.add_argument('--model', required=True)
    parse.add_argument('--optimizer', choices=['adam', 'momentum'], default='momentum')
    parse.add_argument('--initial_learning_rate', type=float, default=0.01)
    parse.add_argument('--n_decay_steps', type=int, default=512)
    parse.add_argument('--no_decay_staircase', action='store_true')
    parse.add_argument('--decay_rate', type=float, default=0.8)
    parse.add_argument('--dropout_rate', type=float, default=0.2)
    parse.add_argument('--momentum', type=float, default=0.8)
    parse.add_argument('--beta1', type=float, default=0.9)
    parse.add_argument('--beta2', type=float, default=0.999)
    parse.add_argument('--epsilon', type=float, default=1e-8)
    parse.add_argument('--np_seed', type=int, default=123)
    parse.add_argument('--tf_seed', type=int, default=1234)
    parse.add_argument('--restart', action='store_true')

    args = parse.parse_args()
    summary_dir = 'tf-log/%s-%d' % (os.path.basename(args.checkpoint_dir), time.time())
    np.random.seed(args.np_seed)
    tf.set_random_seed(args.tf_seed)
    model = importlib.import_module('models.%s' % args.model)

    # load data
    print '* loading data from'
    x = []
    y = []
    for path in glob.glob('data/cifar-10-batches-py/data_batch_*'):
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

    # model
    img_ph, keep_prob_ph, logits, probs = model.build_model()
    label_ph = tf.placeholder('int64', name='label')

    # loss
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, label_ph), name='loss')

    # optimization
    global_step = tf.Variable(0, trainable=False, name='global_step')
    learning_rate = tf.train.exponential_decay(args.initial_learning_rate, global_step, args.n_decay_steps, args.decay_rate, staircase=not args.no_decay_staircase)

    if args.optimizer == 'adam':
        train_op = tf.train.AdamOptimizer(learning_rate, args.beta1, args.beta2, args.epsilon).minimize(loss, global_step=global_step)
    else:
        train_op = tf.train.MomentumOptimizer(learning_rate, args.momentum).minimize(loss, global_step=global_step)

    # evaluation
    correct_prediction = tf.equal(tf.argmax(logits, 1), label_ph)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # summary
    tf.scalar_summary('learning_rate', learning_rate)
    tf.scalar_summary('loss', loss)
    tf.scalar_summary('accuracy', accuracy)
    summary_op = tf.merge_all_summaries()

    saver = tf.train.Saver(max_to_keep=2, keep_checkpoint_every_n_hours=1)
    with tf.Session() as sess:
        writer = tf.train.SummaryWriter(summary_dir, sess.graph)
        restore_vars(saver, sess, args.checkpoint_dir, args.restart)

        n_samples = len(x)
        for i in tqdm.tqdm(xrange(args.n_train_steps)):
            ind = np.random.choice(n_samples, args.batch_size, replace=False)
            if i % args.n_eval_interval == 0:
                val_feed = {
                    img_ph: x[ind],
                    label_ph: y[ind],
                    keep_prob_ph: 1.0,
                }
                # print 'step', i,
                # print 'loss %g accuracy %g', sess.run([loss, accuracy], feed_dict=val_feed)
                writer.add_summary(sess.run(summary_op, feed_dict=val_feed), global_step.eval())

            train_feed = {
                img_ph: x[ind],
                label_ph: y[ind],
                keep_prob_ph: 1. - args.dropout_rate,
            }
            train_op.run(feed_dict=train_feed)

            if i % args.n_save_interval == 0:
                saver.save(sess, args.checkpoint_dir + '/model', global_step=global_step.eval())


if __name__ == '__main__':
    main()
