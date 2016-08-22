#!/usr/bin/env python

import sys
from train import train, load_train_data, build_argparser

def main():
    x, y = load_train_data('../data/cifar-10-batches-py')
    parse = build_argparser()
    for seed in xrange(int(sys.argv[1])):
        hp = {
            'model': 'cp2f1d',
            'batch_size': 512,
            'n_train_steps': int(sys.argv[2]),
            'np_seed': seed,
            'checkpoint_dir': 'checkpoints/train_order/cp2f1d-s%i' % seed,
        }
        str_hp = sum(map(lambda k: ['--%s' % k, '%s' % hp[k]], hp), []) + ['--restart']
        print '* arguments'
        print str_hp
        args = parse.parse_args(str_hp)
        train(x, y, vars(args))

if __name__ == '__main__':
    main()
