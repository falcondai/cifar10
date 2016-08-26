#!/usr/bin/env python

# measure the impact of model architectures

import sys, glob, os
from train import train, load_train_data, build_argparser

def main():
    x, y = load_train_data('../data/cifar-10-batches-py')
    parse = build_argparser()
    # skip __init__.py
    for path in sorted(glob.glob('models/*.py'))[1:]:
        model_name = os.path.basename(path)[:-3]
        checkpoint_dir = 'checkpoints/model_arch/%s' % model_name
        print model_name
        if not os.path.exists(checkpoint_dir):
            hp = {
                'model': model_name,
                'batch_size': 256,
                'n_train_steps': int(sys.argv[1]),
                'checkpoint_dir': checkpoint_dir,
            }
            str_hp = sum(map(lambda k: ['--%s' % k, '%s' % hp[k]], hp), []) + ['--restart']
            print '* arguments'
            print str_hp
            args = parse.parse_args(str_hp)
            train(x, y, vars(args))
        else:
            print '%s exists' % checkpoint_dir

if __name__ == '__main__':
    main()
