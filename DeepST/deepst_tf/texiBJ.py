from __future__ import print_function
import os
import sys
import cPickle as pickle
import time
import numpy as np
import h5py

import tensorflow as tf

from datasets.TaxiBJ import load_data
from deepst_tf import model
from deepst_tf import model_gc
import deepst_tf.nn3 as nn

flags = tf.app.flags
FLAGS = flags.FLAGS

# number of epoch at training stage
# flags.DEFINE_integer('nb_epoch', 500, '')
# # number of epoch at training (cont) stage
# flags.DEFINE_integer('nb_epoch_cont', 100, '')
# # batch size
# flags.DEFINE_integer('batch_size', 32, '')
# # number of time intervals in one day
# flags.DEFINE_integer('T', 48, '')
# # learning rate
# flags.DEFINE_float('lr', 0.0002, '')

nb_epoch = 500
nb_epoch_cont = 100
batch_size = 32
T = 48
lr = 0.0002

len_closeness = 3  # length of closeness dependent sequence
len_period = 1  # length of peroid dependent sequence
len_trend = 1  # length of trend dependent sequence
nb_flow = 2  # there are two types of flows: inflow and outflow
nb_residual_unit = 2
# divide data into two subsets: Train & Test, of which the test set is the
# last 4 weeks
days_test = 7 * 4
len_test = T * days_test
map_height, map_width = 32, 32  # grid size
path_result = 'RET'
path_model = 'MODEL'

if os.path.isdir(path_result) is False:
    os.mkdir(path_result)
if os.path.isdir(path_model) is False:
    os.mkdir(path_model)
if os.path.isdir(path_cache) is False:
    os.mkdir(path_cache)

def config(rank = 0):
    # Configs for multi-GPU
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=False)
    # Allocate only as much GPU memory based on runtime allocations.
    config.gpu_options.allow_growth = True
    print 'rank is {}'.format(rank)
    config.gpu_options.visible_device_list = str(rank % 4)
    import multiprocessing
    config.intra_op_parallelism_threads = multiprocessing.cpu_count() / 3
    config.inter_op_parallelism_threads = multiprocessing.cpu_count() / 3

    # XLA
    jit_level = tf.OptimizerOptions.ON_1
    config.graph_options.optimizer_options.global_jit_level = jit_level
    return config

def read_cache(fname):
    mmn = pickle.load(open('preprocessing.pkl', 'rb'))

    f = h5py.File(fname, 'r')
    num = int(f['num'].value)
    X_train, Y_train, X_test, Y_test = [], [], [], []
    for i in xrange(num):
        X_train.append(f['X_train_%i' % i].value)
        X_test.append(f['X_test_%i' % i].value)
    Y_train = f['Y_train'].value
    Y_test = f['Y_test'].value
    external_dim = f['external_dim'].value
    timestamp_train = f['T_train'].value
    timestamp_test = f['T_test'].value
    f.close()

    return X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test

def cache(fname, X_train, Y_train, X_test, Y_test, external_dim, timestamp_train, timestamp_test):
    h5 = h5py.File(fname, 'w')
    h5.create_dataset('num', data=len(X_train))

    for i, data in enumerate(X_train):
        h5.create_dataset('X_train_%i' % i, data=data)
    # for i, data in enumerate(Y_train):
    for i, data in enumerate(X_test):
        h5.create_dataset('X_test_%i' % i, data=data)
    h5.create_dataset('Y_train', data=Y_train)
    h5.create_dataset('Y_test', data=Y_test)
    external_dim = -1 if external_dim is None else int(external_dim)
    h5.create_dataset('external_dim', data=external_dim)
    h5.create_dataset('T_train', data=timestamp_train)
    h5.create_dataset('T_test', data=timestamp_test)
    h5.close()


def build_model(external_dim):
    c_conf = (len_closeness, nb_flow, map_height,
              map_width) if len_closeness > 0 else None
    p_conf = (len_period, nb_flow, map_height,
              map_width) if len_period > 0 else None
    t_conf = (len_trend, nb_flow, map_height,
              map_width) if len_trend > 0 else None

    model = model.Model(c_conf, p_conf, t_conf, external_dim, nb_residual_unit)
    model = model_gc.ModelGC(c_conf, p_conf, t_conf, external_dim, nb_residual_unit, False)

    train_op, loss = model.inference()
    opt = tf.train.AdamOptimizer(lr)
    global_step = tf.contrib.framework.get_or_create_global_step()
    train_op = opt.minimize(loss, global_step = global_step)
    return train_op, loss, global_step

def main():
    print("loading data...")
    ts = time.time()
    fname = os.path.join(DATAPATH, 'CACHE', 'TaxiBJ_C{}_P{}_T{}.h5'.format(
        len_closeness, len_period, len_trend))
    if os.path.exists(fname) and CACHEDATA:
        X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = read_cache(
            fname)
        print("load %s successfully" % fname)
    else:
        X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = TaxiBJ.load_data(
            T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend, len_test=len_test,
            preprocess_name='preprocessing.pkl', meta_data=True, meteorol_data=True, holiday_data=True)
        if CACHEDATA:
            cache(fname, X_train, Y_train, X_test, Y_test,
                  external_dim, timestamp_train, timestamp_test)

    print("\n days (test): ", [v[:8] for v in timestamp_test[0::T]])
    print("\nelapsed time (loading data): %.3f seconds\n" % (time.time() - ts))

    print('=' * 10)
    print("compiling model...")
    print(
        "**at the first time, it takes a few minites to compile if you use [Theano] as the backend**")

    XC_train, XP_train, XT_train, meta_train = X_train
    train_data = nn.Data(XC_train, XP_train, XT_train, meta_train, Y_train)
    XC_test, XP_test, XT_test, meta_test = X_test
    test_data = nn.Data(XC_test, XP_test, XT_test, meta_test, Y_test)
    train_op, loss, global_step = build_model(external_dim)

    train_hooks = [
        hvd.BroadcastGlobalVariablesHook(0),
        tf.train.StopAtStepHook(last_step = nb_epoch * X_train.size() / batch_size),
        tf.train.LoggingTensorHook(tensors={'step': global_step, 'loss':loss},
            every_n_iter=10)
    ]

    # ts = time.time()
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=checkpoint_dir, hooks=train_hooks, 
        config=config(),
        save_checkpoint_steps=2000) as sess:
        while not mon_sess.should_stop():
            data = train.data.next_batch(batch_size)
            _, loss_val = sess.run([train_op, loss], feed_dict=model.train_param(data))

    test_hooks = [
        tf.train.StopAtStepHook(last_step = X_test.size() / batch_size)
    ]
    sum_loss = 0
    count = 0
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=checkpoint_dir, hooks=test_hooks, 
        config=config(),
        save_checkpoint_steps=2000) as sess:
    while not mon_sess.should_stop():
        data = test_data.next_batch(batch_size)
        sum_loss += sess.run(loss, feed_dict=model.test_param(data))
        count += 1
    print('total loss={}'.format(sum_loss / count3))
if __name__ == '__main__':
    main()