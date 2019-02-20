from __future__ import print_function
import os
import sys
import cPickle as pickle
import time
import numpy as np
import h5py
import ConfigParser

# deepst
import tensorflow as tf

from deepst.datasets.TaxiBJ import load_data, load_plain_data
from deepst_tf import model as modelnn
from deepst_tf import model_gc
import deepst_tf.nn3 as nn
import deepst_tf.framework as fw

# ha
from comparation.ha import HA

# lr
from deepst_tf.lr_model import LRModel

# gbdt
from comparation import gbdt

tf.logging.set_verbosity(tf.logging.INFO)

cf = ConfigParser.ConfigParser()    
cf.read(sys.argv[1]) 

# DATAPATH = os.environ.get('DATAPATH')
CACHEDATA = True
# path_cache = os.path.join(DATAPATH, 'CACHE')

# nb_epoch = 500
# nb_epoch_cont = 100
# batch_size = 32
# T = 48
# lr = 0.0002

# len_closeness = 3  # length of closeness dependent sequence
# len_period = 1  # length of peroid dependent sequence
# len_trend = 1  # length of trend dependent sequence
# nb_flow = 2  # there are two types of flows: inflow and outflow
# nb_residual_unit = 2
# # divide data into two subsets: Train & Test, of which the test set is the
# # last 4 weeks
# days_test = 7 * 4
# len_test = T * days_test
# map_height, map_width = 32, 32  # grid size
# checkpoint_dir = '/home/fengchaodavid/ridedispatcher/DeepST/ckpt/ckpt'

#if os.path.isdir(path_cache) is False:
#    os.mkdir(path_cache)

def read_cache(fname):
    mmn = pickle.load(open('preprocessing.pkl', 'rb'))

    f = h5py.File(fname, 'r')
    num = int(f['num'].value)
    X_train, Y_train, X_eval, Y_eval, X_test, Y_test = [], [], [], [], [], []
    for i in xrange(num):
        X_train.append(f['X_train_%i' % i].value)
        X_eval.append(f['X_eval_%i' % i].value)
        X_test.append(f['X_test_%i' % i].value)

    Y_train = f['Y_train'].value
    Y_eval = f['Y_eval'].value
    Y_test = f['Y_test'].value
    external_dim = f['external_dim'].value
    timestamp_train = f['T_train'].value
    timestamp_eval = f['T_eval'].value
    timestamp_test = f['T_test'].value
    f.close()

    return X_train, Y_train, X_eval, Y_eval, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_eval, timestamp_test

def cache(fname, X_train, Y_train, X_eval, Y_eval, X_test, Y_test, external_dim, timestamp_train, timestamp_eval, timestamp_test):
    h5 = h5py.File(fname, 'w')
    h5.create_dataset('num', data=len(X_train))

    for i, data in enumerate(X_train):
        h5.create_dataset('X_train_%i' % i, data=data)
    for i, data in enumerate(X_eval):
        h5.create_dataset('X_eval_%i' % i, data=data)
    # for i, data in enumerate(Y_train):
    for i, data in enumerate(X_test):
        h5.create_dataset('X_test_%i' % i, data=data)
    h5.create_dataset('Y_train', data=Y_train)
    h5.create_dataset('Y_eval', data=Y_eval)
    h5.create_dataset('Y_test', data=Y_test)
    external_dim = -1 if external_dim is None else int(external_dim)
    h5.create_dataset('external_dim', data=external_dim)
    h5.create_dataset('T_train', data=timestamp_train)
    h5.create_dataset('T_eval', data=timestamp_eval)
    h5.create_dataset('T_test', data=timestamp_test)
    h5.close()


def build_lr_model(external_dim, nb_flow, len_closeness, 
    lr, cf):
    adj_path = cf.get('general', 'adjpath')
    adj_mat = np.load(adj_path)
    regions = adj_mat.shape[0]
    conf = (len_closeness, nb_flow, regions) if len_closeness > 0 else None
    model = LRModel(conf, external_dim)
    model.inference(lr)
    return model

def build_model(external_dim, len_closeness, len_period,
        len_trend, nb_flow, 
        nb_residual_unit, cf):

    taskname = cf.get('task', 'name')
    if taskname == 'deepst':
        map_height = 32
        map_width = 32
        c_conf = (len_closeness, nb_flow, map_height,
              map_width) if len_closeness > 0 else None
        p_conf = (len_period, nb_flow, map_height,
              map_width) if len_period > 0 else None
        t_conf = (len_trend, nb_flow, map_height,
              map_width) if len_trend > 0 else None

        model = modelnn.Model(c_conf, p_conf, t_conf, 
            external_dim, nb_residual_unit)
    else:
        adj_path = cf.get('general', 'adjpath')
        adj_mat = np.load(adj_path)
        regions = adj_mat.shape[0]
        c_conf = (len_closeness, nb_flow,
              regions) if len_closeness > 0 else None
        p_conf = (len_period, nb_flow,
              regions) if len_period > 0 else None
        t_conf = (len_trend, nb_flow, 
              regions) if len_trend > 0 else None
        model = model_gc.ModelGC(c_conf, p_conf, t_conf,
            external_dim, nb_residual_unit, adj_mat)
    #model = model_gc.ModelGC(c_conf, p_conf, t_conf, external_dim, nb_residual_unit, False)

    model.inference()
    # opt = tf.train.AdamOptimizer(lr)
    # global_step = tf.train.get_or_create_global_step()
    # grad = opt.compute_gradients(model.loss)
    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies(update_ops):
    #     train_op = opt.apply_gradients(grad, global_step = global_step)
    # return model, train_op, loss, global_step
    return model

def ha2(cf):
    T = cf.getint('deepst', 'T')
    nb_flow = cf.getint('deepst', 'nb_flow')
    days_eval = cf.getint('deepst', 'day_eval')
    days_test = cf.getint('deepst', 'day_test')
    len_eval = T * days_eval
    len_test = T * days_test
    use_meta = cf.getboolean('deepst', 'meta_data')

    data_start = cf.getint('general', 'data_start')
    data_end = cf.getint('general', 'data_end')
    datapath = cf.get('general', 'datapath')
    dataname = cf.get('general', 'dataname')
    mmnpath = cf.get('general', 'mmnpath')
    cachepath = cf.get('general', 'cachepath')
    checkpoint_dir = cf.get('general', 'ckpt')

    len_closeness = 15

    print("loading data...")
    ts = time.time()
    if os.path.isdir(cachepath) is False:
        os.mkdir(cachepath)
    fname = os.path.join(cachepath, 'C{}_P0_T0.h5'.format(
        len_closeness))
    if os.path.exists(fname) and CACHEDATA:
        X_train, Y_train, X_eval, Y_eval, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_eval, timestamp_test = read_cache(
            fname)
        print("load %s successfully" % fname)
    else:
        X_train, Y_train, X_eval, Y_eval, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_eval, timestamp_test = load_data(
            T=T, nb_flow=nb_flow, len_closeness=len_closeness, 
            len_period=0, len_trend=0, len_eval=len_eval, len_test=len_test,
            preprocess_name=mmnpath, meta_data=use_meta, 
            meteorol_data=False, holiday_data=False, 
            data_start=data_start, data_end=data_end, datafolder=datapath,
            dataname=dataname)
        if CACHEDATA:
            cache(fname, X_train, Y_train, X_eval, Y_eval, X_test, Y_test,
                  external_dim, timestamp_train, timestamp_eval, timestamp_test)

    print("\n days (test): ", [v[:8] for v in timestamp_test[0::T]])
    print("\nelapsed time (loading data): %.3f seconds\n" % (time.time() - ts))

    print('=' * 10)
    print("compiling model...")
    print("external:{}".format(external_dim))

    X_eval = X_eval[0][:,1::2,:]
    Y_eval = Y_eval[:,1,:]

    ha = HA()
    ha.eval(X_eval, Y_eval, mmn)

def ha(cf):
    T = cf.getint('deepst', 'T')
    nb_flow = cf.getint('deepst', 'nb_flow')
    day_test = cf.getint('deepst', 'day_test')
    len_test = T * day_test

    data_start = cf.getint('general', 'data_start')
    data_end = cf.getint('general', 'data_end')
    datapath = cf.get('general', 'datapath')
    dataname = cf.get('general', 'dataname')
    mmnpath = cf.get('general', 'mmnpath')
    
    print("loading data...")
    ts = time.time()
    
    X_train, X_test, mmn, timestamp_train, timestamp_test = load_plain_data(
        T=T, nb_flow=nb_flow, data_start=data_start, 
        data_end=data_end, datafolder=datapath, 
        dataname=dataname, len_test=len_test,
        preprocess_name=mmnpath)
    
    print("\n days (test): ", [v[:8] for v in timestamp_test[0::T]])
    print("\nelapsed time (loading data): %.3f seconds\n" % (time.time() - ts))

    print('=' * 10)
    print('X_train:', X_train.shape)
    print('X_test:', X_test.shape)
    print('timestamp_train:', timestamp_train.shape)
    print('timestamp_test:', timestamp_test.shape)
    regions = X_train.shape[2]
    ha = HA(regions)

    X_train = X_train[:,1,:]
    print(X_train.shape, timestamp_train.shape)
    ha.record(X_train, timestamp_train)

    sum_val = 0
    for d, t in zip(X_test, timestamp_test):
        #print(t)
        pred = ha.infer(t)
        sum_val += np.sqrt(np.sum(np.square(pred - d)))
    rmse = sum_val / len(timestamp_test)
    print('rmse={}, real_rmse={}'.format(rmse, rmse * (mmn._max - mmn._min) / 2))

def gbrt(cf):
    T = cf.getint('deepst', 'T')
    nb_flow = cf.getint('deepst', 'nb_flow')
    days_eval = cf.getint('deepst', 'day_eval')
    days_test = cf.getint('deepst', 'day_test')
    len_eval = T * days_eval
    len_test = T * days_test
    use_meta = cf.getboolean('deepst', 'meta_data')

    data_start = cf.getint('general', 'data_start')
    data_end = cf.getint('general', 'data_end')
    datapath = cf.get('general', 'datapath')
    dataname = cf.get('general', 'dataname')
    mmnpath = cf.get('general', 'mmnpath')
    cachepath = cf.get('general', 'cachepath')
    checkpoint_dir = cf.get('general', 'ckpt')

    len_closeness = 15

    print("loading data...")
    ts = time.time()
    if os.path.isdir(cachepath) is False:
        os.mkdir(cachepath)
    fname = os.path.join(cachepath, 'C{}_P0_T0.h5'.format(
        len_closeness))
    if os.path.exists(fname) and CACHEDATA:
        X_train, Y_train, X_eval, Y_eval, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_eval, timestamp_test = read_cache(
            fname)
        print("load %s successfully" % fname)
    else:
        X_train, Y_train, X_eval, Y_eval, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_eval, timestamp_test = load_data(
            T=T, nb_flow=nb_flow, len_closeness=len_closeness, 
            len_period=0, len_trend=0, len_eval=len_eval, len_test=len_test,
            preprocess_name=mmnpath, meta_data=use_meta, 
            meteorol_data=False, holiday_data=False, 
            data_start=data_start, data_end=data_end, datafolder=datapath,
            dataname=dataname)
        if CACHEDATA:
            cache(fname, X_train, Y_train, X_eval, Y_eval, X_test, Y_test,
                  external_dim, timestamp_train, timestamp_eval, timestamp_test)

    print("\n days (test): ", [v[:8] for v in timestamp_test[0::T]])
    print("\nelapsed time (loading data): %.3f seconds\n" % (time.time() - ts))

    print('=' * 10)
    print("compiling model...")
    print("external:{}".format(external_dim))

    if external_dim != None and external_dim > 0:
        XC_train, meta_train = X_train
        train_data = gbdt.GBDTData([XC_train], [meta_train], Y_train)
        XC_eval, meta_eval = X_eval
        eval_data = gbdt.GBDTData([XC_eval], [meta_eval], Y_eval)
        XC_test, meta_test = X_test
        test_data = gbdt.GBDTData([XC_test], [meta_train], Y_test)
    else:
        XC_train = X_train[0]
        XC_train = np.transpose(XC_train, [0, 2, 1])
        train_shape = XC_train.shape
        XC_train = np.reshape(XC_train, [-1, train_shape[2]])

        Y_train = Y_train[:,1,:]
        Y_train = np.reshape(Y_train, [-1])
        train_data = gbdt.GBDTData([XC_train], [], Y_train)

        XC_eval = X_eval[0]
        XC_eval = np.transpose(XC_eval, [0, 2, 1])
        eval_shape = XC_eval.shape
        XC_eval = np.reshape(XC_eval, [-1, eval_shape[2]])

        Y_eval = Y_eval[:,1,:]
        Y_eval = np.reshape(Y_eval, [-1])
        eval_data = gbdt.GBDTData([XC_eval], [], Y_eval)

        XC_test = X_test[0]
        XC_test = np.transpose(XC_test, [0, 2, 1])
        test_shape = XC_test.shape
        XC_test = np.reshape(XC_test, [-1, test_shape[2]])

        Y_test = Y_test[:,1,:]
        Y_test = np.reshape(Y_test, [-1])
        test_data = gbdt.GBDTData([XC_test], [], Y_test)

    X_train, category_idx = train_data.getAllFeats()
    y_train = train_data.getAllLabel()
    X_eval, category_idx = eval_data.getAllFeats()
    y_eval = eval_data.getAllLabel()
    X_test, category_idx = test_data.getAllFeats()
    y_test = test_data.getAllLabel()

    nb_epoch = cf.getint('train', 'nb_epoch')
    lr = cf.getfloat('train', 'lr')

    ckpt_path = '{}/model.txt'.format(checkpoint_dir)
    gbdt.train(X_train, y_train, X_eval, y_eval, ckpt_path, category_idx, mmn)
    Y_pred = gbdt.inference(ckpt_path, X_test)

    dump_path = cf.get('inference', 'predict_path')
    real_Y_pred = mmn.inverse_transform(Y_pred)
    np.savetxt(dump_path, real_Y_pred, delimiter=',')

def linearregression(cf):
    T = cf.getint('deepst', 'T')
    nb_flow = cf.getint('deepst', 'nb_flow')
    days_eval = cf.getint('deepst', 'day_eval')
    days_test = cf.getint('deepst', 'day_test')
    len_eval = T * days_eval
    len_test = T * days_test
    use_meta = cf.getboolean('deepst', 'meta_data')

    data_start = cf.getint('general', 'data_start')
    data_end = cf.getint('general', 'data_end')
    datapath = cf.get('general', 'datapath')
    dataname = cf.get('general', 'dataname')
    mmnpath = cf.get('general', 'mmnpath')
    cachepath = cf.get('general', 'cachepath')
    checkpoint_dir = cf.get('general', 'ckpt')

    len_closeness = 15

    print("loading data...")
    ts = time.time()
    if os.path.isdir(cachepath) is False:
        os.mkdir(cachepath)
    fname = os.path.join(cachepath, 'C{}_P0_T0.h5'.format(
        len_closeness))
    if os.path.exists(fname) and CACHEDATA:
        X_train, Y_train, X_eval, Y_eval, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_eval, timestamp_test = read_cache(
            fname)
        print("load %s successfully" % fname)
    else:
        X_train, Y_train, X_eval, Y_eval, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_eval, timestamp_test = load_data(
            T=T, nb_flow=nb_flow, len_closeness=len_closeness, 
            len_period=0, len_trend=0, len_eval=len_eval, len_test=len_test,
            preprocess_name=mmnpath, meta_data=use_meta, 
            meteorol_data=False, holiday_data=False, 
            data_start=data_start, data_end=data_end, datafolder=datapath,
            dataname=dataname)
        if CACHEDATA:
            cache(fname, X_train, Y_train, X_eval, Y_eval, X_test, Y_test,
                  external_dim, timestamp_train, timestamp_eval, timestamp_test)

    print("\n days (test): ", [v[:8] for v in timestamp_test[0::T]])
    print("\nelapsed time (loading data): %.3f seconds\n" % (time.time() - ts))

    print('=' * 10)
    print("compiling model...")
    print("external:{}".format(external_dim))

    if external_dim != None and external_dim > 0:
        XC_train, XP_train, XT_train, meta_train = X_train
        train_data = nn.Data(XC_train, XP_train, XT_train, meta_train, Y_train)
        XC_eval, XP_eval, XT_eval, meta_eval = X_eval
        eval_data = nn.Data(XC_eval, XP_eval, XT_eval, meta_eval, Y_eval)
        XC_test, XP_test, XT_test, meta_test = X_test
        test_data = nn.Data(XC_test, XP_test, XT_test, meta_test, Y_test)
    else:
        XC_train = X_train[0]
        #print(len(X_train), XC_train.shape, Y_train.shape)
        train_data = nn.Data(XC_train, Y_train)
        XC_eval = X_eval[0]
        eval_data = nn.Data(XC_eval, Y_eval)
        XC_test = X_test[0]
        test_data = nn.Data(XC_test, Y_test)

    nb_epoch = cf.getint('train', 'nb_epoch')
    nb_epoch_cont = cf.getint('train', 'nb_epoch_cont')
    batch_size = cf.getint('train', 'batch_size')
    lr = cf.getfloat('train', 'lr')

    model = build_lr_model(external_dim, nb_flow, len_closeness, lr, cf)

    early_stop = fw.EarlyStop(need_max=False, max_epoch=10)
    with fw.tf_session(checkpoint_dir) as framework:
        for i in range(nb_epoch):
            avg_loss = framework.train(train_data, model, batch_size)
            avg_metric = framework.eval(eval_data, model, batch_size)
            print('epoch {}, loss={}, metric={}, real_metric={}'.format(
                i, avg_loss, avg_metric, avg_metric * (mmn._max - mmn._min) / 2))
            need_save, need_stop = early_stop.should_stop(avg_metric)
            if need_save:
                framework.save(i)
            if need_stop:
                print('eary stop...')
                break
            
        for i in range(nb_epoch, nb_epoch + nb_epoch_cont):
            avg_loss = framework.train(train_data, model, batch_size)
            avg_metric = framework.eval(eval_data, model, batch_size)
            print('epoch {}, loss={}, metric={}, real_metric={}'.format(
                i, avg_loss, avg_metric, avg_metric * (mmn._max - mmn._min) / 2))
            need_save, need_stop = early_stop.should_stop(avg_metric)
            if need_save:
                framework.save(i)

        pred_res = framework.inference(test_data, model, batch_size)
        real_pred_res = np.array(mmn.inverse_transform(pred_res), dtype=np.int32)
        np.savetxt(inference_path, real_pred_res, delimiter=',')
    print('best metric={}, real={}'.format(early_stop.pre_record, early_stop.pre_record * (mmn._max - mmn._min) / 2))

def deepst(cf):
    T = cf.getint('deepst', 'T')
    nb_flow = cf.getint('deepst', 'nb_flow')
    days_eval = cf.getint('deepst', 'day_eval')
    days_test = cf.getint('deepst', 'day_test')
    len_test = T * days_test
    len_eval = T * days_eval
    use_meta = cf.getboolean('deepst', 'meta_data')
    use_meteorol = cf.getboolean('deepst', 'meteorol_data')
    use_holiday = cf.getboolean('deepst', 'holiday_data')
    len_closeness = cf.getint('deepst', 'len_closeness')
    len_period = cf.getint('deepst', 'len_period')
    len_trend = cf.getint('deepst', 'len_trend')
    nb_residual_unit = cf.getint('deepst', 'nb_residual_unit')

    data_start = cf.getint('general', 'data_start')
    data_end = cf.getint('general', 'data_end')
    datapath = cf.get('general', 'datapath')
    dataname = cf.get('general', 'dataname')
    mmnpath = cf.get('general', 'mmnpath')
    cachepath = cf.get('general', 'cachepath')
    checkpoint_dir = cf.get('general', 'ckpt')
    inference_path = cf.get('inference', 'inference_path')
    
    print("loading data...")
    ts = time.time()
    if os.path.isdir(cachepath) is False:
        os.mkdir(cachepath)
    fname = os.path.join(cachepath, 'C{}_P{}_T{}.h5'.format(
        len_closeness, len_period, len_trend))
    if os.path.exists(fname) and CACHEDATA:
        X_train, Y_train, X_eval, Y_eval, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_eval, timestamp_test = read_cache(
            fname)
        print("load %s successfully" % fname)
    else:
        X_train, Y_train, X_eval, Y_eval, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_eval, timestamp_test = load_data(
            T=T, nb_flow=nb_flow, len_closeness=len_closeness, 
            len_period=len_period, len_trend=len_trend, len_eval=len_eval, len_test=len_test,
            preprocess_name=mmnpath, meta_data=use_meta, 
            meteorol_data=use_meteorol, holiday_data=use_holiday, 
            data_start=data_start, data_end=data_end, datafolder=datapath,
            dataname=dataname)
        if CACHEDATA:
            cache(fname, X_train, Y_train, X_eval, Y_eval, X_test, Y_test,
                  external_dim, timestamp_train, timestamp_eval, timestamp_test)

    print("\n days (test): ", [v[:8] for v in timestamp_test[0::T]])
    print("\nelapsed time (loading data): %.3f seconds\n" % (time.time() - ts))

    print('=' * 10)
    print("compiling model...")
    print("external:{}".format(external_dim))

    if external_dim != None and external_dim > 0:
        XC_train, XP_train, XT_train, meta_train = X_train
        train_data = nn.Data(XC_train, XP_train, XT_train, meta_train, Y_train)
        XC_eval, XP_eval, XT_eval, meta_eval = X_eval
        eval_data = nn.Data(XC_eval, XP_eval, XT_eval, meta_eval, Y_eval)
        XC_test, XP_test, XT_test, meta_test = X_test
        test_data = nn.Data(XC_test, XP_test, XT_test, meta_test, Y_test)
    else:
        XC_train, XP_train, XT_train = X_train
        train_data = nn.Data(XC_train, XP_train, XT_train, Y_train)
        XC_eval, XP_eval, XT_eval = X_eval
        eval_data = nn.Data(XC_eval, XP_eval, XT_eval, Y_eval)
        XC_test, XP_test, XT_test= X_test
        test_data = nn.Data(XC_test, XP_test, XT_test, Y_test)

    nb_epoch = cf.getint('train', 'nb_epoch')
    nb_epoch_cont = cf.getint('train', 'nb_epoch_cont')
    batch_size = cf.getint('train', 'batch_size')
    lr = cf.getfloat('train', 'lr')
    lr_cont = cf.getfloat('train', 'lr_cont')

    model = build_model(external_dim, len_closeness,
        len_period, len_trend, nb_flow, 
        nb_residual_unit, cf)

    early_stop = fw.EarlyStop(need_max=False, max_epoch=10)
    with fw.tf_session(checkpoint_dir) as framework:
        for i in range(nb_epoch):
            avg_loss = framework.train(train_data, model, batch_size, lr)
            avg_metric = framework.eval(eval_data, model, batch_size)
            print('epoch {}, loss={}, metric={}, real_metric={}'.format(
                i, avg_loss, avg_metric, avg_metric * (mmn._max - mmn._min) / 2))
            need_save, need_stop = early_stop.should_stop(avg_metric)
            if need_save:
                framework.save(i)
            if need_stop:
                print('eary stop...')
                break

        for i in range(nb_epoch, nb_epoch + nb_epoch_cont):
            avg_loss = framework.train(train_data, model, batch_size, lr_cont)
            avg_metric = framework.eval(eval_data, model, batch_size)
            print('epoch {}, loss={}, metric={}, real_metric={}'.format(
                i, avg_loss, avg_metric, avg_metric * (mmn._max - mmn._min) / 2))
            need_save, need_stop = early_stop.should_stop(avg_metric)
            if need_save:
                framework.save(i)

        pred_res, gt_res = framework.inference(test_data, model, batch_size)
        real_pred_res = np.array(mmn.inverse_transform(pred_res), dtype=np.int32)
        real_pred_res = real_pred_res[:,1,:]
        real_pred_res = real_pred_res.astype(np.int)
        real_gt_res = np.array(mmn.inverse_transform(gt_res), dtype=np.int32)
        real_gt_res = real_gt_res[:,1,:]
        print(real_pred_res)
        np.savetxt(inference_path, real_pred_res, fmt='%d', delimiter=',')
        np.savetxt(inference_path + '_real', real_gt_res, fmt='%d', delimiter=',')

    print('best metric={}, real={}'.format(early_stop.pre_record, early_stop.pre_record * (mmn._max - mmn._min) / 2))

if __name__ == '__main__':
    cf = ConfigParser.ConfigParser()    
    cf.read(sys.argv[1])
    taskname = cf.get('task', 'name')
    if taskname == 'deepst' or taskname == 'deepst_gc':
        deepst(cf)
    elif taskname == 'ha':
        ha2(cf)
    elif taskname == 'lr':
        linearregression(cf)
    elif taskname == 'gbrt':
        gbrt(cf)
