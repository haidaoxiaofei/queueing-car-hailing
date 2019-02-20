import os
from tqdm import tqdm
import tensorflow as tf
from contextlib import contextmanager
import numpy as np

@contextmanager
def tf_session(ckpt_dir):
    framework = Framework(ckpt_dir)
    yield(framework)
    framework.close()

class EarlyStop(object):
    def __init__(self, need_max=True, max_epoch=3):
        self.need_max = need_max
        self.max_epoch = max_epoch
        if self.need_max:
            self.pre_record = -1
        else:
            self.pre_record = 1000000
        self.last_epoch = 0

    def should_stop(self, metric):
        judge_save = False
        if self.need_max:
            if metric > self.pre_record:
                self.pre_record = metric
                self.last_epoch = 1
                judge_save = True
            else:
                self.last_epoch += 1
        else:
            if metric < self.pre_record:
                self.pre_record = metric
                self.last_epoch = 1
                judge_save = True
            else:
                self.last_epoch += 1
        judge_stop = True if self.last_epoch > self.max_epoch else False
        return judge_save, judge_stop



class Framework(object):

    def config(self, rank = 0):
        # Configs for multi-GPU
        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False)
        # Allocate only as much GPU memory based on runtime allocations.
        config.gpu_options.allow_growth = True
        print('rank is {}'.format(rank))
        config.gpu_options.visible_device_list = str(rank % 4)
        import multiprocessing
        config.intra_op_parallelism_threads = multiprocessing.cpu_count() / 3
        config.inter_op_parallelism_threads = multiprocessing.cpu_count() / 3

        # XLA
        jit_level = tf.OptimizerOptions.ON_1
        config.graph_options.optimizer_options.global_jit_level = jit_level
        return config

    def __init__(self, ckpt_dir):
        self.session = tf.Session(config=self.config())
        self.ckpt_dir = ckpt_dir
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
        self.session.run(tf.global_variables_initializer())
        ckpt_dir = os.path.dirname(ckpt_dir)
        latest_model = tf.train.latest_checkpoint(ckpt_dir)
        print 'find ckpt_path={}'.format(ckpt_dir)
        if latest_model is not None:
            print 'restore from {}'.format(latest_model)
            self.saver.restore(self.session, latest_model)
        else:
            print 'train from scratch'

    def run(self, target, feed_dict_val):
        return self.session.run(target, feed_dict=feed_dict_val)

    def train(self, train_data, model, batch_size, lr):
        val = 0
        count = 0
        with tqdm(train_data.epoch(batch_size)) as t:
            for data in t:
                _, loss_val, step = self.session.run([model.train_op, model.loss,
                    model.global_step], feed_dict=model.train_param(data, lr))
                val += loss_val
                count += 1

                t.set_postfix(loss=val / count)
        return float(val) / count

    def eval(self, test_data, model, batch_size):
        val = 0
        count = 0
        with tqdm(test_data.epoch(batch_size)) as t:
            for data in t:
                val += self.session.run(model.metric, feed_dict=model.test_param(data))
                count += 1
                t.set_postfix(loss=val / count)
        return float(val) / count

    def inference(self, test_data, model, batch_size):
        pred_res = []
        real_res = []
        for data in tqdm(test_data.epoch(batch_size)):
            pred_val, real_val = self.session.run([model.pred, model.target], feed_dict=model.test_param(data))
            pred_res.append(pred_val)
            real_res.append(real_val)
        pred_res = np.vstack(pred_res)
        real_res = np.vstack(real_res)
        return pred_res, real_res

    def save(self, step):
        print(self.ckpt_dir)
        self.saver.save(self.session, self.ckpt_dir, global_step=step)

    def close(self):
        self.session.close()
