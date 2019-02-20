import tensorflow as tf
import numpy as np
from collections import defaultdict
from tensorflow.python.ops import init_ops
import pdb

def mse(target, pred):
    return tf.reduce_mean(tf.squared_difference(target, pred))

def rmse(target, pred):
    return tf.sqrt(mse(target, pred))
    #return tf.nn.l2_loss(target - pred)

def cross_entropy_loss(target, pred):
    return tf.losses.softmax_cross_entropy(target, pred)

def float32_variable_storage_getter(getter, name, shape=None, dtype=None,
                                    initializer=None, regularizer=None,
                                    trainable=True,
                                    *args, **kwargs):
    """Custom variable getter that forces trainable variables to be stored in
    float32 precision and then casts them to the training precision.
    """
    storage_dtype = tf.float32 if trainable else dtype
    variable = getter(name, shape, dtype=storage_dtype,
                      initializer=initializer, regularizer=regularizer,
                      trainable=trainable,
                      *args, **kwargs)
    if trainable and dtype != tf.float32:
        variable = tf.cast(variable, dtype)
    return variable

def gradients_with_loss_scaling(loss, variables, loss_scale=128):
    """Gradient calculation with loss scaling to improve numerical stability
    when training with float16.
    """
    return [grad / loss_scale
            for grad in tf.gradients(loss * loss_scale, variables)]

class DummyScope(object):
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class NNBuilder(object):
    def __init__(self, is_training, dtype=tf.float32,
        data_format='NHWC',
        activation='ReLU', use_batch_norm=True,
        batch_norm_config = {'decay':   0.9,
                              'epsilon': 1e-4,
                              'scale':   True,
                              'zero_debias_moving_mean': False},
        use_xla=False):
        self.dtype             = dtype
        self.data_format       = data_format
        self.activation_func   = tf.nn.relu
        self.is_training       = is_training
        self.use_batch_norm    = use_batch_norm
        self.batch_norm_config = batch_norm_config
        self._layer_counts     = defaultdict(lambda: 0)
        if use_xla:
            self.jit_scope = tf.contrib.compiler.jit.experimental_jit_scope
        else:
            self.jit_scope = DummyScope

    def _get_variable(self, name, shape, dtype=None,
            initializer=None, seed=None):
        if dtype is None:
            dtype = self.dtype
        if initializer is None:
            initializer = init_ops.glorot_uniform_initializer(seed=seed)
        elif (isinstance(initializer, float) or
              isinstance(initializer, int)):
            initializer = tf.constant_initializer(float(initializer))
        return tf.get_variable(name, shape, dtype, initializer)

    def _bias(self, input_layer):
        shape_list = input_layer.get_shape().as_list()
        if len(shape_list) == 4:
            if self.data_format == 'NCHW':
                num_outputs = shape_list[1]
            else:
                num_outputs = shape_list[3]
        else:
            num_outputs = shape_list[-1]

        biases = self._get_variable('bias', [num_outputs], input_layer.dtype,
                                    initializer=0)
        if len(input_layer.get_shape()) == 4:
            return tf.nn.bias_add(input_layer, biases,
                                  data_format=self.data_format)
        else:
            return input_layer + biases

    def _batch_norm(self, input_layer, name):
        with tf.variable_scope(name) as scope:
            return tf.contrib.layers.batch_norm(input_layer,
                is_training=self.is_training,
                scope=scope,
                data_format='NHWC',
                fused=True,
                **self.batch_norm_config)

    def _bias_or_batch_norm(self, input_layer, scope, use_batch_norm):
        if use_batch_norm is None:
            use_batch_norm = self.use_batch_norm
        if use_batch_norm:
            return self._batch_norm(input_layer, scope)
        else:
            return self._bias(input_layer)

    def conv_layer(self, bottom, name, filter_size, out_channel, stride=2):
        with tf.variable_scope(name) as scope:
            shape = bottom.get_shape().as_list()
            if self.data_format == 'NCHW':
                in_channel = shape[1]
                stride_size = [1, 1, stride, stride]
            else:
                in_channel = shape[-1]
                stride_size = [1, stride, stride, 1]

            filt = self._get_variable('W', shape=[filter_size, filter_size, in_channel, out_channel])
            # filt = tf.get_variable('W', shape=[filter_size, filter_size, in_channel, out_channel], 
            #                        initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))

            conv = tf.nn.conv2d(bottom, filt, stride_size, padding='SAME', data_format=self.data_format)

            # conv_biases = tf.get_variable('bias', shape=[out_channel], initializer=tf.constant_initializer(bias))
            # bias = tf.nn.bias_add(conv, conv_biases, data_format=data_format)
        return conv

    def as_fc_input(self, bottom, name):
        with tf.variable_scope(name) as scope:
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                 dim *= d
            x = tf.reshape(bottom, [-1, dim])
        return x

    def fc_layer(self, bottom, name, n_fc, bias=0.0):
        with tf.variable_scope(name) as scope:
            in_shape = bottom.get_shape().as_list()[-1]
            weights = self._get_variable('W', shape=[in_shape, n_fc])
            fc = tf.matmul(bottom, weights)
        return fc

    # currently we only support dim=3
    def graph_conv(self, bottom, support, name, n_fc, is_sparse):
        with tf.variable_scope(name) as scope:
            in_shape = bottom.get_shape().as_list()
            in_len = in_shape[-1]
            #if len(in_shape) > 2:
            weights = self._get_variable('W', shape=[in_len, n_fc])
            if is_sparse:
                pre_sup = tf.sparse_tensor_dense_matmul(bottom, weights)
                gconv = tf.sparse_tensor_dense_matmul(support, pre_sup)
            else:
                bottom = tf.reshape(bottom, [-1, in_len])
                pre_sup = tf.matmul(bottom, weights)
                pre_sup = tf.reshape(pre_sup, [-1, in_shape[1], n_fc])
                pre_sup = tf.transpose(pre_sup, [1, 0, 2] )
                pre_sup = tf.reshape(pre_sup, [in_shape[1], -1])
                gconv = tf.matmul(support, pre_sup)
                gconv = tf.reshape(gconv, [in_shape[1], -1, n_fc])
                gconv = tf.transpose(gconv, [1, 0, 2])
                
            #if len(in_shape) > 2:
            #gconv = tf.reshape(gconv, [-1] + in_shape[1:-1] + [n_fc])
        return gconv

    def graph_conv_block(self, x, support, n_fc, is_sparse):
        filter_in = x.get_shape()[-1]
        shortcut = x
        with tf.variable_scope('A') as scope:
            bn1 = self.activation_func(self._batch_norm(x,'bn1'))
            x = self.graph_conv(bn1, support, 'conv1', n_fc, is_sparse)

        with tf.variable_scope('B') as scope:
            bn2 = self.activation_func(self._batch_norm(x,'bn2'))
            x = self.graph_conv(bn2, support, 'conv2', n_fc, is_sparse)

        return shortcut + x

    def block(self, x, filter_out, stride):
        filter_in = x.get_shape()[-1]

        shortcut = x
        with tf.variable_scope('A') as scope:
            bn1 = self.activation_func(self._batch_norm(x, 'bn1'))
            x = self.conv_layer(bn1, 'conv1', 3, filter_out, stride)    

        with tf.variable_scope('B') as scope:
            bn2 = self.activation_func(self._batch_norm(x, 'bn2'))
            x = self.conv_layer(bn2, 'conv2', 3, filter_out, 1)

        with tf.variable_scope('shortcut'):
            if filter_out != filter_in or stride != 1:
                shortcut = self.conv_layer(shortcut, 'conv', 3, filter_out, stride)
        return shortcut + x

    def stack(self, name, x, block_num, filter_out, start_stride=1, 
        train_phase = True, data_format='NHWC'):
        for n in range(block_num):
            s = start_stride if n == 0 else 1
            with tf.variable_scope(name + 'block{}'.format(n)):
                x = self.block(x, filter_out, s)
        return x

    def resnet20_tiny(self, x):
        with tf.variable_scope('scale1'):
            conv1 = self.conv_layer(x, "conv1", 9, 16, 4)
            bn1 = self._batch_norm(conv1, 'bn1')
            relu1 = self.activation_func(bn1, 'relu1')
            conv2 = self.conv_layer(relu1, 'conv2', 7, 32, 2)
            bn2 = self._batch_norm(conv2, 'bn2')
            relu2 = self.activation_func(bn2, 'relu2')
            relu3 = self.conv_layer(relu2, 'conv3', 5, 64, 2)

        with tf.variable_scope('scale2'):
            s1 = self.stack('stack1', relu3, 3, 64, 1, self.is_training)

        with tf.variable_scope('scale3'):
            s2 = self.stack('stack2', s1, 3, 128, 2, self.is_training)

        with tf.variable_scope('scale4'):
            s3 = self.stack('stack3', s2, 3, 256, 2, self.is_training)

        if self.data_format == 'NHWC':
            x = tf.reduce_mean(s3, reduction_indices=[1,2], name='avg_pool')
        else:
            x = tf.reduce_mean(s3, reduction_indices=[2,3], name='avg_pool')
        return x

    def resnet20(self, x):
        with tf.variable_scope('scale1'):
            conv1 = self.conv_layer(x, "conv1", 9, 32, 4)
            bn1 = self._batch_norm(conv1, 'bn1')
            relu1 = self.activation_func(bn1, 'relu1')
            conv2 = self.conv_layer(relu1, "conv2", 7, 64, 2)
            bn2 = self._batch_norm(conv2, 'bn2')
            relu2 = self.activation_func(bn2, 'relu2')
            relu3 = self.conv_layer(relu2, "conv3", 5, 128, 2)

        with tf.variable_scope('scale2'):
            s1 = self.stack('stack1', relu3, 3, 128, 1, self.is_training)

        with tf.variable_scope('scale3'):
            s2 = self.stack('stack2', s1, 3, 256, 2, self.is_training)

        with tf.variable_scope('scale4'):
            s4 = self.stack('stack3', s2, 3, 512, 2, self.is_training)
        # TODO: need recover later
        # with tf.variable_scope('scale5'):
        #     s4 = stack('stack4', s3, 3, 1024, 2, is_training)

        if self.data_format == 'NHWC':
            x = tf.reduce_mean(s4, reduction_indices=[1, 2], name='avg_pool')
        else:
            x = tf.reduce_mean(s4, reduction_indices=[2, 3], name='avg_pool')
        return x

class Data(object):
    def __init__(self, *args):
        assert len(args) > 0
        for i in range(1, len(args)):
            assert args[i].shape[0] == args[0].shape[0]
        self.dataset = list(args)
        self.index = np.random.permutation(len(self.dataset[0]))
        self.start = 0
        self.data_size = len(self.index)

    def next_batch(self, count):
        if self.start + count >= len(self.dataset):
            self.index = np.random.permutation(self.index)
            self.start = 0
        ret = []
        for item in self.dataset:
            ret.append(item[self.start: self.start + count])
        self.start += count
        return ret

    def epoch(self, count):
        self.start = 0
        self.index = np.random.permutation(self.index)
        while self.start < self.data_size:
            real_count = count if self.start + count < self.data_size else self.data_size - self.start
            ret = []
            for item in self.dataset:
                ret.append([item[idx] for idx in self.index[self.start: self.start + real_count]])
            self.start += real_count
            yield ret

    def size(self):
        return self.data_size

if __name__ == '__main__':
    d = Data(range(100))
    for i in d.epoch(10):
        print i
