import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import nn3 as nn

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    rowsum = np.sum(adj, axis=1)
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    # d_mat_inv_sqrt.transpose().np.matmul()
    # adj /= np.expand_dims(d_inv_sqrt,1)
    # rowsum = np.array(adj.sum(1))
    # d_inv_sqrt = .flatten()
    # d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    # d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    # return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

'''
as our adj matrix is not so big, so we just use dense matrix
to calc, and store it into the tf_const for calc 
'''
def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + np.eye(adj.shape[0]))
    return adj_normalized
    # return sparse_to_tuple(adj_normalized)

class ModelGC(object):
    def __init__(self, c_conf, p_conf, t_conf, 
        external_dim, nb_residual_unit, adj, is_sparse=False):
        self.c_conf = c_conf
        self.p_conf = p_conf
        self.t_conf = t_conf
        self.external_dim = external_dim
        self.nb_residual_unit = nb_residual_unit
        self.gen_placeholder()
        self.is_sparse = is_sparse

        with tf.device('/gpu:0'):
            self.support = tf.constant(preprocess_adj(adj), dtype=tf.float32)
        self.out_channel = 2

    def gen_placeholder(self):
        len_seq, nb_flow, regions = self.c_conf
        self.c_data = tf.placeholder(tf.float32, [None, len_seq * nb_flow, regions])
        len_seq, nb_flow, regions = self.p_conf
        self.p_data = tf.placeholder(tf.float32, [None, len_seq * nb_flow, regions])
        len_seq, nb_flow, regions = self.t_conf
        self.t_data = tf.placeholder(tf.float32, [None, len_seq * nb_flow, regions])
        # self.support = tf.constant(dtype=tf.float32)
        if self.external_dim != None and self.external_dim > 0:
            self.external_input = tf.placeholder(tf.float32, [None, self.external_dim])
        self.target = tf.placeholder(tf.float32, [None, nb_flow, regions])
        self.train = tf.placeholder(tf.bool, [])
        self.lr = tf.placeholder(tf.float, [])

    def train_param(self, data, lr):
        ret = self.param(data, lr)
        ret[self.train] = True
        return ret

    def test_param(self, data, lr):
        ret = self.param(data, lr)
        ret[self.train] = False
        return ret

    def param(self, data, lr):
        if self.external_dim != None and self.external_dim > 0:
            batch_XC, batch_XP, batch_XT, batch_meta, batch_Y = data
            return {
                self.c_data : batch_XC,
                self.p_data : batch_XP,
                self.t_data : batch_XT,
                self.external_input : batch_meta,
                self.target : batch_Y,
                self.lr : lr
            }
        else:
            batch_XC, batch_XP, batch_XT, batch_Y = data
            return {
                self.c_data : batch_XC,
                self.p_data : batch_XP,
                self.t_data : batch_XT,
                self.target : batch_Y,
                self.lr : lr
            }

    def inference(self):
        builder = nn.NNBuilder(self.train, data_format='NCHW')
        outputs = []
        for xin, name, conf in zip([self.c_data, self.p_data, self.t_data],
            ['closeness', 'period', 'trend'], [self.c_conf, self.p_conf, self.t_conf]):
            len_seq, nb_flow, regions = conf
            with tf.variable_scope(name):
                xin = tf.transpose(xin, [0, 2, 1])
                conv1 = builder.graph_conv(xin, self.support, 'conv1', 64, self.is_sparse)
                for i in range(self.nb_residual_unit):
                    with tf.variable_scope('resnet_{}'.format(i)):
                        conv1 = builder.graph_conv_block(conv1, self.support, 64, self.is_sparse)
                relu1 = builder.activation_func(conv1)
                conv2 = builder.graph_conv(relu1, self.support, 'conv2', 2, self.is_sparse)
                conv2 = builder._bias(conv2)
                outputs.append(conv2)
        new_outputs = []
        for i,output in enumerate(outputs):
            W = builder._get_variable('ilayer_{}'.format(i), [1, regions, nb_flow])
            new_output = tf.multiply(output, W)
            new_outputs.append(new_output)
        main_output = tf.add_n(new_outputs)
        main_output = builder._bias(main_output)
        # reshape as [batch flow node]
        main_output = tf.transpose(main_output, [0, 2, 1])

        if self.external_dim != None and self.external_dim > 0:
            embedding = builder.fc_layer(self.external_input, 'external', 10)
            embedding = builder.activation_func(embedding)
            h1 = builder.fc_layer(embedding, 'fc', nb_flow * regions)
            h1 = builder._bias(h1)
            activation = builder.activation_func(h1)
            external_output = tf.reshape(activation, [-1, nb_flow, regions])
            main_output = tf.add(main_output, external_output, name='add')

        self.pred = tf.nn.tanh(main_output)
        self.loss = nn.mse(self.target, self.pred)
        self.metric = nn.rmse(self.target, self.pred)

        opt = tf.train.AdamOptimizer(self.lr)
        self.global_step = tf.train.get_or_create_global_step()
        grad =  opt.compute_gradients(self.loss)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = opt.apply_gradients(grad, global_step = self.global_step)
