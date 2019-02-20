import tensorflow as tf
import nn3 as nn

class LRModel(object):
    def __init__(self, conf, external_dim):
        self.conf = conf
        self.external_dim = external_dim
        self.gen_placeholder()

    def gen_placeholder(self):
        len_seq, nb_flow, regions = self.conf 
        self.data = tf.placeholder(tf.float32, [None, len_seq * nb_flow, regions])
        if self.external_dim != None and self.external_dim > 0:
            self.external_input = tf.placeholder(tf.float32, [None, self.external_dim])
        self.target = tf.placeholder(tf.float32, [None, nb_flow, regions])

    def train_param(self, data):
        return self.param(data)

    def test_param(self, data):
        return self.param(data)

    def param(self, data):
        if self.external_dim != None and self.external_dim > 0:
            batch_data, batch_meta, batch_Y = data
            return {
                self.data : batch_data,
                self.external_input : batch_meta,
                self.target : batch_Y
            }
        else:
            batch_data, batch_Y = data
            return {
                self.data : batch_data,
                self.target : batch_Y
            }

    def inference(self, lr):
        builder = nn.NNBuilder(True, data_format='NCHW')
        len_seq, nb_flow, regions = self.conf 
        x = tf.transpose(self.data, [0, 2, 1])
        x = tf.reshape(x, [-1, len_seq * nb_flow])
        output = builder.fc_layer(x, 'fc1', nb_flow)
        output = tf.reshape(output, [-1, regions, nb_flow])
        output = tf.transpose(output, [0, 2, 1])
        if self.external_dim != None and self.external_dim > 0:
            embedding = builder.fc_layer(self.external_input, 'external', 10)
            embedding = builder.activation_func(embedding)
            h1 = builder.fc_layer(embedding, 'fc', nb_flow * regions)
            h1 = tf.reshape(h1, [-1, nb_flow, regions])
            output = tf.add(output, h1)
        output = builder._bias(output)

        self.pred = tf.nn.tanh(output)
        self.loss = nn.mse(self.target, self.pred)
        self.metric = nn.rmse(self.target, self.pred)

        opt = tf.train.AdamOptimizer(lr)
        self.global_step = tf.train.get_or_create_global_step()
        grad = opt.compute_gradients(self.loss)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = opt.apply_gradients(grad, global_step = self.global_step)
