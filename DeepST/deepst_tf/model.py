import tensorflow as tf
import nn3 as nn

class Model(object):
    def __init__(self, c_conf, p_conf, t_conf, external_dim, nb_residual_unit):
        self.c_conf = c_conf
        self.p_conf = p_conf
        self.t_conf = t_conf
        self.external_dim = external_dim
        self.nb_residual_unit = nb_residual_unit
        self.gen_placeholder()
        self.out_channel = 2

    def gen_placeholder(self):
        len_seq, nb_flow, map_height, map_width = self.c_conf
        self.c_data = tf.placeholder(tf.float32, [None, len_seq * nb_flow, map_height, map_width])
        len_seq, nb_flow, map_height, map_width = self.p_conf
        self.p_data = tf.placeholder(tf.float32, [None, len_seq * nb_flow, map_height, map_width])
        len_seq, nb_flow, map_height, map_width = self.t_conf
        self.t_data = tf.placeholder(tf.float32, [None, len_seq * nb_flow, map_height, map_width])
        if self.external_dim != None and self.external_dim > 0:
            self.external_input = tf.placeholder(tf.float32, [None, self.external_dim])
        self.target = tf.placeholder(tf.float32, [None, nb_flow, map_height, map_width])
        self.train = tf.placeholder(tf.bool, [])

    def train_param(self, data):
        ret = self.param(data)
        ret[self.train] = True
        return ret
        

    def test_param(self, data):
        ret = self.param(data)
        ret[self.train] = False
        return ret

    def param(self, data):
        if self.external_dim != None and self.external_dim > 0:
            batch_XC, batch_XP, batch_XT, batch_meta, batch_Y = data
            return {
                self.c_data : batch_XC,
                self.p_data : batch_XP,
                self.t_data : batch_XT,
                self.external_input : batch_meta,
                self.target : batch_Y
            }
        else:
            batch_XC, batch_XP, batch_XT, batch_Y = data
            return {
                self.c_data : batch_XC,
                self.p_data : batch_XP,
                self.t_data : batch_XT,
                self.target : batch_Y
            }
           

    def inference(self, lr):
        builder = nn.NNBuilder(self.train, data_format='NCHW')
        outputs = []
        for xin, name, conf in zip([self.c_data, self.p_data, self.t_data],
            ['closeness', 'period', 'trend'], [self.c_conf, self.p_conf, self.t_conf]):
            len_seq, nb_flow, map_height, map_width = conf
            with tf.variable_scope(name):
                conv1 = builder.conv_layer(xin, 'conv1', 3, 64, 1)
                # resnet unit
                for i in range(self.nb_residual_unit):
                    with tf.variable_scope('resnet_{}'.format(i)):
                        conv1 = builder.block(conv1, 64, 1)
                relu1 = builder.activation_func(conv1)
                conv2 = builder.conv_layer(relu1, 'conv2', 3, self.out_channel, 1)
                outputs.append(conv2)
        new_outputs = []
        for i,output in enumerate(outputs):
            W = builder._get_variable('ilayer_{}'.format(i), [1, nb_flow, map_height, map_width])
            new_output = tf.multiply(output, W)
            new_outputs.append(new_output)
        main_output = tf.add_n(new_outputs)

        if self.external_dim != None and self.external_dim > 0:
            embedding = builder.fc_layer(self.external_input, 'external', 10)
            embedding = builder.activation_func(embedding)
            h1 = builder.fc_layer(embedding, 'fc', nb_flow * map_height * map_width)
            h1 = builder._bias(h1)
            activation = builder.activation_func(h1)
            external_output = tf.reshape(activation, [-1, nb_flow, map_height, map_width])
            main_output = tf.add(main_output, external_output, name='add')

        self.pred = tf.nn.tanh(main_output)
        self.loss = nn.mse(self.target, self.pred)
        self.metric = nn.rmse(self.target, self.pred)

        opt = tf.train.AdamOptimizer(lr)
        self.global_step = tf.train.get_or_create_global_step()
        grad = opt.compute_gradients(self.loss)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = opt.apply_gradients(grad, global_step = self.global_step)



