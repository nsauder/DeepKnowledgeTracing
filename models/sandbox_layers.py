import numpy as np
import theano
import theano.tensor as T
import lasagne
import utils
import abc
fX = theano.config.floatX

class Layer(object):
    __metaclass_ = abc.ABCMeta
    
    @property
    def params(self):
        return filter(utils.is_shared_variable, self.__dict__.values())

    @abc.abstractmethod
    def apply(self, x):
        pass
    

class Recurrent(object):
    def __init__(self,
                 num_input,
                 num_hidden,
                 hidden_to_hidden_init=lasagne.init.Orthogonal,
                 default_init=lasagne.init.Uniform):
        
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.hidden_to_hidden_init = hidden_to_hidden_init
        self.default_init = default_init

    @property
    def params(self):
        return filter(utils.is_shared_variable, self.__dict__.values())

    
class RNN(Recurrent):
    def __init__(self, *args):
        super(RNN, self).__init__(*args)
        
        self.h = utils.create_shared_variable(self.default_init,
                                              (1, self.num_hidden))
        self.x_to_h = utils.create_shared_variable(self.hidden_to_hidden_init,
                                                   (self.num_input,
                                                    self.num_hidden))
        self.h_to_h = utils.create_shared_variable(self.hidden_to_hidden_init,
                                                   (self.num_hidden,
                                                    self.num_hidden))
        self.b_h = utils.create_shared_variable(self.default_init,
                                                (self.num_hidden,))


    def apply(self, x):
        x_feat = x.dot(self.x_to_h) + self.b_h.dimshuffle('x', 'x', 0)
        x_feat = x_feat.dimshuffle(1, 0, 2)

        initial_h = T.repeat(self.h,
                             x.shape[0],
                             axis=0)

        def step(x_feat, h, h_to_h):

            new_h = T.tanh(x_feat + h.dot(h_to_h))
            return new_h

        outputs, _ = theano.scan(
            fn=step,
            sequences=[x_feat],                     
            outputs_info=[dict(initial=initial_h)],
            non_sequences=[self.h_to_h],
        )
        
        states = outputs

        return states.dimshuffle(1, 0, 2)


class LSTM(Recurrent):
    def __init__(self, *args):
        super(LSTM, self).__init__(*args)
        
        self.h = utils.create_shared_variable(self.default_init,
                                              (1, self.num_hidden))
        self.c = utils.create_shared_variable(self.default_init,
                                              (1, self.num_hidden))        
        self.x_to_i = utils.create_shared_variable(self.default_init,
                                                   (self.num_input, self.num_hidden,))
        self.x_to_f = utils.create_shared_variable(self.default_init,
                                                   (self.num_input, self.num_hidden,))
        self.x_to_g = utils.create_shared_variable(self.default_init,
                                                   (self.num_input, self.num_hidden,))
        self.x_to_o = utils.create_shared_variable(self.default_init,
                                                   (self.num_input, self.num_hidden,))
        self.h_to_i = utils.create_shared_variable(self.hidden_to_hidden_init,
                                                   (self.num_hidden, self.num_hidden,))
        self.h_to_f = utils.create_shared_variable(self.hidden_to_hidden_init,
                                                   (self.num_hidden, self.num_hidden,))
        self.h_to_g = utils.create_shared_variable(self.hidden_to_hidden_init,
                                                   (self.num_hidden, self.num_hidden,))
        self.h_to_o = utils.create_shared_variable(self.hidden_to_hidden_init,
                                                   (self.num_hidden, self.num_hidden,))
        self.b_i = utils.create_shared_variable(self.default_init,
                                                (self.num_hidden,))
        self.b_f = utils.create_shared_variable(self.default_init,
                                                (self.num_hidden,))
        self.b_g = utils.create_shared_variable(self.default_init,
                                                (self.num_hidden,))
        self.b_o = utils.create_shared_variable(self.default_init,
                                                (self.num_hidden,))

    def apply(self, x):
        x_to_inter = T.concatenate([self.x_to_f,
                                    self.x_to_i,
                                    self.x_to_g,
                                    self.x_to_o],
                                   axis=1)

        h_to_inter = T.concatenate([self.h_to_f,
                                    self.h_to_i,
                                    self.h_to_g,
                                    self.h_to_o],
                                   axis=1)

        b_inter = T.concatenate([self.b_f,
                                 self.b_i,
                                 self.b_g,
                                 self.b_o])
        
        x_feat = x.dot(x_to_inter) + b_inter.dimshuffle('x', 'x', 0)
        x_feat = x_feat.dimshuffle(1, 0, 2)

        initial_h = T.repeat(self.h,
                             x.shape[0],
                             axis=0)

        initial_c = T.repeat(self.c,
                             x.shape[0],
                             axis=0)


        def step(x_feat, h, c, h_to_inter):
            
            intermediates = T.tanh(x_feat + h.dot(h_to_inter))
    
            i = intermediates[:, :self.num_hidden]
            o = intermediates[:, self.num_hidden:2 * self.num_hidden]
            f = intermediates[:, 2 * self.num_hidden:3 * self.num_hidden]
            g = intermediates[:, 3 * self.num_hidden:]
    
            i = T.nnet.sigmoid(i)
            o = T.nnet.sigmoid(o)
            f = T.nnet.sigmoid(f)
            g = T.tanh(g)
                
            new_c = f * c + i * g
            new_h = o * new_c
        
            return new_h, new_c
    
        outputs, _ = theano.scan(fn=step,
                                 sequences=[x_feat],                     
                                 outputs_info=[dict(initial=initial_h),
                                               dict(initial=initial_c)],
                                 non_sequences=[h_to_inter])

        _, states = outputs

        return states.dimshuffle(1, 0, 2)


class GRU(Recurrent):
    def __init__(self, *args):
        super(GRU, self).__init__(*args)
        
        self.h = utils.create_shared_variable(self.default_init,
                                              (1, self.num_hidden))
        self.x_to_z = utils.create_shared_variable(self.default_init,
                                                   (self.num_input, self.num_hidden,))
        self.x_to_r = utils.create_shared_variable(self.default_init,
                                                   (self.num_input, self.num_hidden,))
        self.x_to_h = utils.create_shared_variable(self.default_init,
                                                   (self.num_input, self.num_hidden,))
        self.h_to_z = utils.create_shared_variable(self.hidden_to_hidden_init,
                                                   (self.num_hidden, self.num_hidden,))
        self.h_to_r = utils.create_shared_variable(self.hidden_to_hidden_init,
                                                   (self.num_hidden, self.num_hidden,))
        self.h_to_h = utils.create_shared_variable(self.hidden_to_hidden_init,
                                                   (self.num_hidden, self.num_hidden,))
        self.b_z = utils.create_shared_variable(self.default_init,
                                                (self.num_hidden,))
        self.b_r = utils.create_shared_variable(self.default_init,
                                                (self.num_hidden,))
        self.b_h = utils.create_shared_variable(self.default_init,
                                                (self.num_hidden,))

    def apply(self, x):
        x_to_inter = T.concatenate([self.x_to_z,
                                    self.x_to_r,
                                    self.x_to_h],
                                   axis=1)

        b_inter = T.concatenate([self.b_z,
                                 self.b_r,
                                 self.b_h])
        
        x_feat = x.dot(x_to_inter) + b_inter.dimshuffle('x', 'x', 0)
        x_feat = x_feat.dimshuffle(1, 0, 2)

        initial_h = T.repeat(self.h,
                             x.shape[0],
                             axis=0)

        def step(x_feat, h, h_to_z, h_to_r, h_to_h):
            x_to_z = x_feat[:, :self.num_hidden]
            x_to_r = x_feat[:, self.num_hidden:2*self.num_hidden]
            x_to_h = x_feat[:, 2*self.num_hidden:]

            z_gate = T.nnet.sigmoid(x_to_z + h.dot(self.h_to_z))
            r_gate = T.nnet.sigmoid(x_to_r + h.dot(self.h_to_r))
            
            h_added = T.tanh(x_to_z + (h*r_gate).dot(self.h_to_h))
            new_h = z_gate * h_added + (1-z_gate) * h
    
            return new_h
    
        outputs, _ = theano.scan(fn=step,
                                 sequences=[x_feat],                     
                                 outputs_info=[dict(initial=initial_h)],
                                 non_sequences=[self.h_to_z,
                                                self.h_to_r,
                                                self.h_to_h])

        states = outputs

        return states.dimshuffle(1, 0, 2)


class Dense(object):
    def __init__(self,
                 num_input,
                 num_output,
                 weight_init=lasagne.init.Orthogonal,
                 bias_init=lasagne.init.Uniform):
        
        self.num_input = num_input
        self.num_output = num_output
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.w = utils.create_shared_variable(self.weight_init,
                                              (self.num_input,
                                               self.num_output,))
        self.b = utils.create_shared_variable(self.bias_init,
                                              (self.num_output,))


    @property
    def params(self):
        return filter(utils.is_shared_variable, self.__dict__.values())

    
    def apply(self, x):
        return x.dot(self.w) + self.b
