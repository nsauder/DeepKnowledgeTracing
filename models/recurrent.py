import numpy as np
import theano
import theano.tensor as T
import lasagne
import utils
fX = theano.config.floatX

class Recurrent(object):
    def __init__(self, num_input, num_hidden, num_output):
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output

    @property
    def params(self):
        return filter(utils.is_shared_variable, self.__dict__.values())


class RNN(Recurrent):
    def __init__(self, *args):
        super(RNN, self).__init__(*args)
        self.w_hh = utils.create_shared_variable(lasagne.init.Orthogonal,
                                                 (self.num_hidden, self.num_hidden))
        self.w_xh = utils.create_shared_variable(lasagne.init.Orthogonal,
                                                 (self.num_input, self.num_hidden))
        self.w_hy = utils.create_shared_variable(lasagne.init.Orthogonal,
                                                 (self.num_hidden,self.num_output))
        self.h = utils.create_shared_variable(lasagne.init.Uniform,
                                              (self.num_hidden,))
        self.b_h = utils.create_shared_variable(lasagne.init.Uniform,
                                                (self.num_hidden,))
        self.b_y = utils.create_shared_variable(lasagne.init.Uniform,
                                                (self.num_output,))

        


class LSTM(Recurrent):
    def __init__(self, *args):
        super(RNN, self).__init__(*args)
        self.w_hh = utils.create_shared_variable(lasagne.init.Orthogonal,
                                           (self.num_hidden, self.num_hidden))
        self.w_xh = utils.create_shared_variable(lasagne.init.Orthogonal,
                                           (self.num_input, self.num_hidden))
        self.w_hy = utils.create_shared_variable(lasagne.init.Orthogonal,
                                           (self.num_hidden,self.num_output))
        self.h = utils.create_shared_variable(lasagne.init.Uniform,
                                        (self.num_hidden,))
        self.b_h = utils.create_shared_variable(lasagne.init.Uniform,
                                          (self.num_hidden,))
        self.b_y = utils.create_shared_variable(lasagne.init.Uniform,
                                          (self.num_output,))

        
    def step(self, x, h):
        new_h = T.nnet.sigmoid(T.dot(h, self.w_hh) + T.dot(x, self.w_xh) + b_h)
        new_y = T.nnet.softmax(T.dot(new_h, self.w_hy) + b_y)

        return new_h, new_y
