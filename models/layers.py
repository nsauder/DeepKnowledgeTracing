import numpy as np
import theano
import theano.tensor as T
import lasagne
import utils
fX = theano.config.floatX

LETTER_TO_LAYER = {
    'l': LSTM,
    'r': RNN,
    'd': Dense,
}

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

        def step(x_feat, h, h_to_h, h_to_y, b_y):

            new_h = T.tanh(x_feat + h.dot(h_to_h))
            return new_h

        outputs, _ = theano.scan(
            fn=step,
            sequences=[x_feat],                     
            outputs_info=[dict(initial=initial_h)],
            non_sequences=[self.h_to_h],
        )
        
        _, states = outputs

        return states


class LSTM(Recurrent):
    def __init__(self, *args):
        super(LSTM, self).__init__(*args)
        
        self.h = utils.create_shared_variable(self.default_init,
                                              (1, self.num_hidden))
        self.c = utils.create_shared_variable(self.default_init,
                                              (1, self.num_hidden))        
        self.x_to_inter = utils.create_shared_variable(self.hidden_to_hidden_init,
                                                       (self.num_input,
                                                        4*self.num_hidden,))
        self.h_to_inter = utils.create_shared_variable(self.hidden_to_hidden_init,
                                                       (self.num_hidden,
                                                        4*self.num_hidden,))
        self.b_inter = utils.create_shared_variable(self.default_init,
                                                  (4*self.num_hidden,))

    def apply(self, x):
        x_feat = x.dot(self.x_to_inter) + self.b_inter.dimshuffle('x', 'x', 0)
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
                                 non_sequences=[self.h_to_inter])

        _, states = outputs

        return states

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

# x_in = np.random.random((50,
#                          200,
#                          5)).astype(np.float32)

# fn = theano.function(inputs=[x],
#                      outputs=[y_out])

# fn(x_in)
