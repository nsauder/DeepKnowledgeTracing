import numpy as np
import theano
import theano.tensor as T
import lasagne
from theano.tensor.shared_randomstreams import RandomStateSharedVariable
from theano.tensor.sharedvar import SharedVariable
fX = theano.config.floatX


def is_shared_variable(variable):
    return (isinstance(variable, SharedVariable) and
            not isinstance(variable, RandomStateSharedVariable) and
            not hasattr(variable.tag, 'is_rng'))


def create_shared_variable(initializer, shape):
    init = initializer()
    shared_var = theano.shared(init.sample(shape).astype(fX))
    return shared_var


def reduce_with_method(classes, method_name, initial_value):
    accum_value = initial_value
    for cls in classes:
        accum_value = cls.__getattribute__(method_name)(accum_value)
        
    return accum_value


def get_params(layers):
    return [param for lyr in layers for param in lyr.params]
