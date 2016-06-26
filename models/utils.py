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
