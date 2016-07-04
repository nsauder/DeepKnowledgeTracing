import numpy as np
import theano
import theano.tensor as T
import lasagne
import sandbox_layers
import knewton_dataset as dataset
import utils
from theano.tensor.shared_randomstreams import RandomStateSharedVariable
from theano.tensor.sharedvar import SharedVariable
import sklearn.metrics
import sys
sys.path.append('/home/ubuntu/du')
sys.path.append('/Users/nathanielsauder/n/repos/du')
import du
fX = theano.config.floatX

#################### Initial Parameters and Theano Variables ##########



params = du.AttrDict(
    num_hidden=200,
    num_problems=140,
    batch_size=1,
    num_iterations=10**9,
    valid_interval=10**3,
    num_valid=10**2,
)


x = T.tensor3('x')
mask = T.imatrix('mask')
labels = T.ivector('labels')

num_units = [20,
             200,
             params.num_problems]


LETTER_TO_LAYER = {
    'l': sandbox_layers.LSTM,
    'r': sandbox_layers.RNN,
    'g': sandbox_layers.GRU,
    'd': sandbox_layers.Dense,
}

layers = []
for i, letter in enumerate('gd'):
    layers.append(LETTER_TO_LAYER[letter](*num_units[i:i+2]))


y_out = utils.reduce_with_method(classes=layers,
                                 method_name='apply',
                                 initial_value=x) 

reshaped_output = y_out.dimshuffle(1, 0, 2).reshape((-1, params.num_problems))
preds = T.nnet.softmax(reshaped_output)[mask.nonzero()]
cost = T.mean(T.nnet.binary_crossentropy(preds, labels))


updates = lasagne.updates.adam(cost,
                               utils.get_params(layers),
)

train_fn = theano.function(inputs=[x, mask, labels],
                           outputs=[cost],
                           updates=updates)

test_fn = theano.function(inputs=[x, mask, labels],
                          outputs=[cost, preds])



# #################### Data Flow  ####################


data = dataset.Data(params)
train_ds = data.dataset()
test_ds = data.dataset(is_test=True)


with test_ds as test_gen:
    test_chunks = list(test_gen)
 
    
with train_ds as train_gen:
    enum_gen = enumerate(train_gen)
    while True:
        iter_num, data_batch = enum_gen.next()

        if iter_num > params.num_iterations:
            break
        
        train_fn(data_batch['x'],
                 data_batch['mask'],
                 data_batch['is_correct'])        
        
        if iter_num % params.valid_interval == 0:
            costs = []
            all_preds = []
            all_labels = []
            
            for test_batch in test_chunks:
                cost, preds = test_fn(test_batch['x'],
                                      test_batch['mask'],
                                      test_batch['is_correct'],)
                
        
                costs.append(cost)
                all_preds.append(preds)
                all_labels.append(test_batch['is_correct'])

            assert len(all_labels) == len(all_preds)
            test_auc = sklearn.metrics.roc_auc_score(np.concatenate(all_labels),
                                                     np.concatenate(all_preds))

            test_acc = sklearn.metrics.accuracy_score(np.concatenate(all_labels),
                                                      np.concatenate(all_preds) > 1 / 10.)

            
            test_msg = "At iteration {}, test acc was {}, and test auc was {}"
            print test_msg.format(iter_num,
                                  test_acc,
                                  test_auc,)
