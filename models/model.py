import numpy as np
import theano
import theano.tensor as T
import lasagne
import recurrent
import dataset
import utils
from theano.tensor.shared_randomstreams import RandomStateSharedVariable
from theano.tensor.sharedvar import SharedVariable
import sklearn.metrics
import sys
sys.path.append('/Users/nathanielsauder/n/repos/du')
import du
fX = theano.config.floatX

#################### Initial Parameters and Theano Variables ##########

params = du.AttrDict(
    num_hidden=200,
    num_problems=140,
    batch_size=128,
    num_iterations=10**9,
    valid_interval=10**3,
    num_valid=10**2,
)


x = T.matrix('x')
indices = T.ivector('indices')
labels = T.ivector('labels')


#################### Model Definition ####################

rnn = recurrent.RNN(int(np.log(2*params.num_problems)),
                    params.num_hidden,
                    params.num_problems)

def step_fn(x,
            idx,
            label,
            h,
            w_xh,
            w_hh,
            w_hy,
            b_h,
            b_y):
    
    new_h = T.nnet.sigmoid(T.dot(h, w_hh) + T.dot(x, w_xh) + b_h)
    new_y = T.nnet.softmax(T.dot(new_h, w_hy) + b_y).flatten(1)

    y_prob = new_y[idx]
    cost = T.nnet.binary_crossentropy(y_prob, label)

    return new_h, cost, y_prob


outputs, _ = theano.scan(
    fn=step_fn,
    sequences=[dict(input=x, taps=[-1]),
               dict(input=indices, taps=0),
               dict(input=labels, taps=0)],
    outputs_info=[dict(initial=rnn.h),
                  None,
                  None],
    non_sequences=[rnn.w_xh,
                   rnn.w_hh,
                   rnn.w_hy,
                   rnn.b_h,
                   rnn.b_y]
)

states, costs, preds = outputs
cost = T.mean(costs)
updates = lasagne.updates.adam(cost,
                               rnn.params)
# updates = lasagne.updates.sgd(cost,
#                               rnn.params,
#                               learning_rate=0.001)

test_fn = theano.function(inputs=[x, indices, labels],
                          outputs=[cost, preds])

train_fn = theano.function(inputs=[x, indices, labels],
                           outputs=cost,
                           updates=updates)


# #################### Data Flow  ####################

data = dataset.Data(params)
train_ds = data.dataset()
test_ds = data.dataset(is_test=True)

with test_ds as test_gen:
    test_chunks = []
    for _ in range(params.num_valid):
        test_chunks.append(test_gen.next())

    
with train_ds as train_gen:
    for iter_num in range(params.num_iterations):
        data_batch = train_gen.next()

        train_fn(data_batch['x'],
                 data_batch['problems'],
                 data_batch['is_correct'])
        
        if iter_num % params.valid_interval == 0:
            costs = []
            all_preds = []
            all_labels = []
            
            for data_batch in test_chunks:
                cost, preds = test_fn(data_batch['x'],
                                      data_batch['problems'],
                                      data_batch['is_correct'],)
                costs.append(cost)
                all_preds.append(preds)
                all_labels.append(data_batch['is_correct'][:-1])

            assert len(all_labels) == len(all_preds)
            test_auc = sklearn.metrics.roc_auc_score(np.concatenate(all_labels),
                                                     np.concatenate(all_preds))

            test_msg = "At iteration {}, the test error was {} and auc was {}"
            print test_msg.format(iter_num,
                                  np.mean(costs),
                                  test_auc)
