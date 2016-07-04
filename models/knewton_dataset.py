import numpy as np
import numpy.random as rng
import os
import functools
import pandas as pd
import sys
sys.path.append('/home/ubuntu/du')
sys.path.append('/Users/nathanielsauder/n/repos/du')

import du
fX = np.float32

DATA_DIR = '../data/assistments'
data_file = os.path.join(DATA_DIR, 'assistments_with_fold.csv')


def to_one_hot(index_sequence, params):
    array = np.zeros((index_sequence.shape[0], params.num_problems))
    for row, col in enumerate(index_sequence):
        array[row, col] = 1

    return array.astype(np.int32)


def pad_arrays(arrays, lengths):
    max_len = max(lengths)

    padded_arrays = []
    for arr, length in zip(arrays, lengths):
        npad = ((0,0), (0, max_len-length), (0,0))
        padded_arrays.append(np.pad(arr,
                                    pad_width=npad,
                                    mode='constant',
                                    constant_values=0))

    return np.concatenate(padded_arrays)


def generate_embeddings(size):
    prng = rng.RandomState(42)
    length = 4 * int(np.log(size))

    return rng.multivariate_normal(mean=np.zeros(length),
                                   cov=np.eye(length),
                                   size=size).astype(fX)


def combine_representations(problem_array, answer_array, params):
    def expand(prob, is_corr):
        return prob + is_corr*params.num_problems
    
    return np.array(map(expand, problem_array, answer_array))


def lookup_embedding(indices, embedding):
    return np.array([embedding[index] for index in indices])


class Data(object):
    def __init__(self, params):
        self.params = params
        self.embedding = generate_embeddings(params.num_problems*2)

    def initial_rows(self):
        df = pd.read_csv(data_file)
        datamaps = []
        
        for _, group in df.groupby('user_idx'):
            problem_array = group.item_idx.values.astype(np.int32)
            answer_array = group.correct.values.astype(np.int32)
            fold_idx, = np.unique(group.fold_idx.values)

            datamaps.append(
                {"problems": problem_array,
                 "is_correct": answer_array,
                 "fold_num": fold_idx,}
            )
                
        return datamaps

    
    def dataset(self, fold_num=0, is_test=False):
        rows = self.initial_rows()
        ds = du.dataset.from_list(rows)

        if is_test:
            ds = ds.filter(
                key='fold_num',
                fn=lambda x: x == fold_num,
            )

        else:
            ds = ds.filter(
                key='fold_num',
                fn=lambda x: x != fold_num,
            ).random_sample()

        ds = ds.map(
            key='problems',
            fn=len,
            out='length',
        ).filter(
            key='length',
            fn=lambda x: x >= 2,
        ).map(
            key=['problems', 'is_correct'],
            fn=lambda p, c: combine_representations(p, c, self.params),
            out='combined_representation',
        ).map(
            key='combined_representation',
            fn=lambda indices: lookup_embedding(indices, self.embedding),
            out='x',
        ).map(
            key='problems',
            fn=lambda arr: to_one_hot(arr, self.params),
            out='mask',
        ).map_key(
            key='is_correct',
            fn=lambda x: x[1:],
        ).map_key(
            key='x',
            fn=lambda x: x[:-1][np.newaxis],
        ).map_key(
            key='mask',
            fn=lambda x: x[1:][np.newaxis],
        ).chunk(
            chunk_size=self.params.batch_size,
        ).map_key(
            key='is_correct',
            fn=np.concatenate,
        ).map(
            key=['x', 'length'],
            fn=pad_arrays,
            out='x',
        ).map(
            key=['mask', 'length'],
            fn=pad_arrays,
            out='mask',
        ).map_key(
            key='mask',
            fn=lambda x: x.reshape(-1, self.params.num_problems)
        )

        
        return ds
