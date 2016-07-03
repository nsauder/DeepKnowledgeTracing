import numpy as np
import numpy.random as rng
import os
import functools
import pandas as pd
import sys
sys.path.append('/Users/nathanielsauder/n/repos/du')
import du
fX = np.float32

DATA_DIR = '../data/assistments'
data_file = os.path.join(DATA_DIR, 'assistments_with_fold.csv')


params = du.AttrDict(
    num_problems=140,
)


def generate_embeddings(size):
    prng = rng.RandomState(42)
    length = int(np.log(size))

    return rng.multivariate_normal(mean=np.zeros(length),
                                   cov=np.eye(length),
                                   size=size).astype(fX)


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

            expand = lambda prob, is_corr: prob + is_corr*params.num_problems
            combined_representation = np.array(map(expand, problem_array, answer_array))
                
            datamaps.append(
                {"problems": problem_array,
                 "is_correct": answer_array,
                 "combined_representation": combined_representation.astype(np.int32),
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
            key='combined_representation',
            fn=len,
            out='length',
        ).map(
            key='combined_representation',
            fn=lambda indices: lookup_embedding(indices, self.embedding),
            out='x',
        ).filter(
            key='length',
            fn=lambda x: x >= 2,
        )


        return ds
