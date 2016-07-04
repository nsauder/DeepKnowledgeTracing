import numpy as np
import numpy.random as rng
import os
import functools
import sys
sys.path.append('/home/ubuntu/du')
import du
fX = np.float32

DATA_DIR = '../data/assistments'
train_file = os.path.join(DATA_DIR, 'builder_train.csv')
test_file = os.path.join(DATA_DIR, 'builder_test.csv')


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
    length = int(np.log(size))

    return rng.multivariate_normal(mean=np.zeros(length),
                                   cov=np.eye(length),
                                   size=size).astype(fX)


def lookup_embedding(indices, embedding):
    return np.array([embedding[index] for index in indices])


def read_csv(file_name):
    f = open(file_name, 'r')
    lines = f.readlines()
    del lines[::3]
    
    lines = [map(int, li.split(',')[:-1]) for li in lines]
    problems = lines[::2]
    answers = lines[1::2]
    assert len(problems) == len(answers)

    return zip(problems, answers)


def combine_representations(problem_array, answer_array, params):
    def expand(prob, is_corr):
        return prob + is_corr*params.num_problems
    
    return np.array(map(expand, problem_array, answer_array))



class Data(object):
    def __init__(self, params):
        self.params = params
        self.embedding = generate_embeddings(self.params.num_problems*2)

    def initial_rows(self):
        train_data = read_csv(train_file)
        # dedup the test set
        # test_data = filter(lambda row: row not in train_data, read_csv(test_file))
        test_data = read_csv(test_file)

        def add_datamaps(input_data, is_test=False):
            datamaps = []
            for problem_array, answer_array in input_data:
                datamaps.append(
                    {"problems": np.array(problem_array).astype(np.int32),
                     "is_correct": np.array(answer_array).astype(np.int32),
                     "is_test": is_test}
                )
                
            return datamaps

        return add_datamaps(train_data) + add_datamaps(test_data, is_test=True)

    
    def dataset(self, is_test=False):
        rows = self.initial_rows()
        ds = du.dataset.from_list(rows)

        if is_test:
            ds = ds.filter(
                key='is_test',
                fn=du.identity
            )

        else:
            ds = ds.filter(
                key='is_test',
                fn=lambda x: not x,
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
