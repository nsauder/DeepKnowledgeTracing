import numpy as np
import numpy.random as rng
import os
import functools

import sys
sys.path.append('/Users/nathanielsauder/n/repos/du')
import du
import du.dataset 

DATA_DIR = '../data/assistments'

params = du.AttrDict(
    num_problems=140,
    )

def generate_embeddings(size):
    prng = rng.RandomState(42)
    length = int(np.log(size))

    return rng.multivariate_normal(mean=np.zeros(length),
                                         cov=np.eye(length),
                                         size=size)


def lookup_embedding(indices, embedding):
    return np.array([embedding[index] for index in indices])


def combined_representation(problem_array, correct_array, num_problems):
    return np.array([prob*(1+is_corr)*num_problems for prob, is_corr
                     in zip(problem_array, correct_array)])

    
class Data(object):
    def __init__(self, params):
        self.params = params
        self.embedding = generate_embeddings(params.num_problems*2)

    def initial_rows(self):
        datamaps = []
        train_file = os.path.join(DATA_DIR, 'builder_train.csv')
        test_file = os.path.join(DATA_DIR, 'builder_test.csv')

        def add_datamaps(file_name, is_test=False):
            f = open(file_name, 'r')
            lines = f.readlines()
            del lines[::3]
            
            lines = [map(int, li.split(',')[:-1]) for li in lines]
            problems = lines[::2]
            answers = lines[1::2]
            assert len(problems) == len(answers)

            datamaps = []
            for problem_array, answer_array in zip(problems, answers):
                
                combined_representation = np.array([prob+is_corr*params.num_problems
                                                    for prob, is_corr
                                                    in zip(problem_array, answer_array)])

                datamaps.append({"problems": np.array(problem_array),
                                 "is_correct": np.array(answer_array),
                                 "combined_representation": combined_representation,
                                 "is_test": is_test})
                
            return datamaps
        
        return add_datamaps(train_file) + add_datamaps(test_file, is_test=True)

    
    def dataset(self):
        rows = self.initial_rows()
        ds = du.dataset.from_list(rows)

        ds = ds.map(
            key='combined_representation',
            fn=len,
            out='length',
        ).map(
            key='combined_representation',
            fn=lambda indices: lookup_embedding(indices, self.embedding),
            out='x',
        )

        return ds

    
data = Data(params)
ds = data.dataset()
