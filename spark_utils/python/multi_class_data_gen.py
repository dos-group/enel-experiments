from sklearn import datasets
import numpy as np


class MultiClassDataGenerator:
    def __init__(self, no_samples, no_features, no_class):
        self.no_class = no_class
        self.no_features = no_features
        self.no_samples = no_samples

    def generate_data(self, file_path):
        X, Y = datasets.make_blobs(n_samples=self.no_samples, centers=self.no_class, n_features=self.no_features)
        Y = Y.astype(int)
        data = np.concatenate((Y[:, np.newaxis], X), axis=1)
        np.savetxt(file_path, data, fmt="%.5f", delimiter=",")
        return data

    # def persist(self):
    #     from hdfs import InsecureClient
    #     client = InsecureClient('http://host:port', user='ann')
    #
    #     from json import dump, dumps
    #     records = [
    #         {'name': 'foo', 'weight': 1},
    #         {'name': 'bar', 'weight': 2},
    #     ]
    #
    #     # As a context manager:
    #     with client.write('data/records.jsonl', encoding='utf-8') as writer:
    #         dump(records, writer)
    #
    #     # Or, passing in a generator directly:
    #     client.write('data/records.jsonl', data=dumps(records), encoding='utf-8')
