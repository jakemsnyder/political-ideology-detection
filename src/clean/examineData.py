import operator

from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np

TRAINING_DATA_PATH = '../../data/csv/model/train.csv'


def getSortedWordFrequencies(doc_list, topNWords):
    count_vect = CountVectorizer(input='content')
    matrix = count_vect.fit_transform(doc_list)
    frequencyCounts = np.squeeze(np.asarray(matrix.sum(axis=0)))
    frequencies = list(zip(count_vect.get_feature_names(), frequencyCounts))
    frequencies.sort(key=lambda x: x[1], reverse=True)
    return frequencies[:topNWords]


data = pd.read_csv(TRAINING_DATA_PATH)
sentences = data['sentence']

wordFrequencies = getSortedWordFrequencies(sentences, 100)

print("Word Frequencies")
for word, frequency in wordFrequencies:
    print('{:s} : {:d}'.format(word, frequency))
