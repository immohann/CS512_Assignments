import numpy as np
import pandas as pd
import re
from collections import OrderedDict


def read_for_crf(filename):
    data = []
    labels = []
    word_index = []
    with open(filename, 'r') as f:
        for lines in f.read().splitlines():
            inner_list = []
            for x in lines.split(' ')[5:]:
                inner_list.append(int(x))  #one row of train data
            data.append(np.array(inner_list)) #get all training data without the label in one array
            labels.append(ord(lines.split(' ')[1])-97) #get the labels of training data in one array
            word_index.append(int(lines.split(' ')[3]))
    np_data = np.asarray(data)
    ids = {}
    X_dataset = []
    y_dataset = []
    current = 0
    y = labels

    X_dataset.append([])
    y_dataset.append([])
    for i in range(len(y)):
        # computes an inverse map of word id to id in the dataset
        if word_index[i] not in ids:
            ids[word_index[i]] = current

        X_dataset[current].append(data[i])
        y_dataset[current].append(y[i])

        if (i + 1 < len(y) and word_index[i] != word_index[i + 1]):
            X_dataset[current] = np.array(X_dataset[current])
            y_dataset[current] = np.array(y_dataset[current])

            X_dataset.append([])
            y_dataset.append([])

            current = current + 1

    # if not return_ids:
    return X_dataset, y_dataset
    # else:
    #     return X_dataset, y_dataset, ids
    # return np_data, labels

def _load_structured_svm_data(filename):
    file = open(filename, 'r')
    X = []
    y = []
    word_ids = []

    for line in file:
        temp = line.split()
        # get label
        label_string = temp[0]
        # y to 0...25 instead of 1...26
        y.append(int(label_string) - 1)

        # get word id
        word_id_string = re.split(':', temp[1])[1]
        word_ids.append(int(word_id_string))

        x = np.zeros(128)
        for elt in temp[2:]:
            index = re.split(':', elt)
            x[int(index[0]) - 1] = 1

        X.append(x)

    y = np.array(y)
    word_ids = np.array(word_ids)

    return X, y, word_ids

def prepare_structured_dataset(filename, return_ids=False):
    # get initial output
    X, y, word_ids = _load_structured_svm_data(filename)
    ids = {}
    X_dataset = []
    y_dataset = []
    current = 0

    X_dataset.append([])
    y_dataset.append([])

    for i in range(len(y)):
        # computes an inverse map of word id to id in the dataset
        if word_ids[i] not in ids:
            ids[word_ids[i]] = current

        X_dataset[current].append(X[i])
        y_dataset[current].append(y[i])

        if (i + 1 < len(y) and word_ids[i] != word_ids[i + 1]):
            X_dataset[current] = np.array(X_dataset[current])
            y_dataset[current] = np.array(y_dataset[current])

            X_dataset.append([])
            y_dataset.append([])

            current = current + 1

    if not return_ids:
        return X_dataset, y_dataset
    else:
        return X_dataset, y_dataset, ids

# train_data, labels = read_for_crf('train.txt')
# print(train_data, labels)
# print(type(train_data))
# test_data = read_for_crf('test.txt')
# placeholder_read()
