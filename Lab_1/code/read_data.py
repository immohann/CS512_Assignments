import numpy as np
import pandas as pd
import os
import scipy
import re
from collections import OrderedDict, defaultdict

path = os.getcwd()

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
    init = 0
    y = labels

    X_dataset.append([])
    y_dataset.append([])
    for i in range(len(y)):
        # computes an inverse map of word id to id in the dataset
        if word_index[i] not in ids:
            ids[word_index[i]] = init

        X_dataset[init].append(data[i])
        y_dataset[init].append(y[i])

        if (i + 1 < len(y) and word_index[i] != word_index[i + 1]):
            X_dataset[init] = np.array(X_dataset[init])
            y_dataset[init] = np.array(y_dataset[init])

            X_dataset.append([])
            y_dataset.append([])

            init = init + 1

    return X_dataset, y_dataset

#load SVM data
def _load_structured_svm_data(filename):
    file = open(filename, 'r')
    X, y,id = [], [], []

    for line in file:
        temp = line.split()
        label_string = temp[0]
        y.append(int(label_string) - 1)

       
        word_id_string = re.split(':', temp[1])[1]
        id.append(int(word_id_string))

        x = np.zeros(128)
        for elt in temp[2:]:
            index = re.split(':', elt)
            x[int(index[0]) - 1] = 1

        X.append(x)

    y = np.array(y)
    id = np.array(id)

    return X, y, id

#to prepare X, Y dataset
def prepare_structured_dataset(filename, return_ids=False):
    # get initial output
    X, y, id = _load_structured_svm_data(filename)
    ids = {}
    
    X_dataset, y_dataset = [], []

    init=0

    X_dataset.append([])
    y_dataset.append([])

    for i in range(len(y)):
        # computes an inverse map of word id to id in the dataset
        if id[i] not in ids:
            ids[id[i]] = init

        X_dataset[init].append(X[i])
        y_dataset[init].append(y[i])

        if (i + 1 < len(y) and id[i] != id[i + 1]):
            X_dataset[init] = np.array(X_dataset[init])
            y_dataset[init] = np.array(y_dataset[init])

            X_dataset.append([])
            y_dataset.append([])

            init = init + 1

    if not return_ids:
        return X_dataset, y_dataset
    else:
        return X_dataset, y_dataset, ids

#Function for loading Q2 data
def crf_dataLoader():
    training_data,testing_data = OrderedDict(),OrderedDict()
    
    with open(path + '/data/train.txt', 'r') as file:
        training_dataset = file.readlines()


    for line in training_dataset:
        character = re.findall('[a-z]', str(line))
        line = re.findall('\d+', str(line))
        character_id = int(line[0])
        next_word = line[1]
        word_id = line[2]
        position = line[3]
        pixel_value = np.array(line[4:], dtype=np.float32)
        training_data[character_id] = [character, next_word, word_id, position, pixel_value]

    
    with open(path + '/data/test.txt', 'r') as file:
        testing_dataset = file.readlines()

    for line in testing_dataset:
        character = re.findall(r'[a-z]', line)
        line = re.findall(r'\d+', line)
        character_id = line[0]
        next_word = line[1]
        word_id = line[2]
        position = line[3]
        pixel_value = np.array(line[4:], dtype=np.float32)
        testing_data[character_id] = [character, next_word, word_id, position, pixel_value]

    return training_data, testing_data

# #returns trained model parameters
# def load_model_txt(filename):
#     w_t = []
#     with open(filename, 'r') as f:
#         for i, parameters in enumerate(f):
#             w_t.append(float(parameters))

#     return np.array(w_t)


def crf_dataLoader_distort(training_dataset, testing_dataset):
    
    training_data,testing_data = OrderedDict(),OrderedDict()
    
    for line in training_dataset:
        character = re.findall('[a-z]', str(line))
        line = re.findall('\d+', str(line))
        character_id = int(line[0])
        next_word = line[1]
        word_id = line[2]
        position = line[3]
        pixel_value = np.array(line[4:], dtype=np.float32)
        training_data[character_id] = [character, next_word, word_id, position, pixel_value]

    
    with open(path + '/data/test.txt', 'r') as file:
        testing_dataset = file.readlines()

    for line in testing_dataset:
        character = re.findall(r'[a-z]', line)
        line = re.findall(r'\d+', line)
        character_id = line[0]
        next_word = line[1]
        word_id = line[2]
        position = line[3]
        pixel_value = np.array(line[4:], dtype=np.float32)
        testing_data[character_id] = [character, next_word, word_id, position, pixel_value]

    return training_data, testing_data


def linearSVM_dataLoader(training_dataset, limit):
    if limit == 0:
        return training_dataset

    word_dict = defaultdict(list)

    for key, value in training_dataset.items():
        word_id = value[2]
        word_dict[word_id].append(key)

    with open(path + '/data/transform.txt', 'r') as f:
        lines = f.readlines()

    lines = lines[:limit]

    for line in lines:
        splits = line.split()
        action = splits[0]
        target_word = splits[1]
        args = splits[2:]

        # get all of the ids in train set which have this word in them
        target_image_ids = word_dict[target_word]

        for image_id in target_image_ids:
            value_set = training_dataset[image_id]
            image = value_set[-1]

            if action == 'r':
                alpha = args[0]
                image = rotate_image(image, alpha)
            else:
                offsets = args
                image = translate_image(image, offsets)

            value_set[-1] = image.flatten()

            training_dataset[image_id] = value_set

    return training_dataset


#helper functions for data conversion

# returns characters from words
def get_Characters(X):
    characters = []
    for word in X:
        for char in word:
            characters.append(char)
    return np.array(characters)


#returns words from characters
def get_Words(X, y):
    words = [[]]
    cnt = 0
    for idx, word in enumerate(y):
        length = len(word)
        for _ in range(length):
            words[idx].append(X[cnt])
            cnt += 1
        words.append([])
    return words

#rotates an image by theta degrees

# train_data, labels = read_for_crf('train.txt')
# print(train_data, labels)
# print(type(train_data))
# test_data = read_for_crf('test.txt')
# placeholder_read()

def rotate_image(image, theta):

    image = image.reshape((16, 8))
    transformed_image = scipy.ndimage.interpolation.rotate(image, angle=float(theta))
    delta_height = int((transformed_image.shape[0] + 1 - image.shape[0]) // 2)
    delta_width = int((transformed_image.shape[1] + 1 - image.shape[1]) // 2)

    transformed_image = transformed_image[delta_height:delta_height + image.shape[0], delta_width: delta_width + image.shape[1]]

    loc = np.where(transformed_image == 0)
    transformed_image[loc] = image[loc]

    return transformed_image

#returns the target value of an image
def translate_image(image, offsets):
    image = image.reshape((16, 8))
    target_value = image
    delta_height, delta_width = int(offsets[0]), int(offsets[1])

    target_value[max(0, delta_height): min(image.shape[0], image.shape[0] + delta_height),
    max(0, delta_width): min(image.shape[1], image.shape[1] + delta_width)] = image[max(0, 1 - delta_height): min(image.shape[0], image.shape[0] - delta_height),
                                                          max(0, 1 - delta_width): min(image.shape[1], image.shape[1] - delta_width)]

    target_value[delta_height: image.shape[0], delta_width: image.shape[1]] = image[0: image.shape[0] - delta_height, 0: image.shape[1] - delta_width]

    return target_value

# train_data, labels = read_for_crf('train.txt')
# print(train_data, labels)
# print(type(train_data))
# test_data = read_for_crf('test.txt')
# placeholder_read()
