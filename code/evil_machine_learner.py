import numpy as np
import os,pickle
from subprocess import run, PIPE
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

import scipy.optimize as opt
from read_data import read_for_crf, prepare_structured_dataset, crf_dataLoader, linearSVM_dataLoader, get_Words, get_Characters
from all_functions import *
from ref_optimize import fmin_tnc_results,ref_optimize
from SVM import linear_SVM_training_performance

# with open('saved_dictionary.pkl', 'rb') as f:
#     fmin_tnc_res = pickle.load(f)
path = os.getcwd()
char_scores, word_scores = [], []

q4_train_struct = "/data/train_struct.txt"
q4_test_struct = "/data/test_struct.txt"

def plotAccuracy(dataset_range, scale = 'log', X_label = 'C'):

    #character-wise accuracy
    plt.plot(dataset_range, char_scores, label = 'Character-wise Scores', color='red', marker='o', linewidth=3)
    plt.title('Character Level Prediction Accuracy', fontsize=14)
    plt.xlabel(X_label)
    if scale is not None: 
        plt.xscale(scale)
    plt.xticks(dataset_range)
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    #word-wise accuracy
    plt.plot(dataset_range, word_scores, label='Word-wise Scores', color='red', marker='o', linewidth=3)
    plt.title('Word Level Prediction Accuracy', fontsize=14)
    plt.xlabel(X_label)
    if scale is not None: 
        plt.xscale(scale)
    plt.xticks(dataset_range)
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

def linearSVM_transformedData(C= 1.0, transform = True, limit = None):
    X ,y = [], []
    training_dataset, test_dataset = crf_dataLoader(path_ext='/data/train.txt')

    if transform:
        training_dataset = linearSVM_dataLoader(training_dataset, limit) 

    for _, v in training_dataset.items():
        X.append(v[-1])
        y.append(v[0])

    X, y = np.array(X), np.array(y)

    linearSVM_Model = LinearSVC(C=C, max_iter=1000, verbose=10, random_state=0)
    linearSVM_Model.fit(X, y)

    X , y = [], []
    Target_Word_idx = []

    for _, v in test_dataset.items():
        X.append(v[-1])
        y.append(v[0])
        Target_Word_idx.append(v[2])

    X, y = np.array(X), np.array(y)[:, 0]

    predicted_words = linearSVM_Model.predict(X)
    char_acc, word_acc = linear_SVM_performance(y, predicted_words, Target_Word_idx)

    char_scores.append(char_acc)
    word_scores.append(word_acc)


if __name__ == '__main__':

    # #Linear SVM performance on transformed Data
    L_values = [500, 1000, 1500, 2000]

    char_scores, word_scores = [], []

    for limit in L_values:
        linearSVM_transformedData(C=1.0, transform = True, limit = limit)
    
    plotAccuracy(L_values, scale = None, X_label='distortion count')

    #CRF performance on Transformed Data
    char_scores.clear()
    word_scores.clear()

    Testing_dataset, Testing_targets = prepare_structured_dataset(path + '/data/test_struct.txt')

    W_T = load_model_txt(path + '/result/c_equal_1000.txt')
    W, t = extract_W_T(W_T)

    Testing_predictions = decode_crf(Testing_dataset, W, t)

    word_acc, char_acc = compute_word_char_accuracy_score(Testing_predictions, Testing_targets)
    char_scores.append(char_acc)
    word_scores.append(word_acc)

    for limit in L_values:
        print("transforming data of first %d ids" % (limit))

        # #comment out after training is complete
        # X_train, Y_train = prepare_structured_dataset(path + '/data/train_distorted_%d.txt' % (limit))
        # X_train = crf_dataLoader(X_train)
        # ref_optimize(W_T, X_train, Y_train, C=1000, model_name='model_%d_distortion' % limit)

        W_T = load_model_txt(path + '/result/model_%d_distortion' % limit + '.txt')
        W, t = extract_W_T(W_T)

        Test_predictions = decode_crf(Testing_dataset, W, t)

        word_accuracy, character_accuracy = compute_word_char_accuracy_score(Test_predictions, Testing_targets)

        char_scores.append(character_accuracy)
        word_scores.append(word_accuracy)

    L_values = [0] + L_values
    plotAccuracy(L_values, scale=None, X_label='distortion count')

