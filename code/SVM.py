#%%
import numpy as np
import os
from subprocess import run, PIPE
from sklearn.svm import LinearSVC

import scipy.optimize as opt
from read_data import read_for_crf, prepare_structured_dataset, crf_dataLoader, linearSVM_dataLoader, get_Words, get_Characters
from all_functions import *
from ref_optimize import fmin_tnc_results
import matplotlib.pyplot as plt

path = os.getcwd()

#paths fir data and models
q3_model_struct = '/data/model_trained.txt'
q3_struct_test_predictions = '/data/test_predictions.txt'
q3_struct_train = '/data/train_struct.txt'
q3_struct_test = '/data/test_struct.txt'

# scores to be used for plotting 
char_scores,word_scores = [],[]


#function for plotting character-wise and word-wise accuracy scores
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

#function for training SVMHMM
def SVM_struct_training(C=1.0):
    parameters = ['svm_hmm_windows/svm_hmm_learn', '-c', str(C), path + q3_struct_train, path + q3_model_struct]
    print('SVM trained')
    result = run(parameters, stdin=PIPE)
    print('RESULT ========', result)

#function for evaluation SVMHMM
def SVM_struct_performance():
    # if not os.path.exists(q3_model_struct):
    #     print("SVM-HMM model does not exist.Please train the SVM-HMM model.")
    #     exit(0)
    # else:
    parameters = ['svm_hmm_windows/svm_hmm_classify',path + q3_struct_test,path + q3_model_struct, path + q3_struct_test_predictions]
    result = run(parameters,stdin=PIPE)
    print(result)
    character_accuracy, word_accuracy = structuredSVM_performance(path + q3_struct_test, path + q3_struct_test_predictions)
    char_scores.append(character_accuracy)
    word_scores.append(word_accuracy)

# function for training and evaluating the performance of linear SVM 
def linear_SVM_training_performance(C = 1.0):

    Training_dataset, Training_targets = prepare_structured_dataset(path + '/data/train_struct.txt')
    testing_dataset, Testing_targets = prepare_structured_dataset(path + '/data/test_struct.txt')

    #loading training dataset and target values + testing dataset
    xtrain, ytrain, xtest = get_Characters(Training_dataset), get_Characters(Training_targets), get_Characters(testing_dataset)

    #defining and training linear SVM model
    Linear_SVM_model = LinearSVC(C=C, verbose=10, random_state=0)
    Linear_SVM_model.fit(xtrain, ytrain)

    #Fetching predictions on test data with the trained model
    Testing_predictions = Linear_SVM_model.predict(xtest)

    #Fetching words from the predictions on test data
    Testing_predictions = get_Words(Testing_predictions, Testing_targets)

    #Calculating word-wise and character-wise accuracies
    wordwise_accuracy, characterwise_accuracy = compute_word_char_accuracy_score(Testing_predictions, Testing_targets)

    char_scores.append(characterwise_accuracy)
    word_scores.append(wordwise_accuracy)


def structured_SVM():
    C_values = [1.0, 10.0, 100.0, 1000.0]

    #C value for max scores at word and character level
    #C_values = [1000.0]

    char_scores.clear()
    word_scores.clear()

    for C in C_values:
        SVM_struct_training(C=C)
        SVM_struct_performance()

    plotAccuracy(C_values)


def linear_SVM():

    C_values = [1e-3, 1e-2, 1e-1, 1.0]

    #C value for max scores at word and character level
    #C_values = [1.0]

    char_scores.clear()
    word_scores.clear()

    for C in C_values:
        linear_SVM_training_performance(C=C)

    plotAccuracy(C_values)

def CRF():

    Training_dataset, Training_targets = prepare_structured_dataset(path + '/data/train_struct.txt')
    testing_dataset, Testing_targets = prepare_structured_dataset(path + '/data/test_struct.txt')
    parameters = load_model_txt(path + "/data/model.txt")

    C_values = [1, 10, 100, 1000]

    #Comment out after optimization is done once
    # for C in C_values:
    #     optimize(parameters, Training_dataset, Training_targets, C, 'solution' + str(C))
    #     print("optimization Complete" + str(C))

    char_scores.clear()
    word_scores.clear()

    for C in C_values:
        print("Computing CRF predictions for C value = %d"%(C))
        #params = load_model_txt('model_C_' + str(C))
        W_T = load_model_txt(path + '/result/c_equal_'+ str(C)+".txt")
        W, t = extract_W_T(W_T)

        prediction = decode_crf(testing_dataset,W,t)

        word_accuracy, char_accuracy = compute_word_char_accuracy_score(prediction, Testing_targets)

        word_scores.append(word_accuracy)
        char_scores.append(char_accuracy)

    plotAccuracy(C_values)


if __name__ == '__main__':
    structured_SVM()
    linear_SVM()
    CRF()
    
