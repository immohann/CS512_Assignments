# Imports
import numpy as np
import os
from subprocess import run, PIPE
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
import pickle

import scipy.optimize as opt
from read_data import read_for_crf, prepare_structured_dataset, crf_dataLoader, linearSVM_dataLoader, get_Words, get_Characters
from all_functions import *

fmin_tnc_results = {}
path = os.path.dirname(os.getcwd())



#for 2a   #Train data to calculate the gradient on
X_gradient_compute, y_gradient_compute = read_for_crf(path + "/data/train.txt")

#Model.txt contains the weights for W and T. W_T_matrix is the list containing both W and T.
W_T_matrix = load_model_txt(path + "/data/model.txt")


#for 2b
X_train, y_train = prepare_structured_dataset(path + "/data/train_struct.txt")    
X_test, y_test = prepare_structured_dataset(path + "/data/train_struct.txt")

def gradient_compute(X_train, y_train,W_T_matrix):
    # print(path)

 
    check_gradient(W_T_matrix, X_train, y_train)
    # print(X_train)

    start = time.time()
    average_gradient = averaged_gradient(W_T_matrix, X_train, y_train, len(X_train))
    print("Total time:", time.time() - start)


    with open(path + "/result/gradient.txt", "w") as text_file:
        for i, elt in enumerate(average_gradient):
            text_file.write(str(elt))
            text_file.write("\n")

    report_value = calc_log_py_gvn_xavg(W_T_matrix, X_train, y_train, len(X_train))
    print("This is the value to be reported for 2.a",report_value)

    return report_value


def ref_optimize(x0, X_train, y_train,X_test, y_test , c, model_name):
    print("Optimizing w and T. Started.")

    start = time.time()
    result = opt.fmin_tnc(crf_obj, x0, crf_obj_gradient, (X_train, y_train, c), disp=1)
    print("Time taken: ", end='')
    print(time.time() - start)

    model = result[0]
    fmin_tnc_results[c] = model
    accuracy = crf_test(model, X_test, y_test, model_name, c)
    print('Accuracy for c = {}: {}'.format(c, accuracy))
    return accuracy

#2a
if __name__ == '__main__':
    gradient_compute(X_gradient_compute, y_gradient_compute,W_T_matrix)

#2b
    for c in [5]:
        ref_optimize(W_T_matrix, X_train, y_train,X_test, y_test , c=c, model_name='c_equal_'+ str(c))


    
    
