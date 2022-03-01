# Imports
import numpy as np
import os

import scipy.optimize as opt
from read_data import read_for_crf, prepare_structured_dataset
from all_functions import *

path = os.path.dirname(os.getcwd())

#for 2a   #Train data to calculate the gradient on
X_gradient_compute, y_gradient_compute = read_for_crf(path + "/data/train.txt")

#Model.txt contains the weights for W and T. W_T_matrix is the list containing both W and T.
W_T_matrix = load_model_txt(path + "/data/model.txt")


#for 2b
X_train, y_train = prepare_structured_dataset(path + "/data/train_struct.txt")    
X_test, y_test = prepare_structured_dataset(path + "/data/train_struct.txt")

def gradient_compute(X_train, y_train,W_T_matrix):
    """Compute the CRF objective and gradient on the list of words (word_list)
    evaluated at the current model x (w_y and T, stored as a vector)
    """
    # print(path)

 
    check_gradient(W_T_matrix, X_train, y_train)
    # print(X_train)

    start = time.time()
    average_gradient = averaged_gradient(W_T_matrix, X_train, y_train, len(X_train))
    print("Total time:", time.time() - start)


    with open("gradient.txt", "w") as text_file:
        for i, elt in enumerate(average_gradient):
            text_file.write(str(elt))
            text_file.write("\n")

    report_value = compute_log_p_y_given_x_avg(W_T_matrix, X_train, y_train, len(X_train))
    print("This is the value to be reported for 2.a",report_value)

    return report_value

#crf_obj  - train_crf ttitto
#xo is params
def ref_optimize(x0, X_train, y_train,X_test, y_test , c, model_name):
    print("Optimizing parameters. This will take a long time (at least 1 hour per model).")

    start = time.time()
    result = opt.fmin_tnc(crf_obj, x0, crf_obj_gradient, (X_train, y_train, c), disp=1)[0]
    print("Total time: ", end='')
    print(time.time() - start)

    model = result[0]

    accuracy = crf_test(model, X_test, y_test, model_name)
    print('CRF test accuracy for c = {}: {}'.format(c, accuracy))
    return accuracy

#2a
gradient_compute(X_gradient_compute, y_gradient_compute,W_T_matrix)
#2b
ref_optimize(W_T_matrix, X_train, y_train,X_test, y_test , c=1000, model_name='new')



    
    
