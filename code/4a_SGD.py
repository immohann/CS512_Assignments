from all_functions import *
from read_data import *
import numpy as np
from numpy.random import rand
import os

path = os.path.dirname(os.getcwd())

X_train, y_train = prepare_structured_dataset(path + "/data/train_struct.txt") 
W_T_matrix = load_model_txt(path + "/data/model.txt")
X_test, y_test = prepare_structured_dataset(path + "/data/train_struct.txt")

GD_results = {}

def gradient_descent(objective, derivative, random_WT, X_train, y_train, c, n_iter, step_size):

	# run the gradient descent
    for i in range(n_iter):
        # calculate gradient

        #Mini-batch size
        M = 1500
        batch_indexes = np.random.randint(0,len(X_train),M)
        # print(random_WT)

        X_train_Batch = []
        y_train_Batch = []

        for j in batch_indexes:
            X_train_Batch.append(X_train[j])
            y_train_Batch.append(y_train[j])

        gradient = derivative(random_WT, X_train_Batch, y_train_Batch, c)
        # take a step
        random_WT = random_WT - step_size * gradient
        # evaluate candidate point
        solution_eval = objective(random_WT, X_train_Batch, y_train_Batch, c)
        # report progress
        print('>%d f(%s) = %.5f' % (i, random_WT, solution_eval))
        model_name = "GD_1000"
        crf_test(random_WT, X_train_Batch, y_train_Batch, model_name, c)

    return [random_WT, solution_eval]



# define range for input
random_WT = np.random.rand(len(W_T_matrix))

# define the total iterations
n_iter = 30
# define the step size
step_size = 0.001

#Function value:  7579.510619407869
# Test accuracy :  (0.23327515997673065, 0.7317073170731707)
# step size 0.001

# Function value:  15838.017756828234
# Test accuracy :  (0.013379872018615475, 0.42665587793318693)
# step size 0.0001


# perform the gradient descent search
c = 1000
model_name = "GD_1000"
best, score = gradient_descent(crf_obj, crf_obj_gradient, random_WT, X_train, y_train, c, n_iter, step_size)
print('Done!')
print('f(%s) = %f' % (best, score))

GD_results[c] = best
accuracy = crf_test(best, X_test, y_test, model_name, c)
# print('CRF test accuracy for c = {}: {}'.format(c, accuracy))