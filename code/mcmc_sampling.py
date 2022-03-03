import numpy as np
from read_data import *
from all_functions import *
import copy

path = os.path.dirname(os.getcwd())

def sampler(X_train, W, T, num_samples):

        index = np.random.randint(0, len(X_train)-1)
        m = len(X_train[index])

        initial_sample_letters = np.random.randint(0, 26, m)
        samples = []
        probability = np.zeros(26)
        sample = copy.deepcopy(initial_sample_letters)
        for k in range(num_samples):
            rand_index = np.random.randint(0, m-1)
            for j in range(26):
                energies = np.inner(X_train[index][rand_index], W[j])
                if rand_index == 0:
                    probability[j] = np.exp(energies + T[j][sample[rand_index + 1]])
                elif rand_index == m - 1:
                    probability[j] = np.exp(energies + T[sample[rand_index - 1]][j])
                else:
                    probability[j] = np.exp(energies + T[sample[rand_index - 1]][j] + T[j][sample[rand_index + 1]])

            # print("probability",probability)
            max_probability = np.argmax(probability)
            # print("max_probability",max_probability)
            sample[rand_index] = max_probability

            # print("sample",sample)
            samples.append(copy.deepcopy(sample))

        return (samples, index)


X_gradient_compute, y_gradient_compute = read_for_crf(path + "/data/train.txt")
W_T_matrix = load_model_txt(path + "/data/model.txt")

w,t = extract_W_T(W_T_matrix)

y_sampled_batch, index = sampler(X_gradient_compute,w,t,10)

# print("y_sampled_batch",y_sampled_batch)

total_counter = np.zeros((len(y_sampled_batch[0]),26))
# print(total_counter)

# print(total_counter.shape)
# # x1 , x2 , x3 , x4
# print(total_counter)

#MCMC sampling
for samples in y_sampled_batch:
    
    for yi in range(len(samples)):  #0 - 9
        
        # print("yi",yi)
        # print("samples[yi]",samples[yi])

        total_counter[yi][int(samples[yi])] += 1

        # print("total_counter[yi][int(samples[yi])]",total_counter[yi][int(samples[yi])])
        # yi = yi + 1
        # print("counter_list after each sample",total_counter)


# print("total_counter end",total_counter)

final_sample = []
for every in total_counter:
    final_sample.append(np.argmax(every))

print("final_sample",type(final_sample))

y_train_array = [final_sample]

# calculate gradient
x_train_array = [X_gradient_compute[index]]



print(type(y_gradient_compute))

print(np.array(final_sample))

# crf_obj_gradient(W_T_matrix,x_train_array,,1000)

