import numpy as np
import time
from scipy.optimize import check_grad
import os

path = os.getcwd()

#returns trained model parameters
def load_model_txt(filename):
    w_t = []
    with open(filename, 'r') as f:
        for i, parameters in enumerate(f):
            w_t.append(float(parameters))

    return np.array(w_t)

#matricizes W and T
def extract_W_T(w_t_list):
    W = np.zeros((26, 128))
    T = np.zeros((26, 26))
    
    index = 0

    for i in range(26):
        W[i] = w_t_list[128 * i: 128 * (i + 1)]
        for j in range(26):
            T[j][i] = w_t_list[128 * 26 + index]
            index += 1    
    
    return W,T


def compute_forward_message(x, w, t):
    w_x = np.dot(x,w.T)
    num_words = len(w_x)
    M = np.zeros((num_words, 26))

    # iterate through all characters in each word
    for i in range(1, num_words):
        alpha = M[i - 1] + t.transpose()
        alpha_max = np.max(alpha, axis=1)
        # prepare V - V.max()
        alpha = (alpha.transpose() - alpha_max).transpose()
        M[i] = alpha_max + np.log(np.sum(np.exp(alpha + w_x[i - 1]), axis=1))

    return M


def compute_backward_message(x, w, t):
    # get the index of the final letter of the word
    w_x = np.dot(x,w.T)
    fin_index = len(w_x) - 1
    M = np.zeros((len(w_x), 26))

    for i in range(fin_index - 1, -1, -1):
        beta = M[i + 1] + t
        beta_max = np.max(beta, axis=1)
        # prepare V - V.max()
        beta = (beta.transpose() - beta_max).transpose()
        M[i] = beta_max + np.log(np.sum(np.exp(beta + w_x[i + 1]), axis=1))

    return M


def compute_numerator(y, x, w, t):
    w_x = np.dot(x,w.T)
    sum_ = 0
    # for every word
    for i in range(len(w_x)):
        sum_ += w_x[i][y[i]]

        if (i > 0):
            # t stored as T{current, prev}
            sum_ += t[y[i - 1]][y[i]]

    return np.exp(sum_)


def compute_denominator(alpha, x, w):
    # forward propagate to the end of the word and return the sum
    return np.sum(np.exp(alpha[-1] + np.dot(x,w.T)[-1]))


def compute_gradient_wrt_Wy(X, y, w, t, alpha, beta, denominator):
    gradient = np.zeros((26, 128))

    w_x = np.dot(X,w.T)

    for i in range(len(X)):
        gradient[y[i]] += X[i]

        # for each position, reduce the probability of the character
        temp = np.ones((26, 128)) * X[i]
        temp = temp.transpose()
        temp = temp * np.exp(alpha[i] + beta[i] + w_x[i]) / denominator

        gradient -= temp.transpose()

    return gradient.flatten()


def compute_gradient_wrt_Tij(y, x, w, t, alpha, beta, denominator):
    gradient = np.zeros(26 * 26)
    w_x = np.dot(x,w.T)
    for i in range(len(w_x) - 1):
        for j in range(26):
            gradient[j * 26: (j + 1) * 26] -= np.exp(w_x[i] + t.transpose()[j] + w_x[i + 1][j] + beta[i + 1][j] + alpha[i])

    # normalize the gradient
    gradient /= denominator

    # add the gradient for the next word
    for i in range(len(w_x) - 1):
        t_index = y[i]
        t_index += 26 * y[i + 1]
        gradient[t_index] += 1

    return gradient

def gradient_word(X, y, w, t, word_index):
    # O(n * |Y|)
    # w_x = np.dot(X[word_index], w.T)
    # O(n * |Y|^2)
    f_mess = compute_forward_message(X[word_index], w, t)
    # O(n * |Y|^2)
    b_mess = compute_backward_message(X[word_index], w, t)
    # O(1)
    den = compute_denominator(f_mess, X[word_index], w)
    # O(n * |Y|^2)
    wy_grad = compute_gradient_wrt_Wy(X[word_index], y[word_index], w, t, f_mess, b_mess, den)
    # O(n * |Y|^2)
    t_grad = compute_gradient_wrt_Tij(y[word_index], X[word_index], w, t, f_mess, b_mess, den)
    return np.concatenate((wy_grad, t_grad))

def compute_log_p_y_given_x(x,w, y, t, word_index):
    f_mess = compute_forward_message(x, w, t)
    return np.log(compute_numerator(y, x, w, t) / compute_denominator(f_mess, x, w))


def compute_log_p_y_given_x_avg(w_t_list, X, y, limit):

    w,t = extract_W_T(w_t_list)

    total = 0
    for i in range(limit):
        # w_x = np.dot(X[i], w)
        total += compute_log_p_y_given_x(X[i],w, y[i], t, i)

    return total / (limit)

def averaged_gradient(w_t_list, X, y, limit):
    
    w,t = extract_W_T(w_t_list)

    total = np.zeros(128 * 26 + 26 ** 2)
    for i in range(limit):
        total += gradient_word(X, y, w, t, i)
    # print(total / (limit))
    return total / (limit)

def check_gradient(w_t_list, X, y):
    # check the gradient of the first 10 words
    grad_value = check_grad(compute_log_p_y_given_x_avg, averaged_gradient, w_t_list,  X, y, 10)
    print("Gradient check (first 10 characters) : ", grad_value)


def crf_obj(params, X, y, C):
    num_examples = len(X)
    l2_regularization = 1 / 2 * np.sum(params ** 2)
    log_loss = compute_log_p_y_given_x_avg(params, X, y, num_examples)
    return -C * log_loss + l2_regularization

def crf_obj_gradient(params, X, y, C):
    num_examples = len(X)
    logloss_gradient = averaged_gradient(params, X, y, num_examples)
    l2_loss_gradient = params
    return -C * logloss_gradient + l2_loss_gradient


def compute_word_char_accuracy_score(y_preds, y_true):
    word_count = 0
    correct_word_count = 0
    letter_count = 0
    correct_letter_count = 0

    for y_t, y_p in zip(y_true, y_preds):
        word_count += 1
        if np.array_equal(y_t, y_p):
            correct_word_count += 1

        letter_count += len(y_p)
        correct_letter_count += np.sum(y_p == y_t)

    return correct_word_count / word_count, correct_letter_count / letter_count

def get_energies(X, w, t):
    # for 100 letters I need an M matrix of 100  X 26
    M = np.zeros((len(X), 26))

    # populates first row
    M[0] = np.inner(X[0], w)

    # go row wise through M matrix, starting at line 2 since first line is populated
    for row in range(1, len(X)):

        # go column wise, populating the best sum of the previous + T[previous letter][
        for cur_letter in range(26):
            # initialize with giant negative number
            best = -np.inf

            # iterate over all values of the previous letter, fixing the current letter
            for prev_letter in range(26):
                temp_product = M[row - 1][prev_letter] + np.inner(X[row], w[cur_letter]) + t[prev_letter][cur_letter]
                if (temp_product > best):
                    best = temp_product
            M[row][cur_letter] = best
    return M
    
def decode_crf_word(X, w, t):
    M = get_energies(X, w, t)

    cur_word_pos = len(M) - 1
    prev_word_pos = cur_word_pos - 1

    cur_letter = np.argmax(M[cur_word_pos])
    cur_val = M[cur_word_pos][cur_letter]

    solution = [cur_letter]

    while (cur_word_pos > 0):
        for prev_letter in range(26):
            energy = np.inner(X[cur_word_pos], w[cur_letter])
            if (np.isclose(cur_val - M[prev_word_pos][prev_letter] - t[prev_letter][cur_letter] - energy, 0,
                           rtol=1e-5)):
                solution.append(prev_letter)
                cur_letter = prev_letter
                cur_word_pos -= 1
                prev_word_pos -= 1
                cur_val = M[cur_word_pos][cur_letter]
                break

    solution = solution[::-1]  # reverse the prediction string
    return np.array(solution)


def decode_crf(X, w, t):
    y_pred = []
    for i, x in enumerate(X):
        preds = decode_crf_word(x, w, t)
        y_pred.append(preds)
    return y_pred


def crf_test(model, X_test, y_test,model_name, C):
    print("Function value: ", crf_obj(model, X_test, y_test, C))

    ''' accuracy '''

    w,t = extract_W_T(model)

    #save it to a file
    with open(path + "/result/" + model_name + ".txt", "w") as text_file:
        for i, elt in enumerate(model):
            text_file.write(str(elt) + "\n")

    #x_test = convert_word_to_character_dataset(X_test)
    y_preds = decode_crf(X_test, w, t)
    #y_preds = convert_character_to_word_dataset(y_preds, y_test)

    with open(path + "/result/prediction.txt", "w") as text_file:
        for i, elt in enumerate(y_preds):
            # convert to characters
            for word in elt:
                text_file.write(str(word + 1))
                text_file.write("\n")

    accuracy =  compute_word_char_accuracy_score(y_preds, y_test)
    print("Test accuracy : ",accuracy)

    return accuracy

#function for evaluation of linear SVM performance
def linear_SVM_performance(y_target, y_predicted, target_ids):
    target_words, predicted_words = [], []
    lastWord = -1000
    character_truepositives, word_truepositives = 0, 0

    for i, (target, predicted) in enumerate(zip(y_target, y_predicted)):
        target_word = int(target_ids[i])

        if target_word == lastWord:
            target_words[-1].append(target)
            predicted_words[-1].append(predicted)
        else:
            target_words.append([target])
            predicted_words.append([predicted])
            lastWord = target_word


    for target_character,predicted_character in zip(y_target, y_predicted):
        if target_character == predicted_character:
            character_truepositives += 1

    Character_Accuracy = float(character_truepositives)/float(y_target.shape[0])

    for target_word,predicted_word in zip(target_words, predicted_words):
        if np.array_equal(target_word, predicted_word):
            word_truepositives += 1

    word_Accuracy = float(word_truepositives)/float(len(target_words))

    print("Letter-wise prediction Accuracy: %0.3f" %(Character_Accuracy))
    print("Word-wise prediction Accuracy: %0.3f" %(word_Accuracy))

    return Character_Accuracy, word_Accuracy


#function for evaluation of Structured SVM performance
def structuredSVM_performance(file_target, file_predicted):

    with open(file_target, 'r') as file_target, open(file_predicted, 'r') as file_predicted:
        target_words, predicted_words, target_characters, predicted_characters = [], [], [], []

        lastWord = -1000
        character_truepositives, word_truepositives = 0, 0

        for target, predicted in zip(file_target, file_predicted):
            targets = target.split()
            target_character = int(targets[0])
            target_characters.append(target_character)

            predicted_character = int(predicted)

            if hasattr(predicted_character, 'len') > 0:
                predicted_character = predicted_character[0]

            predicted_characters.append(predicted_character)
            target_Word = int(targets[1][4:])

            if target_Word == lastWord:
                target_words[-1].append(target_character)
                predicted_words[-1].append(predicted_character)
            else:
                target_words.append([target_character])
                predicted_words.append([predicted_character])
                lastWord = target_Word

        for target_character, predicted_character in zip(target_characters, predicted_characters):
            if target_character == predicted_character:
                character_truepositives += 1

        for target_Word, pred_word in zip(target_words, predicted_words):
            if np.array_equal(target_Word, pred_word):
                word_truepositives += 1

        character_accuracy = float(character_truepositives) / float(len(target_characters))
        word_accuracy = float(word_truepositives) / float(len(target_words))

        print("Character level accuracy : %0.3f" % (character_accuracy))
        print("Word level accuracy : %0.3f" % (word_accuracy))

        return character_accuracy, word_accuracy