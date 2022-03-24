import numpy as np
import time
from scipy.optimize import check_grad
import os

#try either of these
# path = os.getcwd()

path = os.path.dirname(os.getcwd())

global_counter = 0

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

# to calculate alpha, considering from the lecture, alpha fwd and beta backward
def calculate_fwd_bkwd_msg(x, w, t):
    # initializing
    w_x = np.dot(x, w.T)
    t_transpose = t.transpose()
    length_word = len(w_x)
    forward_result = np.zeros((length_word, 26))

    # looping the characters in each word
    for i in range(1, length_word):
        alpha = forward_result[i - 1] + t_transpose
        alpha_max = np.max(alpha, axis=1)
        alpha_transposed = alpha.transpose()
        alpha = alpha_transposed - alpha_max
        alpha = alpha.transpose()
        r = np.log(np.sum(np.exp(alpha + w_x[i - 1]), axis=1))
        forward_result[i] = alpha_max + r

    # get the index of the last letter of the word
    last_index = len(w_x) - 1
    backward_result = np.zeros((len(w_x), 26))

    # reversing the array and looping from the reverse
    for i in range(last_index - 1, -1, -1):
        beta = backward_result[i + 1] + t
        beta_max = np.max(beta, axis=1)
        beta_transposed = beta.transpose()
        beta = beta_transposed - beta_max
        beta = beta.transpose()
        r = np.log(np.sum(np.exp(beta + w_x[i + 1]), axis=1))
        backward_result[i] = beta_max + r


    return forward_result, backward_result


def calc_nmrtr(y, x, w, t):
    #initialization
    result = 0
    w_dot_x = np.dot(x,w.T)

    for i in range(len(w_dot_x)):
        result = result + w_dot_x[i][y[i]]
        if i > 0:
            result = result + t[y[i-1]][y[i]]

    res_exp = np.exp(result)
    return res_exp


def calculate_denom(a, x, w):
    l = a[-1]
    r = np.dot(x,w.T)[-1]
    res = np.exp(l+r)
    return np.sum(res)


def compute_gradient_wrt_Wy(X, y, w, t, alpha, beta, dnm):
    len_x = len(X)
    w_x = np.dot(X, w.T)
    grad = np.zeros((26, 128))

    for i in range(len_x):
        result = (np.ones((26, 128)) * X[i] ).transpose()
        grad[y[i]] = grad[y[i]] + X[i]
        result = result*np.exp(alpha[i] + beta[i] + w_x[i])
        result /= dnm
        grad = grad - result.transpose()

    grad = grad.flatten()
    return grad


def compute_gradient_wrt_Tij(y, x, w, t, alpha, beta, dnm):
    w_x = np.dot(x, w.T)
    grad = np.zeros(26 * 26)

    # calculate_the_gradients
    for i in range(len(w_x)-1):
        for j in range(26):
            sum = w_x[i] + t.transpose()[j] + w_x[i+1][j] + beta[i+1][j] + alpha[i]
            e_sum = np.exp(sum)
            grad[j*26:(j+1)*26] = grad[j*26:(j+1)*26] - e_sum

    # dividing to normalize
    grad = grad/dnm

    for i in range(len(w_x)-1):
        t_i = y[i] + 26*y[i+1]
        grad[t_i] = grad[t_i] + 1

    return grad

def grad_word(X, y, w, t, word_i):
    forward_message, backward_message = calculate_fwd_bkwd_msg(X[word_i], w, t)
    den = calculate_denom(forward_message, X[word_i],w)
    grad_wrt_w = compute_gradient_wrt_Wy(X[word_i],y[word_i],w,t,forward_message,backward_message,den)
    grad_wrt_t = compute_gradient_wrt_Tij(y[word_i],X[word_i],w,t,forward_message,backward_message,den)
    return np.concatenate((grad_wrt_w, grad_wrt_t))

def calculate_log_py_gvn_x(x, w, y, t):
    f_mess, b_mess = calculate_fwd_bkwd_msg(x, w, t)
    return np.log(calc_nmrtr(y, x, w, t) / calculate_denom(f_mess, x, w))


def calc_log_py_gvn_xavg(w_t_list, X, y, num_examples):
    sum = 0
    w,t = extract_W_T(w_t_list)
    # print(w,t)
    # print(w.shape)

    for i in range(num_examples):
        r = calculate_log_py_gvn_x(X[i], w, y[i], t)
        sum = sum + r

    return sum / (num_examples)

def averaged_gradient(w_t_list, X, y, len_x):
    w,t = extract_W_T(w_t_list)
    #print(w,t)
    total = np.zeros(128*26+26**2)
    #print(w.shape, t.shape)

    for i in range(len_x):
        res = grad_word(X, y, w, t, i)
        #print(res, res.shape)
        total = total + res
    # print(total / (limit))
    return total / (len_x)

def check_gradient(w_t_list, X, y):
    # gradient of first 20 words
    grad_value = check_grad(calc_log_py_gvn_xavg, averaged_gradient, w_t_list, X, y, 20)
    print("Gradient limited to 20 words: ", grad_value)


def crf_obj(w_t, X, y, C):
    log_loss = calc_log_py_gvn_xavg(w_t, X, y, len(X))
    # log loss + regularizedloss
    result = -C*log_loss+ 1/2*np.sum(w_t**2)
    return result

def crf_obj_gradient(w_t, X, y, C):
    global global_counter
    global_counter = global_counter + 1
    ll_grad = averaged_gradient(w_t, X, y, len(X))
    print("Gradient evaluation counter - ",global_counter)
    return -C*ll_grad+w_t


def calc_word_char_acc(y_preds, ground_truth):
    # Intitializing word/letter counts
    count, c_count, count_l, c_count_l = 0, 0, 0, 0

    # Iterating through the predicted and actual values
    for y, y_hat in zip(ground_truth, y_preds):
        count = count + 1
        count_l = count_l + len(y_hat)
        c_count_l = c_count_l + np.sum(y_hat == y)
        if np.array_equal(y, y_hat):
            c_count = c_count + 1

    word_ratio = c_count / count
    letter_ratio = c_count_l / count_l
    return word_ratio, letter_ratio

def compute_energies(X, w, t): ####
    # acquiring the matrix shape
    result = np.zeros((len(X), 26))
    result[0] = np.inner(X[0], w)

    for row in range(1, len(X)):
        for letter in range(26):
            neg_num = -10000
            for prev_letter in range(26):
                a = np.inner(X[row], w[letter])
                t_r = result[row - 1][prev_letter] + t[prev_letter][letter] + a
                if t_r > neg_num:
                    neg_num = t_r
            result[row][letter] = neg_num
    return result
    
#....
#get the max from energies
def get_word(X, w, t): ####
    #val = 1e-6
    val = 1e-5
    result_from_e = compute_energies(X, w, t)
    #print("here")
    position = len(result_from_e)-1
    #print(position)
    previous_position = position-1
    letter = np.argmax(result_from_e[position]) #get the max from energies
    cur_val = result_from_e[position][letter]
    result = [letter]

    while position:
        for prev in range(26):
            e = np.inner(X[position], w[letter])
            flag = np.isclose(cur_val - result_from_e[previous_position][prev] - t[prev][letter] - e, 0, rtol=val)
            if flag:
                # print("now here")
                letter = prev
                position = position-1
                previous_position = previous_position-1
                cur_val = result_from_e[position][letter]
                result.append(prev)
                break

    result = result[::-1]
    to_return = np.array(result)
    return to_return


def decode_words_and_get_preditcions(X, w, t):
    y_hat = []

    for x in X:
        #print(x)
        word_pred = get_word(x, w, t)

        #print(word_pred), "here")
        #print(word_pred.shape)
        y_hat.append(word_pred)

    return y_hat


def crf_test(model, X_test, y_test,model_name, C):
    print("Function value: ", crf_obj(model, X_test, y_test, C))

    ''' accuracy '''

    w,t = extract_W_T(model)

    #save it to a file
    with open(path + "/result/" + model_name + ".txt", "w") as text_file:
        for i, elt in enumerate(model):
            text_file.write(str(elt) + "\n")

    #x_test = convert_word_to_character_dataset(X_test)
    y_preds = decode_words_and_get_preditcions(X_test, w, t)
    #y_preds = convert_character_to_word_dataset(y_preds, y_test)

    with open(path + "/result/prediction.txt", "w") as text_file:
        for i, elt in enumerate(y_preds):
            # convert to characters
            for word in elt:
                text_file.write(str(word + 1))
                text_file.write("\n")

    accuracy = calc_word_char_acc(y_preds, y_test)
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