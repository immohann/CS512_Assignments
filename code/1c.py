import numpy as np
import string
import math
import os

path = os.path.dirname(os.getcwd())


decode_input_tolist = np.genfromtxt(path + "/data/decode_input.txt", delimiter = ' ')

#each letter
X = decode_input_tolist[:100*128]
#node weight
W = (decode_input_tolist[100*128:100*128+26*128]).reshape((26, 128))
#edge weight
T = (decode_input_tolist[100*128+26*128:]).reshape((26, 26)).transpose() 


m = int(len(X)/128)           # length of the word 
X = np.reshape(X, (m, 128))

def node_weight(s,i) : # Node potential for < Wys = i, X^t_s >
    return np.dot(W[i,:],X[s,:])  
def edge_weight(i,j) : # Edge potential for Tys =i, ys+1 = j
    return T[i,j] 

#comments to be removed, function name change kar diya hai 


# here we calculates all l1(y1) to l_m-1(y_m-1) using equations 27 and 28
# be carefule that the last node does not take account the edge potential 
# and therefore does not involve g function (which is T)
l = np.zeros((m, 26)) # storing all l's where rows represents 1, 2 ,,, s, ... m letter images; and cols 1...26

def decoder_function(X, W, T): 
    #Recursive for max-sum
    for s in range(1, m) :        # s taking 1 to m-1; where s: sth letter in X^t matrix
        for j in range(26) :      # y_s+1 = j
            temp = []
            for i in range(26) :  # y_s = i
                temp.append(node_weight(s-1,i) + edge_weight(i,j) + l[s-1,i]) # calculate equation 28 till l_m-1(y_m-1)
            l[s,j] = max(temp) 
    l_m = []                      
    for i in range(26):      
        l_m = np.append(l_m, [node_weight(m-1,i) + l[m-1,i]])

    print("The maximum objective value is", np.amax(l_m))
    
    y_pred = np.zeros((m), dtype = int)
    y_pred[m-1] = np.argmax(l_m)     # ym^*

    # Recovery step
    for s in range(m-2, -1, -1):
        temp1 = []
        for i in range(26):
            temp1.append(node_weight(s, i) + edge_weight(i, y_pred[s+1]) + l[s, i])
        y_pred[s] = np.argmax(temp1)
    #re-index the labels
    return y_pred + 1   


decode_output = decoder_function(X, W, T)

np.savetxt(path + "/result/decode_output.txt", decode_output, fmt = '%i')






