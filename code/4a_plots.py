from cProfile import label
from turtle import position
import matplotlib.pyplot as plt
import numpy as np
import os

path = os.getcwd()

lbfgs_function_values = []
mini_SGD_mom_function_values = []
mini_batch_function_values = []
lbfgs_word_acc =[]
mini_SGD_mom_word_acc =[]
mini_SGD_word_acc =[]

with open(path + '/result/4a_output_for_plot/lbfgs_plot_data.txt','r') as f:
    lbfgs_lines = f.readlines()

with open(path + '/result/4a_output_for_plot/mini_batch_SGD_mom.txt','r') as f:
    SGD_mom_lines = f.readlines()

with open(path + '/result/4a_output_for_plot/mini_batch_SGD.txt','r') as f:
    SGD_mini_lines = f.readlines()

for line in lbfgs_lines:
    
    if line[0:17] == "Function value:  ":
        lbfgs_function_values.append(float(line[17:-2]))
    if line[0:17] == "Test accuracy :  ":
        lbfgs_word_acc.append(1.0 - float(line[18:line.index(',')]))

for i in range(3):
    lbfgs_function_values.append(lbfgs_function_values[-1])
    lbfgs_word_acc.append(lbfgs_word_acc[-1])

for line in SGD_mom_lines:
    
    if line[0:17] == "Function value:  ":
        mini_SGD_mom_function_values.append(float(line[17:-2]))
    if line[0:17] == "Test accuracy :  ":
        mini_SGD_mom_word_acc.append(1.0 - float(line[18:line.index(',')]))

for line in SGD_mini_lines:
    
    if line[0:17] == "Function value:  ":
        mini_batch_function_values.append(float(line[17:-2]))
    if line[0:17] == "Test accuracy :  ":
        mini_SGD_word_acc.append(1.0 - float(line[18:line.index(',')]))


x_label = np.arange(0,len(lbfgs_function_values))


#raining Object value vs Effective Number of passes
plt.plot(x_label,lbfgs_function_values, label = 'LBFGS')
plt.plot(x_label,mini_SGD_mom_function_values, label = 'Momentum')
plt.plot(x_label,mini_batch_function_values, label = 'SGD')
plt.title('Training Objective value vs Effective Number of passes', fontsize=14)
plt.xlabel("Effective No. of Passes")
plt.ylabel("Training Objective Value")
plt.legend(loc = 'best')
plt.show()

#Test word wise error vs Effective no. of passes
plt.plot(x_label,lbfgs_word_acc, label = 'LBFGS')
plt.plot(x_label,mini_SGD_mom_word_acc, label = 'Momentum')
plt.plot(x_label,mini_SGD_word_acc, label = 'SGD')
plt.title('Test word wise error vs Effective number of passes', fontsize=14)
plt.xlabel("Effective No. of Passes")
plt.ylabel("Test Word-Wise Error")
plt.legend(bbox_to_anchor=(1,0.2),loc = 'lower right')
plt.show()
