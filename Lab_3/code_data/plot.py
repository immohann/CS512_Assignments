import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


path = "../Data/"

plain_model_epoch = 100 
Adv_epoch = 100
Prox_epoch = 100

test_accuracy_plain = []
train_accuracy_plain = []
adv_model_test_acc = []
prox_model_test_acc = []

# with open(path + "BasicModel_train.txt", "r") as f:
#   for line in f:
#     train_accuracy_plain.append(float(line.strip()))

# with open(path + "BasicModel_test.txt", "r") as f:
#   for line in f:
#     test_accuracy_plain.append(float(line.strip()))

# iterations = [x for x in range(plain_model_epoch)]

# plt.figure(figsize=(12,10))
# plt.plot(iterations, train_accuracy_plain, color = 'purple', linewidth = 2, label='Train')
# plt.plot(iterations, test_accuracy_plain, color = 'red', linewidth = 2, label='Test')
# plt.legend(loc = "upper left")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.title("Accuracy vs Epochs for Plain Model.")
# plt.savefig(path + 'train_test_plain_model.png')


#For question 4
with open(path + "AdvModel_acc.txt", "r") as f:
  for line in f:
    adv_model_test_acc.append(float(line.strip()))

#For question 5
with open(path + "ProxModel_acc.txt", "r") as f:
  for line in f:
    prox_model_test_acc.append(float(line.strip()))


def plot_for_q4_5(epoch, short_epoch, test_ep0, test_ep1, test_ep2, test_ep3, test_ep4, ep0,ep1, ep2, ep3,ep4, file_name, question):

    epoch= [x for x in range(epoch)]
    short_epoch = [x_ep for x_ep in range(short_epoch)]

    path = "../Data/"
    

    plt.figure(figsize=(12,10))
    plt.plot(epoch, test_ep0, color='purple', linewidth=2, label="Optimal Epsilon = " + str(ep0))
    plt.plot(short_epoch, test_ep1, color = 'red', linewidth = 2, label=ep1)
    plt.plot(short_epoch, test_ep2, color = 'blue', linewidth = 2, label=ep2)
    plt.plot(short_epoch, test_ep3, color = 'black', linewidth = 2, label=ep3)
    plt.plot(short_epoch, test_ep4, color = 'green', linewidth = 2, label=ep4)
    plt.legend(loc = "lower right")

    plt.xlabel("Epochs")
    plt.ylabel("Test Accuracy")
    
    if question == 4:
        plt.title("Accuracy vs Epochs for AdvLSTM Model.")
    else:
        plt.title("Accuracy vs Epochs for ProxLSTM Model.")
 
    plt.savefig(path + file_name)

# [0.01, 0.05, 0.1, 1.0, 5.0]


# plot_for_q4_5(epoch = Adv_epoch, short_epoch = 50, test_ep0 = adv_model_test_acc[:50], 
#     test_ep1 = adv_model_test_acc[100:100 + 100], test_ep2 = adv_model_test_acc[200:200 + 50], test_ep3 = adv_model_test_acc[300:300+50],
#     test_ep4 = adv_model_test_acc[400:400+50], 
#     ep0 = 0.01, ep1 = 0.05, ep2 = 0.1, ep3 = 1.0,ep4 = 5.0, file_name = 'Accuracy_AdvModel.png', question= 4)

plot_for_q4_5(epoch = Prox_epoch, short_epoch = 50, test_ep0 = prox_model_test_acc[:100], 
    test_ep1 = prox_model_test_acc[100:100 + 50], test_ep2 = prox_model_test_acc[200:200 + 50], test_ep3 = prox_model_test_acc[300:300+50],
    test_ep4 = prox_model_test_acc[400:400+50], 
    ep0 = 0.01, ep1 = 0.05, ep2 = 0.1, ep3 = 1.0,ep4 = 5.0, file_name = 'Accuracy_ProxModel.png', question= 5)
