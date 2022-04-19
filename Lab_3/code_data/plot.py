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

with open(path + "BasicModel_train.txt", "r") as f:
  for line in f:
    train_accuracy_plain.append(float(line.strip()))

with open(path + "BasicModel_test.txt", "r") as f:
  for line in f:
    test_accuracy_plain.append(float(line.strip()))

iterations = [x for x in range(plain_model_epoch)]

plt.figure(figsize=(12,10))
plt.plot(iterations, train_accuracy_plain, color = 'purple', linewidth = 2, label='Train')
plt.plot(iterations, test_accuracy_plain, color = 'red', linewidth = 2, label='Test')
plt.legend(loc = "upper left")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epochs for Plain Model.")
plt.savefig(path + 'train_test_plain_model.png')


#For question 4
with open(path + "AdvModel_acc.txt", "r") as f:
  for line in f:
    adv_model_test_acc.append(float(line.strip()))

#For question 5
with open(path + "ProxModel_acc.txt", "r") as f:
  for line in f:
    prox_model_test_acc.append(float(line.strip()))


def plot_for_q4_5(epoch, epoch_epsilon, ep_opt, test_opt, test_ep1, test_ep2, test_ep3, ep1, ep2, ep3, file_name, question):

    x= [x for x in range(epoch)]
    x_ep = [x_ep for x_ep in range(epoch_epsilon)]

    path = "../Data/"
    

    plt.figure(figsize=(12,10))
    plt.plot(x, test_opt, color='purple', linewidth=2, label="Optimal Epsilon = " + str(ep_opt))
    plt.plot(x_ep, test_ep1, color = 'red', linewidth = 2, label=ep1)
    plt.plot(x_ep, test_ep2, color = 'blue', linewidth = 2, label=ep2)
    plt.plot(x_ep, test_ep3, color = 'black', linewidth = 2, label=ep3)
    plt.legend(loc = "upper left")

    plt.xlabel("Epochs")
    plt.ylabel("Test Accuracy")
    
    if question == 4:
        plt.title("Accuracy vs Epochs for AdvLSTM Model.")
    else:
        plt.title("Accuracy vs Epochs for ProxLSTM Model.")
 
    plt.savefig(path + file_name)


plot_for_q4_5(epoch = Adv_epoch, epoch_epsilon = 50, ep_opt = 0.08, test_opt = adv_model_test_acc[:100], 
    test_ep1 = adv_model_test_acc[100:100 + 50], test_ep2 = adv_model_test_acc[200:200 + 50], test_ep3 = adv_model_test_acc[300:300+50], 
    ep1 = 0.01, ep2 = 0.1, ep3 = 1.0, file_name = 'Accuracy_AdvModel.png', question= 4)

plot_for_q4_5(epoch = Prox_epoch, epoch_epsilon = 50, ep_opt = 0.08, test_opt = prox_model_test_acc[:100], 
    test_ep1 = prox_model_test_acc[100:100 + 50], test_ep2 = prox_model_test_acc[200:200 + 50], test_ep3 = prox_model_test_acc[300:300+50], 
    ep1 = 0.1, ep2 = 1.0, ep3 = 5.0, file_name = 'Accuracy_ProxModel.png', question= 5)
