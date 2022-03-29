
import importlib.util
from re import L
import sys
import data_loader

import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from torch.nn import functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from data_loader import get_dataset
import torch.optim as optim
from torch.optim import Adam, SGD, LBFGS
import numpy as np


#leNet architecture
import torch.nn as nn
import torch.nn.functional as F

#word accuracies function based on letters, required dataset.nextletter info
def wordaccuracies(pred,actual,dataset,split):
  incorrectwords = 0
  totalwords = 0
  flag = True

  for i in range(len(pred)):

    if pred[i] != actual[i]:
      flag= False
    if dataset.nextletter[split+i] == -1:
      if flag == False:
        incorrectwords+=1
        flag  = True
      totalwords = totalwords+1

  wordaccuracy = 1 - incorrectwords/totalwords
  print("Word accuracy: ", wordaccuracy)
  return wordaccuracy
  print("\n")


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 3,padding=1)
        self.conv2 = nn.Conv2d(3, 3, 3,padding=1)
        self.fc1   = nn.Linear(24, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 26)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


#testing function
def testingmodel(test_loader,cuda,lenetmodel,criterion,wtestingepoc,testingepoc,test,dataset,split):
    finalletteracc= 0 
    testpredictedletters,testactualletters=[],[]
    for i_batch, sample in enumerate(test_loader):
            
    #       print("Batch=", i_batch)
            test_X = sample[0]
            test_Y = sample[1]
    #       print(len(test_X))

            lastflag = (len(test_X)<256)
            if lastflag:
              test_X= test_X.view(len(test_X),1,16,8)
              test_X= test_X.repeat(1,3,1,1)
              test_Y = test_Y.view(len(test_Y),26)
            else:
              test_X= test_X.view(256,1,16,8)
              test_X= test_X.repeat(1,3,1,1)
              test_Y = test_Y.view(256,26)
            
            
            if cuda:
                test_X = test_X.cuda()
                test_Y = test_Y.cuda()
            labels=  torch.max(test_Y, 1)[1]

            outputs = lenetmodel(test_X)
            loss = criterion(outputs,labels)

            running_loss = 0.0
            running_corrects = 0
            _, preds = torch.max(outputs, 1)
            testactualletters.extend(labels.tolist())
            testpredictedletters.extend(preds.tolist())
            running_loss += loss.item() * test_X.size(0)
            running_corrects += torch.sum(preds == (labels.data))
            
            epoch_loss = running_loss / len(test_Y)
            epoch_acc = running_corrects.double() / len(test_Y)
            finalletteracc = finalletteracc + len(test_Y)*epoch_acc
            #print("Letter accuracy =",epoch_acc)

    wtestingepoc.append(wordaccuracies(testpredictedletters,testactualletters,dataset,split))
    testingepoc.append(finalletteracc/len(test))
    print("Testing acc = :",finalletteracc/len(test) )


def main():
    batch_size = 256
    num_epochs = 10
    max_iters  = 1000
    print_iter = 25 # Prints results every n iterations
    conv_shapes = [[1,64,128]] #

    # Model parameters
    input_dim = 128
    embed_dim = 64
    num_labels = 26
    cuda = torch.cuda.is_available()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    #gettinig data using DataLoader class , modified code
    dataset = get_dataset()

        
    #model, loss, optimizers
    lenetmodel =  LeNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optim = Adam(lenetmodel.parameters())


    trainingepoc,testingepoc=[],[]
    wtrainingepoc,wtestingepoc=[],[]
    split = int(0.5 * len(dataset.data)) # train-test split
    train_data, test_data = dataset.data[:split], dataset.data[split:]
    train_target, test_target = dataset.target[:split], dataset.target[split:]
        # Convert dataset into torch tensors
    train = data_utils.TensorDataset(torch.tensor(train_data).float(), torch.tensor(train_target).long())
    test = data_utils.TensorDataset(torch.tensor(test_data).float(), torch.tensor(test_target).long())

    # Define train and test loaders
    train_loader = data_utils.DataLoader(train,  # dataset to load from
                                            batch_size=batch_size,  # examples per batch (default: 1)
                                            shuffle=True,
                                            sampler=None,  # if a sampling method is specified, `shuffle` must be False
                                            num_workers=5,  # subprocesses to use for sampling
                                            pin_memory=False,  # whether to return an item pinned to GPU
                                            )

    test_loader = data_utils.DataLoader(test,  # dataset to load from
                                            batch_size=batch_size,  # examples per batch (default: 1)
                                            shuffle=False,
                                            sampler=None,  # if a sampling method is specified, `shuffle` must be False
                                            num_workers=5,  # subprocesses to use for sampling
                                            pin_memory=False,  # whether to return an item pinned to GPU
                                            )

        
    for i in range(100):
        trainpredictedletters,trainactualletters=[],[]
        trainingacc= 0
        if i%1==0:
            print("\n  Processing epoch {}".format(i))
        

        for i_batch, sample in enumerate(train_loader):
            if i_batch%25==0:
                print("Batch=", i_batch)
            train_X = sample[0]
            train_Y = sample[1]

            lastflag = (len(train_X)<256)
            if lastflag:
                train_X= train_X.view(len(train_X),1,16,8)
                train_X= train_X.repeat(1,3,1,1)
                train_Y = train_Y.view(len(train_Y),26)
            else:
                train_X= train_X.view(256,1,16,8)
                train_X= train_X.repeat(1,3,1,1)
                train_Y = train_Y.view(256,26)
            
            if cuda:
                train_X = train_X.cuda()
                train_Y = train_Y.cuda()
            
            labels=  torch.max(train_Y, 1)[1]

            running_loss = 0.0
            running_corrects = 0

            optim.zero_grad()
            outputs = lenetmodel(train_X)
            loss = criterion(outputs, labels)
            loss.backward()


            optim.step()

            # outputs = lenetmodel(train_X)
            # loss = criterion(outputs, labels)

            print("Batch=", i_batch)
            _, preds = torch.max(outputs, 1)

            
            trainactualletters.extend(labels.tolist())
            trainpredictedletters.extend(preds.tolist())
            
            running_loss += loss.item() * train_X.size(0)
            running_corrects += torch.sum(preds == (labels).data)

            epoch_loss = running_loss / len(train_Y)
            epoch_acc = running_corrects.double() / len(train_Y)
            
            trainingacc = trainingacc + len(train_X)*epoch_acc
            if i_batch%25==0:
                print("Letter accuracy =",epoch_acc)

        wtrainingepoc.append(wordaccuracies(trainpredictedletters,trainactualletters,dataset,split)) 
        trainingepoc.append(trainingacc/len(train))
        print("Training acc = :",trainingacc/len(train))

        #testing
        testingmodel(test_loader,cuda,lenetmodel,criterion,wtestingepoc,testingepoc,test,dataset,split)


        f_trainingepoc = open("Adam_files/wordwise_training.txt", "a")
        f_trainingepoc.write(str(wtrainingepoc[i]) + "\n")
        f_trainingepoc.close()

        f_trainingepoc = open("Adam_files/letterwise_training.txt", "a")
        f_trainingepoc.write(str(trainingepoc[i]) + "\n")
        f_trainingepoc.close()
    
        f_wtestingepoc = open("Adam_files/wordwise_testing.txt", "a")
        f_wtestingepoc.write(str(wtestingepoc[i]) + "\n")
        f_wtestingepoc.close()

        f_testingepoc = open("Adam_files/letterwise_testing.txt", "a")
        f_testingepoc.write(str(testingepoc[i]) + "\n")
        f_testingepoc.close()

if __name__ == "__main__":
    main()