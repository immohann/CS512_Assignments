**CS512 - Lab3 - Adversarial Training on Sequence Classification**

Group Members :
Amrit Raj vardhan
Harsh Mishra
Karan Jogi
Manmohan Dogra
Nikita Thakur

**Hyperparameters:**
batch size = 64
hidden size = 50
basic epoch = 100
out channels = 12
kernel size = 5
stride = 2
lr = 0.01
weight decay = 0.0001


**##Q2 - Training the Basic Model**

-The basic LSTM model has been defined in classifier.py.
-The forward function uses mode = 'plain' for this model.
-Training of the model can be done by running training.py.

-After training, the model has been saved as LSTM.pth in "/code_data"
-Accuracy vs Epoch trend for train and test data have been included in the report and in "/Figures/BasicModel.png'

**##Q3 -  Save and Load Pretrained Model**

- The code for saving a model after it has been trained and then loading it again using the torch module has been implemented in training.py
- Models have been saved in "/code_data"



