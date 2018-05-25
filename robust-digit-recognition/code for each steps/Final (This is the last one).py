
# coding: utf-8

# In[2]:

from  sklearn.datasets  import fetch_mldata
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from scipy.io import loadmat
import numpy as np
import torch.utils.data as utils
import csv
import skorch
from  skorch.net  import NeuralNetClassifier

#process the data
data_path = "./data.mat"
data_raw = loadmat(data_path)

train_img = data_raw["train_img"]
test_img = data_raw["test_img"]
train_lbl = data_raw["train_lbl"]

##############################################
#add noise to the training data
###############################################
#randomly split data into three subsets.
def split(matrix, percent1, percent2, percent3):
    rows = matrix.shape[0]
    end1 = int(rows*percent1/100)
    end2 = end1 + int(rows*percent2/100)
    subset1 = matrix[:end1]
    subset2 = matrix[end1:end2]
    subset3 = matrix[end2:]
    return subset1, subset2, subset3
#split the training set 

pattern1 = train_img
pattern2 = train_img[:25000]

pattern1_lbl = train_lbl
pattern2_lbl = train_lbl[:25000]
#function for applying pattern 1 
def pattern_1(matrix):
    size = matrix.size
    temp = matrix.reshape(size,)
    temp[:50] = 0
    temp[50:] = 255
    temp = np.random.permutation(temp)
    output = temp.reshape(10,10)
    return output


for i in range(50000):
    #apply pattern 1 to pattern1 dataset
    re = pattern1[i].reshape(28,28)
    matrix_center = np.random.randint(9,19)
    a = matrix_center-5
    b = matrix_center+5
    re[a:b,a:b] = pattern_1(re[a:b,a:b])
    if(i < 25000):
        re_shape = pattern2[i].reshape(28,28)
        mu, sigma = 0, 100 # mean and standard deviation
        s = np.random.normal(mu, sigma,(20,20))
        s = s.astype(int)
        center = re_shape[4:24,4:24] +s
        for i in range(20):
            for j in range(20):
                if(center[i][j]<0):
                    center[i][j] = 0
                elif(center[i][j]>255):
                    center[i][j] = 255
        re_shape[4:24,4:24] = center

        
print("already add noise")
train_img=np.concatenate((train_img,pattern1,pattern2), axis=0)
train_lbl = np.concatenate((train_lbl,pattern1_lbl,pattern2_lbl), axis=0)
##############################################
#finish to process the training data
###############################################

X = train_img.astype('float32')
test = test_img.astype('float32')
y =train_lbl.astype('int64')
y = y.reshape(125000,)
X /= 255.0  
test /= 255.0 
XCnn = X.reshape(-1, 1, 28, 28)
test = test.reshape(-1, 1, 28, 28)
XCnn_train, XCnn_test, y_train, y_test = train_test_split(XCnn, y, test_size=0.25, random_state=42)

#build the cnn module
class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1600, 128) # 1600 = number channels * width * height
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3)) # flatten over channel, height and width = 1600
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.softmax(x, dim=-1)
        return x
    
cnn = NeuralNetClassifier(
    Cnn,
    max_epochs=8,
    lr=1,
    optimizer=torch.optim.Adadelta,
    # device='cuda',  # uncomment this to train with CUDA
)
#train the module
cnn.fit(XCnn_train, y_train);
#Use validation set to see the accuracy
cnn_pred = cnn.predict(XCnn_test)
print(np.mean(cnn_pred == y_test))
#predict the test set
cnn_pred_test = cnn.predict(test)


# In[80]:

#write to .csv file
ID = np.arange(1,20001)
ID = ID.tolist()
data = zip(ID,cnn_pred_test)
with open('CNN_v6.csv', 'w',newline='') as outfile:
    mywriter = csv.writer(outfile)
    # manually add header

    mywriter.writerow(['ID', 'Prediction'])
    for d in data:
        mywriter.writerow(d)


# In[ ]:



