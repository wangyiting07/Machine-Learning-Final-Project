
# coding: utf-8

# In[95]:

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


# In[96]:

torch.__version__


# In[97]:

data_path = "./data.mat"
data_raw = loadmat(data_path)

train_img = data_raw["train_img"]
test_img = data_raw["test_img"]
train_lbl = data_raw["train_lbl"]


# In[98]:

train_lbl.shape


# In[99]:

X = train_img.astype('float32')
test = test_img.astype('float32')
y =train_lbl.astype('int64')
y = y.reshape(50000,)


# In[100]:

X.shape


# In[101]:

X.max()


# In[102]:

X /= 255.0  
test /= 255.0  


# In[103]:

X.min(), X.max(),X.shape,y.shape


# In[104]:

X_train = X[:45000]
y_train = y[:45000]
X_test = X[45000:]
y_test = y[45000:]


# In[105]:

X_train.shape, y_train.shape


# In[106]:

from  skorch.net  import NeuralNetClassifier


# In[107]:

XCnn = X.reshape(-1, 1, 28, 28)
test = test.reshape(-1, 1, 28, 28)


# In[108]:

XCnn.shape


# In[109]:

XCnn_train = XCnn[:45000]
y_train = y[:45000]
XCnn_test = XCnn[45000:]
y_test = y[45000:]


# In[110]:

XCnn_train.shape, y_train.shape


# In[72]:

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


# In[79]:

cnn = NeuralNetClassifier(
    Cnn,
    max_epochs=10,
    lr=1,
    optimizer=torch.optim.Adadelta,
    # device='cuda',  # uncomment this to train with CUDA
)


# In[80]:

cnn.fit(XCnn_train, y_train);


# In[111]:

cnn_pred = cnn.predict(XCnn_test)


# In[112]:

np.mean(cnn_pred == y_test)


# In[113]:

cnn_pred_test = cnn.predict(test)


# In[114]:

cnn_pred_test


# In[115]:

ID = np.arange(1,20001)
ID = ID.tolist()
data = zip(ID,cnn_pred_test)
with open('CNN_v1.csv', 'w',newline='') as outfile:
    mywriter = csv.writer(outfile)
    # manually add header

    mywriter.writerow(['ID', 'Prediction'])
    for d in data:
        mywriter.writerow(d)


# In[ ]:



