
# coding: utf-8

# In[14]:

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

predict = []
data_path = "./data.mat"
data_raw = loadmat(data_path)

train_img = data_raw["train_img"]
test_img = data_raw["test_img"]
train_lbl = data_raw["train_lbl"]

#train_img = train_img.astype(np.float64)
# train_img -= np.mean(train_img, axis = 0)
# train_img /= np.std(train_img, axis = 0)
# Max = [None]*784
# for i in range(0,784):
#     Max[i] = max(train_img[:,i])
# train_img = train_img/Max

#process training data 
tensor_train_img = torch.from_numpy(train_img) # transform to torch tensors
tensor_train_lbl = torch.from_numpy(train_lbl)
tensor_train_img = tensor_train_img.type(torch.FloatTensor)
tensor_train_lbl = tensor_train_lbl.type(torch.LongTensor)
# transform = transforms.Compose([
#                        transforms.ToPILImage(),
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ])
my_trainset = utils.TensorDataset(tensor_train_img,tensor_train_lbl) # create your datset

#process testing data
test_lbl = np.zeros((20000))
tensor_test_img = torch.from_numpy(test_img)
tensor_test_lbl = torch.from_numpy(test_lbl)
tensor_test_img = tensor_test_img.type(torch.FloatTensor)
tensor_test_lbl = tensor_test_lbl.type(torch.LongTensor)
my_testset = utils.TensorDataset(tensor_test_img,tensor_test_lbl)
print("finish processing data")

def simple_gradient():
    # print the gradient of 2x^2 + 5x
    x = Variable(torch.ones(2, 2) * 2, requires_grad=True)
    z = 2 * (x * x) + 5 * x
    # run the backpropagation
    z.backward(torch.ones(2, 2))
    print(x.grad)


def create_nn(batch_size=200, learning_rate=0.001, epochs=10,
              log_interval=10):
    
    my_train_dataloader = utils.DataLoader(my_trainset,batch_size=batch_size, shuffle=True) # create your dataloader
    my_test_dataloader = utils.DataLoader(my_testset,batch_size=batch_size, shuffle=True) # create your dataloader


    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(28 * 28, 200)
            self.fc2 = nn.Linear(200, 200)
            self.fc3 = nn.Linear(200, 10)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return F.log_softmax(x)

    net = Net()
    print(net)

    # create a stochastic gradient descent optimizer
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    # create a loss function
    criterion = nn.NLLLoss()

    # run the main training loop
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(my_train_dataloader):
            data, target = Variable(data), Variable(target)
            # resize data from (batch_size, 1, 28, 28) to (batch_size, 28*28)
            data = data.view(-1, 28*28)
            target = target.view(200)
            optimizer.zero_grad()
            net_out = net(data)
#             print("out: ",net_out)
#             print("target: ",target)
            loss = criterion(net_out, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(my_train_dataloader.dataset),
                           100. * batch_idx / len(my_train_dataloader), loss.data[0]))
        print("finish training")
    # predict the test set label

   
    for data in tensor_test_img:
        data = Variable(data, volatile=True)
        data = data.view(-1, 28 * 28)
        net_out = net(data)
        # sum up batch loss
        pred = net_out.data.max(1)[1]  # get the index of the max log-probability
        predict.append(pred[0])

    print("finish testing")
    print("prediction: ", predict)

            
    print("finish writing") 

if __name__ == "__main__":
    run_opt = 2
    if run_opt == 1:
        simple_gradient()
    elif run_opt == 2:

        create_nn()


# In[13]:

ID = np.arange(1,20001)
ID = ID.tolist()
data = zip(ID,predict)
with open('first.csv', 'w',newline='') as outfile:
    mywriter = csv.writer(outfile)
    # manually add header

    mywriter.writerow(['ID', 'Prediction'])
    for d in data:
        mywriter.writerow(d)


# In[ ]:




# In[ ]:



