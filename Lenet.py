# -*- coding: utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import time



localtime = time.asctime( time.localtime(time.time()))
print ('开始的时间是： ',localtime)

trans_image = transforms.ToTensor()

trainset = MNIST(root = 'F:\python_data\lenet', train=True, download=True, transform=trans_image)  #root为文件存放位置，
testset = MNIST(root = 'F:\python_data\lenet', train=False, download=True, transform=trans_image)  #download为True，表示若所在地址没有相应文件，则从网络上下载

trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
testloader = DataLoader(testset, batch_size=4, shuffle=True, num_workers=0)

classes = ('0','1','2','3','4','5','6','7','8','9')

class Lenet(nn.Module):        #创建神经网络cnn
    def __init__(self):
        super(Lenet,self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3,padding=1)   #输入的图片是28*28的，所以在第一个卷积层，为了让图片保持28*28，padding=1
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(400,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        
    def forward(self,x):
    
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view (-1,400)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
        
lenet = Lenet()
lenet = lenet.cuda()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    #调用gpu进行计算
print ('进行运算的是：',device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(lenet.parameters(), lr=0.001, momentum=0.9)

for epoch in range(5):
    
    running_loss = 0.0
    for i,data in enumerate(trainloader,0):
        inputs,labels = data
        inputs,labels = Variable(inputs.cuda()),Variable(labels.cuda())
        
        optimizer.zero_grad()
        
        outputs = lenet(inputs)
        loss = criterion(outputs,labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    print ('%d  loss: %.3f' %(epoch+1,loss))
print ("训练完成！")

correct = 0
total = 0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in testloader:
    inputs, labels = data
    inputs,labels = Variable(inputs.cuda()),Variable(labels.cuda())
    outputs = lenet(inputs)
    _, predicted = torch.max(outputs.data,1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
    
    for i in range(4):
        label = labels[i]
        if predicted[i] == labels[i]:
            class_correct[label] += 1
        class_total[label] += 1
       
    

print ("测试的总正确率为： %d %%" %(100 * correct/total))
for i in range(10):
    print ("%s 的正确率为： %.3f %%"%(classes[i], 100*class_correct[i]/class_total[i]))

localtime = time.asctime( time.localtime(time.time()))
print ('完成的时间是： ',localtime)
    
   












        







