import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data as data
import torchvision

EPOCH = 1
BATCH_SIZE = 50

train_data = torchvision.datasets.MNIST("mnist_data/",train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_data = torchvision.datasets.MNIST("mnist_data",train=False,transform=torchvision.transforms.ToTensor())

train_loader = data.DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)

test_x = Variable(torch.unsqueeze(test_data.test_data,dim=1).type(torch.FloatTensor)[:2000]/255.)
test_y = test_data.test_labels[:2000]

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,16,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.out = nn.Linear(32*7*7,128)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        output = self.out(x)
        return output

cnn = CNN()
print(cnn)

optimizer = torch.optim.Adam(cnn.parameters())
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for i,(x,y) in enumerate(train_loader):
        batch_x = Variable(x)
        batch_y = Variable(y)
        output = cnn(batch_x)
        loss = loss_func(output,batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print("loss:",loss.item())

test_output = cnn(test_x[:10])
pred_y = torch.max(test_output,dim=1)[1]
print("pred_y:",pred_y)
print("test_y:",test_y[:10])