import os
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.utils.data as Data 
import torchvision
import matplotlib.pyplot as plt

LR = 0.01
BATCH_SIZE = 200
EPOCH = 2
DOWNLOAD_MNIST = False

# download mnist dataset and assign it to train_data and test_data datasets
train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     
    transform=torchvision.transforms.ToTensor(),   # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]                     
    download=DOWNLOAD_MNIST,
)

test_data = torchvision.datasets.MNIST(root='./mnist', train=False, download=DOWNLOAD_MNIST)

# see pictures in the training dataset
#print(train_data.train_data.size())
#print(train_data.train_labels.size())
#plt.imshow(train_data.train_data[0])
#plt.title('%i' %train_data.train_labels[0])

# load the data to train
train_loader = Data.DataLoader(dataset = train_data, batch_size = BATCH_SIZE, shuffle= True)

test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:1000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.targets[:1000].numpy()

class CNN(nn.Module):
    """Some Information about CNN"""
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.out = nn.Linear(in_features= 32 * 7 * 7, out_features= 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)

        return output

cnn = CNN()
print(cnn)

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_fn = torch.nn.CrossEntropyLoss()


for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        out = cnn(b_x)
        loss = loss_fn(out, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
            print('Epoch: ', epoch, ' | Step:', step,'| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
            

# print 10 predictions from test data
test_output= cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')

