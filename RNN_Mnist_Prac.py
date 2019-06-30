import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms


LR = 0.01
EPOCH = 2
BATCH_SIZE = 100
DOWNLOAD_MNIST = False
TIME_STEP = 28
INPUT_SIZE = 28

train_data = torchvision.datasets.MNIST(root='./mnist', train=True, transform=transforms.ToTensor())
test_data = torchvision.datasets.MNIST(root='./mnist', train = False, transform=transforms.ToTensor())
test_x = test_data.data.type(torch.FloatTensor)[:1000]/255.
test_y = test_data.targets[:1000].numpy()
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


class RNN(nn.Module):
    """Some Information about RNN"""
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn= nn.LSTM(
            input_size = INPUT_SIZE,
            hidden_size = 64,
            num_layers = 1,
            batch_first = True
        )
        self.out = nn.Linear(64, 10)


    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None) 
        out = self.out(r_out[:, -1, :])
        return out


rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):        # gives batch data
        b_x = b_x.view(-1, 28, 28)              # reshape x to (batch, time_step, input_size)

        output = rnn(b_x)                               # rnn output
        loss = loss_func(output, b_y)                   # cross entropy loss
        optimizer.zero_grad()                           # clear gradients for this training step
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()                                # apply gradients

        if step % 50 == 0:
            test_output = rnn(test_x)                   # (samples, time_step, input_size)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
            print('Epoch: ', epoch, ' | Step: ', step, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

# print 10 predictions from test data
test_output = rnn(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')
