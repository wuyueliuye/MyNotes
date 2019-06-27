#%%
import numpy as np 
import matplotlib.pyplot as plt 
import torch 
import torch.nn as nn
import torch.nn.functional as F
#%%
# try regression
# Scatter for Regression
x = torch.unsqueeze(torch.linspace(-2, 2, 200), dim = 1)
y = x.pow(2) + torch.rand(x.size())
plt.scatter(x.data.numpy(), y.data.numpy())

#%%
# nets 
class MyNet1(nn.Module):
    """Some Information about MyNet1"""
    def __init__(self, n_feature, n_hidden, n_output):
        super(MyNet1, self).__init__()
        self.hidden = nn.Linear(n_feature, n_hidden)
        self.out = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)

        return x

net1 = MyNet1(1, 10, 1)
print(net1)

# quick nets
net2 = nn.Sequential(
    nn.Linear(1, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)
print(net2)
#%%
optimizer1 = torch.optim.SGD(net1.parameters(), lr=0.1)
loss_fn1 = nn.MSELoss()

plt.ion()
for i in range(100):
    y_hat = net1(x)
    loss1 = loss_fn1(y_hat, y)
    optimizer1.zero_grad()
    loss1.backward()
    optimizer1.step()

    if i%5 ==0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), y_hat.data.numpy(), '-r')
        plt.text(1.5, 0.5, 'Loss:%.4f'%loss1.data.numpy())
        plt.pause(0.1)
plt.show()
plt.ioff()

#%%
# try classification 
# data for class
n_num = torch.ones(100, 2)
x0 = torch.normal(2*n_num, 1)
y0 = torch.zeros(100)
x1 = torch.normal(-2*n_num,1)
y1 = torch.ones(100)

x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1), ).type(torch.LongTensor)

plt.scatter(x.data.numpy()[:,0], x.data.numpy()[:,1],c=y.data.numpy(), s=100, lw=0)
plt.show() 

#%%
# nets
net3 = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 2)
)

print(net3)

optimizer2 = torch.optim.SGD(net3.parameters(), lr=1e-2)
loss_fn2 = nn.CrossEntropyLoss()

for i in range(100):
    y_esti = net3(x)
    loss2 = loss_fn2(y_esti, y)
    optimizer2.zero_grad()
    loss2.backward()
    optimizer2.step()

    if i%2 == 0:
        plt.cla()
        y_pred = torch.max(y_esti, 1)[1]
        pred_y = y_pred.data.numpy()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:,0], x.data.numpy()[:,1],c=y_pred.data.numpy(),s=100, lw=0, cmap='RdYlGn')
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 10, 'color':  'red'})
        plt.pause(.1)
plt.show()
plt.ioff()


