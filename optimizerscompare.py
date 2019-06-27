#%%
import numpy as np 
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.utils.data as Data

LR = 0.01
BATCH_SIZE = 32
EPOCH = 12
#%%
# generate data
x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())
plt.scatter(x.data.numpy(), y.data.numpy())
torch_dataset = Data.TensorDataset(x, y)

loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
#%%
class MyNet(nn.Module):
    """Some Information about MyNet"""
    def __init__(self):
        super(MyNet, self).__init__()
        self.hidden = nn.Linear(1, 10)
        self.out = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)

        return x


# define different nets
net_SGD = MyNet()
net_Momentum = MyNet()
netRMSprop = MyNet()
net_Adam = MyNet()
nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

#define different optimizers
opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.5)
opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

loss_fn = nn.MSELoss()
losses_his = [[], [], [], []]
#%%
for epoch in range(EPOCH):
    print('Epoch: ', epoch)
    for step, (b_x, b_y) in enumerate(loader):          # for each training step
        for net, opt, l_his in zip(nets, optimizers, losses_his):
            output = net(b_x)              # get output for every net
            loss = loss_fn(output, b_y)  # compute loss for every net
            opt.zero_grad()                # clear gradients for next train
            loss.backward()                # backpropagation, compute gradients
            opt.step()                     # apply gradients
            l_his.append(loss.data.numpy())     # loss recoder
#%%
labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
for i, l_his in enumerate(losses_his):
    plt.plot(l_his, label=labels[i])
plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim((0, 0.2))
plt.show()