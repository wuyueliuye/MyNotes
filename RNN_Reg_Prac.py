import numpy as np 
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn

LR = 0.02
TIME_STEP = 10
INPUT_SIZE = 1
HIDDEN_SIZE = 32

# use the x(sin curve) to fit y(cos curve)
#steps = np.linspace(0, 2*np.pi, 100)
#x = np.sin(steps)
#y = np.cos(steps)

#plt.plot(steps, x, 'y-', label = 'given (sin)')
#plt.plot(steps, y, 'g-', label = 'target (cos)')
#plt.legend(loc = 'lower right')
#plt.show()

class RNN(nn.Module):
    """Some Information about RNN"""
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size = INPUT_SIZE,
            hidden_size = HIDDEN_SIZE,
            num_layers = 1,
            batch_first = True

        )
        self.out = nn.Linear(32, 1)

    def forward(self, x, hidden_state):
        # x, input, shape:(batch, time_step, input_size)
        # hidden_state, the hidden impacts, shape: (n_layers, time_step, hidden_size)
        # r_out, the output, shape:(batch, time_step, hidden_size )
        r_out, hidden_state = self.rnn(x, hidden_state)
        outs = []
        for time_step in range(r_out.size(1)):    # calculate output for each time step
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), hidden_state

    
rnn = RNN()

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_fn = nn.MSELoss()
h_state = None

plt.ion()
for i in range(60):
    start, end = i*np.pi, (i+1)*np.pi
    steps = np.linspace(start, end, 10, dtype=np.float32, endpoint=False)  
    x_value = np.sin(steps)
    y_value = np.cos(steps)

    x = torch.from_numpy(x_value[np.newaxis, :, np.newaxis])    # shape (batch, time_step, input_size)
    y = torch.from_numpy(y_value[np.newaxis, :, np.newaxis])

    prediction, h_state = rnn(x, h_state)   

    h_state = h_state.data      # repack the hidden state, break the connection from last iteration

    loss = loss_fn(prediction, y)         # calculate loss
    optimizer.zero_grad()                   # clear gradients
    loss.backward()                         # backpropagation
    optimizer.step()                        # apply gradients

    # plotting
    plt.plot(steps, y_value.flatten(), 'y-')
    plt.plot(steps, prediction.data.numpy().flatten(), 'g-')
    plt.draw()
    plt.pause(0.1)

plt.ioff()
plt.show()