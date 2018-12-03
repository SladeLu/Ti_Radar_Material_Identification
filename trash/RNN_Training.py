import torch
from torch import nn
import torchvision.datasets as dsets
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# train the training data n times, to save time, we just train 1 epoch
EPOCH = 1
LR = 0.01
TIME_STEP = 128          # rnn time step / image height
INPUT_SIZE = 37         # rnn input size / image width
BATCH_SIZE = 6

class CustomDatasetFromCSV(Dataset):
    def __init__(self, csv_path, height, width):
        self.data = pd.read_csv(csv_path)
        self.labels = np.asarray(self.data.iloc[:, 0])
        self.height = height
        self.width = width
        self.transforms = transforms

    def __getitem__(self, index):
        single_signal_label = self.labels[index]
        d = self.data.iloc[index][1:]
        sig_as_np = np.asarray(d).reshape(self.height, self.width)
        sig_as_tensor = torch.Tensor(sig_as_np)
        return (sig_as_tensor, single_signal_label)

    def __len__(self):
        return len(self.data.index)

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=TIME_STEP,         # rnn hidden unit
            num_layers=1,           # number of rnn layer
            # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            batch_first=True,
        )
        self.out = nn.Linear(TIME_STEP, 3)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        # None represents zero initial hidden state
        r_out, (h_n, h_c) = self.rnn(x, None)
        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out


train_data = CustomDatasetFromCSV(
    './data/formal_train.csv', TIME_STEP, INPUT_SIZE)
train_loader = torch.utils.data.DataLoader(
    dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = CustomDatasetFromCSV(
    './data/formal_test.csv', TIME_STEP, INPUT_SIZE)
test_loader = torch.utils.data.DataLoader(
    dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

rnn = RNN()
print(rnn)

# optimize all cnn parameters
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
# the target label is not one-hotted
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):        # gives batch data
        # reshape x to (batch, time_step, input_size)
        b_x = b_x.view(-1, TIME_STEP, INPUT_SIZE)
        output = rnn(b_x)                               # rnn output
        loss = loss_func(output, b_y)                   # cross entropy loss
        # clear gradients for this training step
        optimizer.zero_grad()
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()                                # apply gradients

for step, (b_x, b_y) in enumerate(test_loader):
    b_x = b_x.view(-1, TIME_STEP, INPUT_SIZE)
    output = rnn(b_x)
    output = torch.max(output, 1)[1].data.numpy().squeeze()
    print("pred_is", output)
    print("actual is", b_y)
