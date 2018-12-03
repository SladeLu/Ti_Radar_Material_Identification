import torch
from torch import nn
import torchvision.datasets as dsets
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorboardX import SummaryWriter

# train the training data n times, to save time, we just train 1 epoch
EPOCH = 100
LR = 0.01
TIME_STEP = 36          # rnn time step / image height
INPUT_SIZE = 32         # rnn input size / image width
BATCH_SIZE = 20
HIDDEN_SIZE = 30
OUTPUT_SIZE = 3

writer = SummaryWriter()

class CustomDatasetFromCSV(Dataset):
    def __init__(self, csv_path, height, width):
        self.data = pd.read_csv(csv_path,header=None)
        self.labels = np.asarray(self.data.iloc[:, 0])
        self.height = height
        self.width = width

    def __getitem__(self, index):
        label = self.labels[index]
        data_as_np = np.asarray(self.data.iloc[index][1:-1]).reshape(self.height, self.width)
        data_as_tensor = torch.Tensor(data_as_np)
        return (data_as_tensor, label)

    def __len__(self):
        return len(self.data.index)

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,         # rnn hidden unit
            num_layers=2,           # number of rnn layer
            batch_first=True,
        )
        self.out = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        return out


train_data = CustomDatasetFromCSV(
    './data/RP/RP_train.csv', TIME_STEP, INPUT_SIZE)
train_loader = torch.utils.data.DataLoader(
    dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = CustomDatasetFromCSV(
    './data/RP/RP_test.csv', TIME_STEP, INPUT_SIZE)
test_loader = torch.utils.data.DataLoader(
    dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

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
        sv = b_x

        if step % 20 == 0:
            pre_res = []
            act_res = []
            for step, (b_x, b_y) in enumerate(test_loader):
                b_x = b_x.view(-1, TIME_STEP, INPUT_SIZE)
                output = rnn(b_x)
                output = torch.max(output, 1)[1].data.numpy().squeeze()
                # print("pred_is", output)
                pre_res.extend(output)
                # print("actual is", b_y)
                act_res.extend(b_y)
            ri = 0
            for index in range(len(pre_res)):
                if pre_res[index]==act_res[index]:
                    ri += 1
            accuracy = ri/len(pre_res)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

# pre_res = []
# act_res = []
# for step, (b_x, b_y) in enumerate(test_loader):
#     b_x = b_x.view(-1, TIME_STEP, INPUT_SIZE)
#     output = rnn(b_x)
#     output = torch.max(output, 1)[1].data.numpy().squeeze()
#     # print("pred_is", output)
#     pre_res.extend(output)
#     # print("actual is", b_y)
#     act_res.extend(b_y)
# ri = 0
# for index in range(len(pre_res)):
#     if pre_res[index]==act_res[index]:
#         ri += 1
# print(ri/len(pre_res))

writer.add_graph(rnn, (sv,))
writer.export_scalars_to_json("./test.json")
writer.close()