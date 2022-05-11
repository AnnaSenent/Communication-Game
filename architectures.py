import torch
import torch.nn as nn


class SumSender(nn.Module):

    def __init__(self, n_features, n_hidden):
        super(SumSender, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)

    def forward(self, x, _aux_input): # MSE?
        return F.relu(self.fc1(x))


class SumReceiver(nn.Module):

    def __init__(self, n_features, n_hidden):
        super(SumReceiver, self).__init__()

        self.output = nn.Linear(n_hidden, n_features)

    def forward(self, x, _input, _aux_input):

        return self.output(x)
