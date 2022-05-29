import torch.nn as nn
import torch.nn.functional as F


class SumSender(nn.Module):

    def __init__(self, n_features: int, n_hidden: int):
        super(SumSender, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)


    def forward(self, x, _aux_input):
        x = self.fc1(x)

        return x


class SumReceiver(nn.Module):

    def __init__(self, n_features: int, n_hidden: int):
        super(SumReceiver, self).__init__()

        self.output = nn.Linear(n_hidden, n_features)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, _input, _aux_input):
        # return self.output(x)

        x = self.dropout(x)
        x = F.relu(self.output(x))
        return x
