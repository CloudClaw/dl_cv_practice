import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.layer_1 = nn.Linear(input, 128, bias=False)
        # self.batchnorm = nn.BatchNorm1d(128) Как я понял не стоит использовать вместе с dropout на 1 слое
        self.act = nn.ReLU()
        self.layer_2 = nn.Linear(128, output)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.layer_1(x)
        # x = self.batchnorm(x)
        x = self.act(x)
        x = self.dropout(x)
        out = self.layer_2(x)
        return out