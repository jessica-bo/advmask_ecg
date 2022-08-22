import torch
import torch.nn as nn

class mlp(nn.Module):
    def __init__(self, noutputs,):
        super().__init__()

        self.noutputs = noutputs
        self.hidden = 24
        self.dropout = 0.5

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=self.hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=5, stride=5, padding=2),
            nn.Conv1d(in_channels=self.hidden, out_channels=self.hidden, kernel_size=3, padding=1),
        )
        
        self.linear = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features=self.hidden*1000, out_features=self.noutputs),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x,1)
        x = self.linear(x)
        return x


