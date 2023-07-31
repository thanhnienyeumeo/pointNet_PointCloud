import torch
import torch.nn as nn
import numpy as np
class makeSkeleton(nn.Module):
    def __init__(self):
        super(makeSkeleton, self).__init__()
        self.Flat = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 51)
        )
    def forward(self, x):
        return self.Flat(x)

a = np.random.randn(1024)
print(a.shape)
a = torch.from_numpy(a).float()
print(a.shape)
model = makeSkeleton()
print(model(a).shape)