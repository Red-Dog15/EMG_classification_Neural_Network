
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        # define layers here
        # input: 3 features for each electrode, output: 6 features
        self.layer1 = nn.Linear(3, 6) # output 6 for now
        self.layer2 = nn.Linear(6, 6) # keep 6 for now
        self.layer3 = nn.Linear(6, 1) # backWW to 3 features for each electrode
        # initialize activation functions
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x
        