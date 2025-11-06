
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        # define layers here
        # input: 3 features for each electrode, output: 6 features
        self.layer1 = nn.Linear(3, 6) # output 6 for now
        
        # initialize activation functions
        self.relu = nn.ReLU()

    def forward(self, x):
        return x
        