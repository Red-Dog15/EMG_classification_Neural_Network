import os, sys

# Add parent directory of this file (Scripts/) to module search path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
from DATA.Data_Conversion import tensors_list

print (tensors_list[1])

