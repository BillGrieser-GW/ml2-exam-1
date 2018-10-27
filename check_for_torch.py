# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 10:35:04 2018

@author: billg_000
"""

### Test for Torch
import torch
print("Torch version:", torch.__version__)
print("Torch path:", torch.__path__)
print("Has cuda?:", torch.cuda.is_available())

print("Number of cuda devices:", torch.cuda.device_count())

