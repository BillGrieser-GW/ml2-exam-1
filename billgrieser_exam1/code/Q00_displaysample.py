# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 16:10:35 2018

@author: billg_000
"""
import os
import sys
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import datetime
import time
import matplotlib.pyplot as plt

batch_size = 50
#%%

# =============================================================================
# Load training and test data
# =============================================================================
# --------------------------------------------------------------------------------------------
# Define a transformation that converts each image to a tensor and normalizes
# each channel
transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# --------------------------------------------------------------------------------------------

DATA_ROOT = os.path.join("..", "data_cifar")
train_set = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

test_set = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Find the right classes name. Save it as a tuple of size 10.
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# =============================================================================
# Show a sample of the data
# =============================================================================
def imshow(ax, img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    ax.imshow(np.transpose(npimg, (1, 2, 0)))
    
# Show some sample images
dataiter = iter(train_loader)
images, labels = dataiter.next()

#%%
f, ax = plt.subplots(4,4, figsize=(6,7))
f.suptitle("Sample images with labels")
next_idx=0
for xgrid in range(4):
    for ygrid in range(4):
        ax[xgrid, ygrid].tick_params(axis='both', which = 'both', bottom=False, left=False, tick1On=False, tick2On=False,
          labelbottom=False, labelleft=False)
        # ax[xgrid, ygrid].set_title(str(xgrid) + ", " + str(ygrid), fontsize=9)
        imshow(ax[xgrid, ygrid], images[next_idx])
        ax[xgrid, ygrid].set_title(classes[labels[next_idx]], fontsize=10)
        next_idx+=1
        
plt.show()



