# =============================================================================
# Question 1
#
# Fix bugs, supply initial values for tunables, explore performance
#
# The data is located in a folder at the same level as  "code"
#
# =============================================================================

# --------------------------------------------------------------------------------------------
import os
import sys
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import datetime
# --------------------------------------------------------------------------------------------
# Choose the right values for x.

#
# Initially try a network with a single hidden layer of 500 neurons
# and a moderate
# 
CHANNELS = 3
input_size = (CHANNELS * 32 * 32) # 3 color 32x32 images
hidden_size = 500
num_classes = 10
num_epochs = 200
batch_size = 250
learning_rate = .005

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
# --------------------------------------------------------------------------------------------
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()  # Show it now

# --------------------------------------------------------------------------------------------
# Show some sample images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# Just show a small grid to start with their labels
print("A sample of the data.")
imshow(torchvision.utils.make_grid(images[:16], nrow=4, normalize=False))
print("Labels:")
print(''.join('{0:10s}'.format(classes[labels[j]]) + ('\n' if (j+1) % 4 == 0 else ' ') for j in range(16)))
print()

# =============================================================================
# Set up the network
# =============================================================================
# --------------------------------------------------------------------------------------------
# Choose the right argument for xx
#
# Define a model class. This uses purelin() as the second-layer
# transfer function
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
# Define a an alternate model class for comparison that uses
# Softmax for the second-layer transfer function
class NetAlt(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NetAlt, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out
# --------------------------------------------------------------------------------------------
def get_total_parms(module):
    
    total_parms = 0
    # Find the total number of parameters being trained
    for t in module.state_dict().values():
        total_parms += np.prod(t.shape)
    return total_parms
        
# Fixed manual seed
torch.manual_seed(267)

# Choose the right argument for x
        
# Instantiate a model
net = Net(input_size, hidden_size, num_classes)
print(net)
total_net_parms = get_total_parms(net)
print ("Total trainable parameters:", total_net_parms)

# --------------------------------------------------------------------------------------------
# Choose the right argument for x
criterion = nn.CrossEntropyLoss() # nn.NLLLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

# =============================================================================
# Make a run
# =============================================================================
# --------------------------------------------------------------------------------------------
# There is bug here find it and fix it
for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):

        images, labels = data
        images= images.view(-1, CHANNELS * 32 * 32)
        images, labels = Variable(images), Variable(labels)
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_set) // batch_size, loss.data.item()))
            
# =============================================================================
# Display results summary
# =============================================================================
# --------------------------------------------------------------------------------------------
# There is bug here find it and fix it
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.view(-1, CHANNELS * 32 * 32))
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the 10000 test images: {0:0.1f}%'.format(float(100 * correct) / total))
# --------------------------------------------------------------------------------------------
_, predicted = torch.max(outputs.data, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
# --------------------------------------------------------------------------------------------
# There is bug here find it and fix it
class_correct = list(0. for i in range(num_classes))
class_total = list(0. for i in range(num_classes))
for data in test_loader:
    images, labels = data
    images = Variable(images.view(-1, CHANNELS * 32 * 32))
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels)
    for i in range(len(c)):
        label = labels[i]
        class_correct[label] += c[i].int()
        class_total[label] += 1

# --------------------------------------------------------------------------------------------
for i in range(num_classes):
    print('Accuracy of {0:10s} : {1:0.1f}%'.format(classes[i], float(100 * class_correct[i]) / class_total[i]))
# --------------------------------------------------------------------------------------------
# =============================================================================
# Save results to a file for comparison purposes.
# =============================================================================
now = datetime.datetime.now()
suffix = "_" + now.strftime('%m%d_%H%M%S')

if not os.path.exists('results'):
    os.makedirs('results')
    
run_base='console'

if sys.argv[0] != '':
    run_base = os.path.basename(sys.argv[0])
    run_base = os.path.join(os.path.splitext(run_base)[0])
    
run_base=os.path.join('results', run_base)

# Save run artifacts
torch.save(net.state_dict(), run_base + suffix + '.pkl')

with open(run_base + suffix + '_results.txt', 'w') as rfile:
    rfile.write(str(net))
    rfile.write('\n\n')
    rfile.write("Total network weights + biases: {0}\n".format(total_net_parms))
    rfile.write("Epochs: {0}\n".format(num_epochs))
    rfile.write("Learning rate: {0}\n".format(learning_rate))
    rfile.write("Batch Size: {0}\n".format(batch_size))
    rfile.write("Final loss: {0:0.4f}\n".format(loss.data.item()))
    rfile.write('\n')
    rfile.write('Accuracy of the network on the 10000 test images: {0:0.1f}%\n'.format(float(100 * correct) / total))
    rfile.write('\n')
    for i in range(num_classes):
        rfile.write('Accuracy of {0:10s} : {1:0.1f}%\n'.format(classes[i], float(100 * class_correct[i]) / class_total[i]))
    
