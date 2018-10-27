# --------------------------------------------------------------------------------------------
import os
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

#%%
# --------------------------------------------------------------------------------------------
# Choose the right values for x.
input_size = 1024
hidden_size = 250
hidden1_size = 500
hidden2_size = 100
num_classes = 10
num_epochs = 100
batch_size = 250
learning_rate = 0.125

FORCE_CPU = False

#%%
if torch.cuda.is_available() and FORCE_CPU != True:
    print("Using cuda device for Torch")
    run_device = torch.device('cuda')
else:
    print("Using CPU devices for Torch.")
    run_device = torch.device('cpu')

# --------------------------------------------------------------------------------------------
transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# --------------------------------------------------------------------------------------------

train_set = torchvision.datasets.CIFAR10(root='./data_cifar', train=True, download=True, \
                                         transform=transform)
#%%
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data_cifar', train=False, download=True, transform=transform)

test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
#%%
# Find the right classes name. Save it as a tuple of size 10.
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# --------------------------------------------------------------------------------------------
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
#%%
dataiter = iter(train_loader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))
#%%
print(' '.join('%5s' % classes[labels[j]] for j in range(len(labels))))
#%%
# --------------------------------------------------------------------------------------------
# Choose the right argument for xx
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.tsoftmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.tsoftmax(out)
        return out
    
class Net2(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, num_classes):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.transfer1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.transfer2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2_size, num_classes)
        self.transfer3 = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.transfer1(out)
        out = self.fc2(out)
        out = self.transfer2(out)
        out = self.fc3(out)
        out = self.transfer3(out)
        return out
#%%
torch.manual_seed(267)
torch.cuda.manual_seed_all(267)
# --------------------------------------------------------------------------------------------
# Create the network on the selected device
net = Net2(input_size, hidden1_size, hidden2_size, num_classes).to(device=run_device)
#net = Net(input_size, hidden_size, num_classes).to(device=run_device)
#%%
STORED_MODEL = os.path.join(".", "model_cpu_500_100_gray.pkl")

net.load_state_dict(torch.load(STORED_MODEL))
print("Loading from: ", STORED_MODEL)
# --------------------------------------------------------------------------------------------
# Choose the right argument for x
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
# --------------------------------------------------------------------------------------------
# There is bug here find it and fix it
for epoch in range(num_epochs):
    net.train()
    for i, data in enumerate(train_loader):

        images, labels = data
        images= images.view(-1, 1 * 32 * 32)
        
        images, labels = Variable(images).to(device=run_device), Variable(labels).to(device=run_device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_set) // batch_size, loss.data.item()))
# --------------------------------------------------------------------------------------------            
#%%
# There is bug here find it and fix it
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.view(-1, 1 * 32 * 32)).to(device=run_device)
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)

    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
# --------------------------------------------------------------------------------------------

_, predicted = torch.max(outputs.data, 1)
predicted = predicted.cpu()
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
#%%
# --------------------------------------------------------------------------------------------
# There is bug here find it and fix it
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in test_loader:
    images, labels = data
    images = Variable(images.view(-1, 1 * 32 * 32)).to(device=run_device)
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    
    c = (predicted.cpu() == labels)
    
    for i in range(len(c)):
        label = labels[i]
        class_correct[label] += c[i].int()
        class_total[label] += 1

# --------------------------------------------------------------------------------------------
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
#%%
# --------------------------------------------------------------------------------------------
torch.save(net.state_dict(), './model_cpu_500_100_gray_lr0125.pkl')