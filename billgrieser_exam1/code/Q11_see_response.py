# =============================================================================
# Question 4
#
# Try several archictures of neurons in layers. The code to build a 
# network is modified to accept a variable number of layers, and then
# several runs are performed.
#
# =============================================================================

# --------------------------------------------------------------------------------------------
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

# --------------------------------------------------------------------------------------------
# Choose the right values for x.

#
# Initially try a network with a single hidden layer of 500 neurons
# and a moderate
# 
CHANNELS = 3
input_size = (CHANNELS * 32 * 32) # 3 color 32x32 images
hidden_size = [(1500,), ]
optimizers = [torch.optim.Adagrad]
transfer_functions = [nn.ReLU]
dropout= [0.5]
num_classes = 10
num_epochs = 0
batch_size = 32
learning_rate = .005

FORCE_CPU = False

if torch.cuda.is_available() and FORCE_CPU != True:
    print("Using cuda device for Torch")
    run_device = torch.device('cuda')
else:
    print("Using CPU devices for Torch.")
    run_device = torch.device('cpu')
    
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

test_set = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=10000, shuffle=False)

# Find the right classes name. Save it as a tuple of size 10.
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# --------------------------------------------------------------------------------------------
#
# Helper function
#

def get_total_parms(module):
    """Get the total number of trainable parameters in a network"""
    total_parms = 0
    for t in module.state_dict().values():
        total_parms += np.prod(t.shape)
    return total_parms

# =============================================================================
# Model Class
# 
# Define a model class that takes a variable number of hidden layer sizes
# and contructs a network to match. The network uses ReLu as the transfer
# function in each hidden layer. It uses purelin on the output layer.
#
# =============================================================================
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, 
                 transfer_function=nn.ReLU, dropout=0.0):
        super(Net, self).__init__()
        
        last_in = input_size
        self.hidden_layers = []
        
        for idx, sz in enumerate(hidden_size):
            new_module = nn.Linear(last_in, sz)
            self.add_module("layer_{0:02d}".format(idx+1), new_module)
            self.hidden_layers.append((new_module, transfer_function()))    
            last_in = sz
            
        self.dropout_layer=nn.Dropout(dropout)
        
        # Add the output layer (with an implied purelin activation)
        self.output_layer = nn.Linear(last_in, num_classes)

    def forward(self, x):
        out = x
        for layer, transfer in self.hidden_layers:
            out = layer(out)
            out = transfer(out)
        
        # Dropout
        out = self.dropout_layer(out)
        
        # Output layer
        out = self.output_layer(out)
        return out

# =============================================================================
# Function to make a model and load it
# =============================================================================
def make_model(this_hidden_size, run_device, 
                      transfer_function=nn.ReLU, dropout=0.0):

    # Fixed manual seed
    torch.manual_seed(267)
    start_time = time.time()
    
    # Instantiate a model
    net = Net(input_size, this_hidden_size, num_classes, transfer_function=transfer_function,
              dropout=dropout).to(device=run_device)
    print(net)
    net.train(True)
    
    STORED_MODEL = os.path.join("results", "Q08_dropout_1029_181713.pkl")
    net.load_state_dict(torch.load(STORED_MODEL, map_location=run_device))
    print("Loading from: ", STORED_MODEL)
    
    total_net_parms = get_total_parms(net)
    print ("Total trainable parameters:", total_net_parms)
   
    return net, time.time() - start_time
      
# =============================================================================
# Display results 
# =============================================================================
    
def show_one_test_sample(net, sample_idx):
    """Get the test sample with the input index, display it, and show
    the predicted and actual label"""
    net.train(False)
    

def record_test_results(net, run_device, loss, duration, optimizer, 
                        transfer_function):
    net.train(False)
    correct = 0
    total = 0
    
    print("Training run duration (secs): {0:0.1f}\n".format(duration))
    
    for images, labels in test_loader:
        images = Variable(images.view(-1, CHANNELS * 32 * 32)).to(device=run_device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        
        # Bring the predicted values to the CPU to compare against the labels
        correct += (predicted.cpu() == labels).sum()
    
    print('Accuracy of the network on the 10000 test images: {0:0.1f}%'.format(float(100 * correct) / total))

    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    for data in test_loader:
        images, labels = data
        images = Variable(images.view(-1, CHANNELS * 32 * 32)).to(device=run_device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted.cpu() == labels)
        for i in range(len(c)):
            label = labels[i]
            class_correct[label] += c[i].int()
            class_total[label] += 1
    
    # --------------------------------------------------------------------------------------------
    for i in range(num_classes):
        print('Accuracy of {0:10s} : {1:0.1f}%'.format(classes[i], float(100 * class_correct[i]) / class_total[i]))
        
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
        
    run_base = os.path.join('results', run_base)
    
    # Save run artifacts
    torch.save(net.state_dict(), run_base + suffix + '.pkl')
    
    with open(run_base + suffix + '_results.txt', 'w') as rfile:
        rfile.write(str(net))
        rfile.write('\n\n')
        rfile.write("Hidden Layer sizes: {0}\n".format(hidden_size))
        rfile.write("Total network weights + biases: {0}\n".format(get_total_parms(net)))
        rfile.write("Epochs: {0}\n".format(num_epochs))
        rfile.write("Learning rate: {0}\n".format(learning_rate))
        rfile.write("Optimizer: {0}\n".format(optimizer))
        rfile.write("Transfer Function: {0}\n".format(transfer_function))
        rfile.write("Batch Size: {0}\n".format(batch_size))
        rfile.write("Final loss: {0:0.4f}\n".format(loss.data.item()))
        rfile.write("Run device: {0}\n".format(run_device))
        rfile.write("Training run duration (secs): {0:0.1f}\n".format(duration))
        
        rfile.write('\n')
        rfile.write('Accuracy of the network on the 10000 test images: {0:0.1f}%\n'.format(float(100 * correct) / total))
        rfile.write('\n')
        for i in range(num_classes):
            rfile.write('Accuracy of {0:10s} : {1:0.1f}%\n'.format(classes[i], float(100 * class_correct[i]) / class_total[i]))
    
def imshow(img, title='One Image'):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()  # Show it now
    
def imshowax(ax, img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    ax.imshow(np.transpose(npimg, (1, 2, 0)))
    ax.tick_params(axis='both', which = 'both', bottom=False, left=False, tick1On=False, tick2On=False,
                   labelbottom=False, labelleft=False)
    
# =============================================================================
# MAIN -- Accept input from the user for test data and results to display
# =============================================================================
if __name__ == "__main__":
    
    
    # Make a model
    net, duration = make_model(hidden_size[0],
                              run_device=run_device, 
                              transfer_function=transfer_functions[0],
                              dropout=dropout[0])
                    
    # Get some test data
    for images, labels in test_loader:
        images = Variable(images.view(-1, CHANNELS * 32 * 32)).to(device=run_device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.cpu()

#%%
    normalizer = nn.Softmax(dim=1)
    while True:
        image_idx = input("Enter an index from 0 to 9999 from the test data (q to quit): ")
        
        try:
            if image_idx.lower() == 'q':
                break
            
            image_idx = int(image_idx)
            
        except:
            print("Bad input -- assuimg 0")
            image_idx = 0
            
        if image_idx >= 0 and image_idx < 10000:
            #imshow(images[image_idx].reshape(3,32,32), title="Actual: {0} Predicted: {1}".
            #       format(classes[labels[image_idx]], classes[int(predicted[image_idx])]))
            
            f, ax = plt.subplots(1, 2, figsize=(9.5,3.5))
            f.suptitle("Actual: {0} Predicted: {1}".
                   format(classes[labels[image_idx]], classes[int(predicted[image_idx])]))
            imshowax(ax[0], images[image_idx].reshape(3,32,32))
            y_pos = np.arange(len(classes))
            ax[1].set_yticks(y_pos)
            ax[1].set_yticklabels(classes, fontsize=8)
            
            softmaxed = normalizer(outputs[image_idx:image_idx+1])[0]
            
            ax[1].barh(y_pos, softmaxed, align='center',
                    color='blue')
            ax[1].invert_yaxis()
            plt.show()

        
        