# =============================================================================
# Question 10
#
# Load a model and show its confusion matrix versus the test data
#
# =============================================================================

# --------------------------------------------------------------------------------------------
import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import time

# For heatmap
import matplotlib.pyplot as plt

# For classification report
from sklearn.metrics import classification_report

# Identify the model to evaluate
STORED_MODEL = os.path.join("results", "best_model.pkl")

CHANNELS = 3
input_size = (CHANNELS * 32 * 32) # 3 color 32x32 images
hidden_size = [(1500,)]
optimizers = [torch.optim.Adagrad]
transfer_functions = [nn.LeakyReLU]
dropout= [0.5]
num_classes = 10
num_epochs = 0
batch_size = 32
learning_rate = .005

FORCE_CPU = True

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
train_set = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

test_set = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

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
def make_model(this_hidden_size, from_file, run_device, 
                      transfer_function=nn.ReLU, dropout=0.0):

    # Fixed manual seed
    torch.manual_seed(267)
    start_time = time.time()
    
    # Instantiate a model
    net = Net(input_size, this_hidden_size, num_classes, transfer_function=transfer_function,
              dropout=dropout).to(device=run_device)
    
    print(net)
    net.train(False)
    net.load_state_dict(torch.load(from_file, map_location=run_device))
    print("Loading model from: ", from_file)
    
    total_net_parms = get_total_parms(net)
    print ("Total trainable parameters:", total_net_parms)
   
    return net, time.time() - start_time
   
#%%
# =============================================================================
# Display results summary
# =============================================================================
def show_evaluation_metrics(net, run_device):
    
    # x is predicted
    # y is actual
    c_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    all_labels = []
    all_predicted = []
    
    for data in test_loader:
        images, labels = data
        images = Variable(images.view(-1, CHANNELS * 32 * 32)).to(device=run_device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.cpu()
        
        for idx in range(len(predicted)):
            c_matrix[labels[idx], predicted[idx]] += 1
            
        # Accumlate for error metrics
        all_labels += [int(x) for x in labels]
        all_predicted += [int(x) for x in predicted]
            
    # Draw the confusion matrix

    # This uses Seaborn, which gives a nice plot; however, our cloud instances
    # don't have the latest version of matplotlib and this code displays the
    # confusion matrix without the counts instead of including them.
    # import seaborn as sns
    # fig, ax = plt.subplots(figsize=(9, 6))
    # plt.title("Confusion Matrix", fontsize=16)
    # sns.set(font_scale=1.0)  # Label size
    # sns.heatmap(c_matrix, annot=True, annot_kws={"size": 14}, robust=True, fmt='d', \
    #            linecolor='gray', linewidths=0.5, square=False, cbar=True, cmap='Blues',
    #            xticklabels=classes, yticklabels=classes)
    # plt.ylabel('Actual Labels', fontsize=14)
    # plt.xlabel('Predicted Labels', fontsize=14)
    # tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45)
    # plt.show()
    
    # This draws a confusion matrix using just matplotlib
    fig, ax = plt.subplots(figsize=(8, 7))
   
    # Draw the grid squares and color them based on the value of the underlying measure
    ax.matshow(c_matrix, cmap=plt.cm.Blues, alpha=0.6)
    
    # Set the tick labels size
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(14)
        
    for i in range(c_matrix.shape[0]):
        for j in range(c_matrix.shape[1]):
            plt.text(x=j, y=i, s="{0}".format(c_matrix[i,j]), 
                      va='center', ha='center', fontdict={'fontsize':14})
            
    plt.ylabel('Actual Labels', fontsize=12) 
    plt.xlabel('Predicted Labels', fontsize=12)
    tick_marks = np.arange(len(classes))
    #ax.set_ticks_position(position='bottom')
    plt.grid(b=False)
    plt.xticks(tick_marks, classes, rotation=45, fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)       
    plt.suptitle("Confusion Matrix", fontdict={'fontsize':16})
    plt.show()
    
    # Print Classification Report
    print("\nClassification Report\n")
    print(classification_report(all_labels, all_predicted, target_names=classes))
    
    # IN CASE OF EMERGENCY: Uncomment to see a text-only CM
    #print("\nConfusion matrix (non-graphically)\n")
    #print(c_matrix)
    
    return c_matrix

    
#%%  
# =============================================================================
# MAIN -- show the matrix for the loaded model
# =============================================================================
if __name__ == "__main__":
    
    # Make a model and then run the test data through it
    net, duration = make_model(hidden_size[0], STORED_MODEL, run_device, 
                               transfer_functions[0], dropout[0])
#%%
    # Confusion Matrix
    c_matrix = show_evaluation_metrics(net, run_device)