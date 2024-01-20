# sys and os works
import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
test_dir = os.path.join(root_dir, "test")
src_dir = os.path.join(root_dir, "src")
models_dir = os.path.join(root_dir, "models")
datasets_dir = os.path.join(root_dir, "Datasets")

sys.path.insert(0, root_dir) # 1 adds path to end of PYTHONPATH
sys.path.insert(0, models_dir)
sys.path.insert(0, test_dir) 
sys.path.insert(0, src_dir)
sys.path.insert(0, datasets_dir)

# mvm libraries
import src.config as cfg

if cfg.if_bit_slicing and not cfg.dataset:
    from src.pytorch_mvm_class_v3 import *
elif cfg.dataset:
    from geneix.pytorch_mvm_class_dataset import *   # import mvm class from geneix folder
else:
    from src.pytorch_mvm_class_no_bitslice import *

# Load in relevant libraries, and alias where appropriate
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from utils.mydata import get_dateset

# Define relevant variables for the ML task
batch_size = 64
num_classes = 10
learning_rate = 0.001
num_epochs = 10

# Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, test_loader = get_dateset('mnist', datasets_dir, batch_size)
print('Dataset Loaded')
from models.lenet5_mvm import LeNet5_mvm
    
model = LeNet5_mvm(num_classes).to(device)
model.float()
print(model)
model.to(device)
cost = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
total_step = len(train_loader)

total_step = len(train_loader)
for epoch in range(num_epochs):
    t0 = time.time()
    for i, (images, labels) in enumerate(train_loader):  
        # images = images.to(device)
        # labels = labels.to(device)
        # images = images.cuda()
        # labels = labels.cuda()

        # breakpoint()
        # images = images.type(torch.DoubleTensor)

        images = images.to(device)
        labels = labels.to(device)
        
        #Forward pass
        outputs = model(images)
        # outputs = outputs.type(torch.DoubleTensor)
        # outputs = outputs.to(device)
        # breakpoint()

        loss = cost(outputs, labels)
        	
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()    
        optimizer.step()
        		
        if (i+1) % 400 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
        		           .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            
    print('Time for epoch {}: {} seconds'.format(epoch+1, time.time() - t0))
            
# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
  
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

exit()