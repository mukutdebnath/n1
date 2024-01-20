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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, test_loader = get_dateset('mnist', datasets_dir, batch_size)
from models.lenet5 import LeNet5    
model = LeNet5(num_classes).to(device)
cost = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
total_step = len(train_loader)


for epoch in range(num_epochs):
    out_relu1 = []
    out_relu2 = []
    out_relu3 = []
    out_relu4 = []
    t0 = time.time()
    for i, (images, labels) in enumerate(train_loader):  
        images = images.to(device)
        labels = labels.to(device)
        
        #Forward pass
        outputs = model(images)

        # outputs, temp_relu1, temp_relu2, temp_relu3, temp_relu4 = model(images)
        # out_relu1.append(max([item.max() for item in temp_relu1]).cpu().detach().numpy())
        # out_relu2.append(max([item.max() for item in temp_relu2]).cpu().detach().numpy())
        # out_relu3.append(max([item.max() for item in temp_relu3]).cpu().detach().numpy())
        # out_relu4.append(max([item.max() for item in temp_relu4]).cpu().detach().numpy())

        # relu_out.append(torch.max(out_relu).cpu().detach().numpy())
        # relu_out.append(torch.max(out_relu1).cpu().detach().numpy())
        
        # breakpoint()
        loss = cost(outputs, labels)
        	
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        		
        if (i+1) % 400 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
        		           .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            # print('Maximum relu outputs: ReLU1: {}, ReLU2: {}, ReLU: {}, ReLU4: {}'.format(max(out_relu1), max(out_relu2), max(out_relu3), max(out_relu4)))

    print('Time for epoch {}: {} seconds'.format(epoch+1, time.time() - t0))
            
# Test the model

out_relu1 = []
out_relu2 = []
out_relu3 = []
out_relu4 = []
  
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        # outputs, temp_relu1, temp_relu2, temp_relu3, temp_relu4 = model(images)
        # out_relu1.append(max([item.max() for item in temp_relu1]).cpu().detach().numpy())
        # out_relu2.append(max([item.max() for item in temp_relu2]).cpu().detach().numpy())
        # out_relu3.append(max([item.max() for item in temp_relu3]).cpu().detach().numpy())
        # out_relu4.append(max([item.max() for item in temp_relu4]).cpu().detach().numpy())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
    print('Maximum relu outputs: ReLU1: {}, ReLU2: {}, ReLU: {}, ReLU4: {}'.format(max(out_relu1), max(out_relu2), max(out_relu3), max(out_relu4)))
