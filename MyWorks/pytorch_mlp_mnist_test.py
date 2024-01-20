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
dataset_path = os.path.join(datasets_dir, 'MNIST')

import time
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

import torch.nn as nn
import torch.nn.functional as F

from models.mlp_mnist import *
from models.mlp_mnist_mvm import *
from models.mlp_mnist_adc import *

num_workers = 0
batch_size = 20
valid_size = 0.2
transform = transforms.ToTensor()
train_data = datasets.MNIST(root = dataset_path, train = True, download = True, transform = transform)
test_data = datasets.MNIST(root = dataset_path, train = False, download = True, transform = transform)
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_index, valid_index = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_index)
valid_sampler = SubsetRandomSampler(valid_index)
train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, 
                                           sampler = train_sampler, num_workers = num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size,
                                          sampler = valid_sampler, num_workers = num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size,
                                         num_workers = num_workers)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MLP2_CCO_ADC(10, 7)
model.float()
model = model.cuda()
print(model)

# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()
# specify optimizer (stochastic gradient descent) and learning rate = 0.01
optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)

# number of epochs to train the model
n_epochs = 50
# initialize tracker for minimum validation loss

model.load_state_dict(torch.load('model_mlp.pt'))

# bit_weights = 4

# # breakpoint()

# w = torch.mul(torch.sub(model.fc1.weight, float(model.fc1.weight.min())), pow(2, bit_weights))
# w = torch.div(w, float(torch.sub(model.fc1.weight.max(), model.fc1.weight.min())))
# w = torch.floor(w)
# w = torch.mul(w, float(torch.sub(model.fc1.weight.max(), model.fc1.weight.min())))
# w = torch.div(w, pow(2, bit_weights))
# w = torch.add(w, float(model.fc1.weight.min()))
# # breakpoint()
# model.fc1.weight = nn.Parameter(w)

# w = torch.mul(torch.sub(model.fc2.weight, float(model.fc2.weight.min())), pow(2, bit_weights))
# w = torch.div(w, float(torch.sub(model.fc2.weight.max(), model.fc2.weight.min())))
# w = torch.floor(w)
# w = torch.mul(w, float(torch.sub(model.fc2.weight.max(), model.fc2.weight.min())))
# w = torch.div(w, pow(2, bit_weights))
# w = torch.add(w, float(model.fc2.weight.min()))
# model.fc2.weight = nn.Parameter(w)

model.cuda()

# initialize lists to monitor test loss and accuracy
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval() # prep model for evaluation

for data, target in test_loader:
    data = torch.where(data > 0.5, 1.0, 0.0)
    # data = data.type(torch.FloatTensor)
    
    data = data.cuda()
    target = target.cuda()
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the loss
    loss = criterion(output, target)
    # update test loss 
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)
    # compare predictions to true label
    correct = np.squeeze(pred.eq(target.data.view_as(pred)))
    # calculate test accuracy for each object class
    for i in range(len(target)):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# calculate and print avg test loss
test_loss = test_loss/len(test_loader.sampler)
print('Test Loss: {:.6f}\n'.format(test_loss))
print('Test Accuracy: {}'.format(100 * np.sum(class_correct) / np.sum(class_total)))
# print('{}, {}'.format(np.sum(class_correct), np.sum(class_total)))

# for i in range(10):
#     if class_total[i] > 0:
#         print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
#             str(i), 100 * class_correct[i] / class_total[i],
#             np.sum(class_correct[i]), np.sum(class_total[i])))
#     else:
#         print('Test Accuracy of %5s: N/A (no training examples)')
#         print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
#     100. * np.sum(class_correct) / np.sum(class_total),
#     np.sum(class_correct), np.sum(class_total)))
        
# print('Average accuracy on the MNIST dataset: {} ({}/{})'.format())