import tensorflow as tf
from time import time
import sys
import os
import logging
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logging.getLogger(__name__)

class LearningTask:
    def __init__(self, params):
        self.classes = params['classes']
        self.batch_size = params['batch_size']
        self.dataset = params['dataset']
        self.data_per_sample = params['data_per_sample']
        self.model = params['model']()
        self.optimizer = params['optimizer'](self.model.parameters(), lr=1e-3)
        self.loss_fcn=  params['loss_fcn']
        
    def gen_learning_task_sample(self):
        """Generates a sample for a given learning task distribution
        A sample is given by a DataLoader which is a subset of our whole dataset that we want to fit a model to.

        Returns:
            _type_: _description_
        """
        sampler = torch.utils.data.RandomSampler(self.dataset, replacement=True, num_samples=self.data_per_sample)
        trainloader = torch.utils.data.DataLoader(self.dataset, sampler=sampler, batch_size=self.batch_size,num_workers=2)
        return trainloader
        
    def view_images(self, images, labels):
        def imshow(img):
            img = img/2+0.5 #converting
            npimg = img.numpy()
            plt.imshow(np.transpose(npimg, (1,2,0)))
            plt.show()
        grid = torchvision.utils.make_grid(images)
        imshow(grid)
        print(' '.join(f'{self.classes[labels[j]]:5s}' for j in range(self.batch_size)))
        
    def train(self, dataloader, epochs):
        for epoch in range(epochs):
            running_loss=0.0
            for i, data in enumerate(dataloader):
                inputs, labels = data
                
                self.optimizer.zero_grad()
                
                output=self.model(inputs)
                loss = self.loss_fcn(output, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss +=loss.item()
                if i % 35 == 34:
                    print(f'[{epoch+1}, {i+1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss=  0.0
        PATH  = './cifar_net.pth'
        torch.save(self.model.state_dict(), PATH)
        print('Finished Training')
     
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,16, 3)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(800, 250)
        self.fc2 = nn.Linear(250, 100)
        self.fc3 = nn.Linear(100, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x=torch.flatten(x, 1)
        x=F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def generate_samples_list(dataset, num_samples):
    pass

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    MNISTtrain = datasets.MNIST(root='./data', train=True,download=True, transform = transform)

    MNISTtest = datasets.MNIST(root='./data', train=False,download=True, transform = transform)

    fullset =torch.utils.data.ConcatDataset((MNISTtrain, MNISTtest))
    
    subsets = torch.utils.data.random_split(fullset, [35000, 35000])
    
    assert len(subsets)==2
    assert len(subsets[0])==35000
    
    trainset = subsets[0]
    testset = subsets[1]
    
    # logging.debug('Trainset variable type: %s' % type(trainset))
    
    # subsets = []
    # numSubsets=  100
    # totalImages = len(fullset)
    # logging.debug('Ideal size of subset: %d' % (totalImages//numSubsets))
    # for i in range(numSubsets):
    #     subset = torch.utils.data.Subset(fullset, range(i*totalImages//numSubsets, (i+1)*totalImages//numSubsets))
    #     assert len(subset)==(totalImages//numSubsets)
    #     subsets.append(subset)
    # logging.debug('Size of subset: %d' % len(subset))
    params={
        'classes':  ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'),
        'batch_size': 10,
        'dataset': trainset,
        'data_per_sample': 700,
        'model': Net,
        'optimizer': torch.optim.Adam,
        'loss_fcn': nn.CrossEntropyLoss(),
    }
    
    
    # samples = generate_samples_list(trainset, 10)
    MNISTclassification = LearningTask(params)
    sample = MNISTclassification.gen_learning_task_sample()
    images, labels = iter(sample).next()
    MNISTclassification.view_images(images, labels)
    # MNISTclassification.train(sample, 3)

if __name__ == '__main__':
    main()
