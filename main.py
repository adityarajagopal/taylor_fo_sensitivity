import os 
import sys
import copy
from tqdm import tqdm

import torch
import torchvision
import numpy as np

from models.lenet import LeNet5
from models.resnet import resnet18
from models.gate_layer import GateLayer

def get_model(name):
    if name == 'lenet':
        return LeNet5() 
    elif name == 'resnet':
        return resnet18(num_classes=10)

def get_data_loader(dataset_name, num_mb=30, batch_size=64):
    print(f"Getting data loader for dataset {dataset_name}")
    if dataset_name == 'mnist':

        transform=torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(32),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])
        dataset = torchvision.datasets.MNIST('/data/mnist',\
                                             download=True,\
                                             transform=transform)
    
    tmp_loader = torch.utils.data.DataLoader(dataset)
    all_indices = list(range(len(tmp_loader)))
    chosen_indexes = np.random.choice(all_indices, num_mb*batch_size)
    sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(chosen_indexes)
    loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sub_sampler,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    return loader

def get_taylor_sensitivities(model, data_loader, gpu=0):             
    device = f"cuda:{gpu}"
    criterion = torch.nn.CrossEntropyLoss()
    model = model.to(device, non_blocking=True) 
    model.eval()
    model.zero_grad()
    with tqdm(total=len(data_loader), desc='Gradient Computation') as iterator:
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.cuda(device, non_blocking=True),\
                                targets.cuda(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            iterator.set_postfix({'loss': loss.item()})
            iterator.update(1)
        
    model.to('cpu')
    sensitivities = {}
    for n,m in model.named_modules():
        if isinstance(m, GateLayer):
            m.weight.grad /= len(data_loader)
            metric = (m.weight*m.weight.grad).data.pow(2).view(m.weight.size(0), -1).sum(dim=1)
            sensitivities[n] = metric

    return sensitivities

def main():
    model = get_model('lenet')
    data_loader = get_data_loader('mnist')
    sensitivities = get_taylor_sensitivities(model, data_loader)
    print({k:len(v) for k,v in sensitivities.items()})

if __name__ == '__main__':
    main()


