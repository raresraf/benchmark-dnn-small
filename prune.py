import argparse
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from copy import deepcopy
from torch.nn.utils import prune
import torch.nn as nn

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

from models.resnet import SmallerResNet18, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from models.vgg import VGG16, VGG16_S
from models.vit import VIT_S, VIT, VIT_timm
from optimizers import parse_optimizer, supported_optimizers
from sklearn.metrics import classification_report


def prune_model_l1_unstructured(model, layer_type, proportion):
    for module in model.modules():
        if isinstance(module, layer_type):
            prune.l1_unstructured(module, 'weight', proportion)
            prune.remove(module, 'weight')
    return model

def prune_model_l1_structured(model, layer_type, proportion):
    for module in model.modules():
        if isinstance(module, layer_type):
            prune.ln_structured(module, 'weight', proportion, n=1, dim=1)
            prune.remove(module, 'weight')
    return model

def prune_model_global_unstructured(model, layer_type, proportion):
    module_tups = []
    for module in model.modules():
        if isinstance(module, layer_type):
            module_tups.append((module, 'weight'))

    prune.global_unstructured(
        parameters=module_tups, pruning_method=prune.L1Unstructured,
        amount=proportion
    )
    for module, _ in module_tups:
        prune.remove(module, 'weight')
    return model


def parse_args(argv=None):
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--model', default='resnet18', type=str, help='model',
                        choices=['resnet18', 'resnet18_s', 'vgg16', 'vgg16_s', 'vit', 'vit_s', 'vit_timm'])
    parser.add_argument('--optim', type=str, help='optimizer', required=True,
                        choices=supported_optimizers())
    parser.add_argument('--seed', type=int, default=42, help='Random seed to use. default=123.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')

    args, optim_args = parser.parse_known_args(argv)
    return args, optim_args


def build_dataset():
    """Build CIFAR10 train and test data loaders. Will download datasets if needed."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                            transform=transform_train)
    train_loader = DataLoader(trainset, batch_size=8, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                           transform=transform_test)
    test_loader = DataLoader(testset, batch_size=8, shuffle=False, num_workers=4)

    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return train_loader, test_loader


def build_model(model, device):
    net = {
        'resnet18': ResNet18,
        'resnet18_s': SmallerResNet18,
        'vgg16': VGG16,
        'vgg16_s': VGG16_S,
        'vit': VIT,
        'vit_s': VIT_S,
        'vit_timm': VIT_timm,
    }[model]()
    net = net.to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        cudnn.deterministict = True

    return net


def test(net, device, data_loader):
    net.eval()
    y_test = np.array([])
    y_pred = np.array([])
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(data_loader), total=len(data_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            y_test = np.append(y_test, targets.cpu())
            y_pred = np.append(y_pred, predicted.cpu())

    # print(classification_report(y_test, y_pred,
    #                            target_names=['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',
    #                                          'truck']))
    accuracy = 100. * correct / total
    print('Test acc %.3f' % accuracy)

    return accuracy


def train_epoch(net, epoch, device, data_loader, optimizer, criterion):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in tqdm(enumerate(data_loader), desc='Epoch {}'.format(epoch),
                                             total=len(data_loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print('train acc %.3f' % accuracy)
    print('train loss %.6f' % train_loss)

    return accuracy, train_loss


def train_cifar10(opt, optimizer_opts):
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    train_loader, test_loader = build_dataset()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = build_model(opt.model, device)
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))

    criterion = CrossEntropyLoss()
    optimizer, optimizer_run_name = parse_optimizer(opt.optim, optimizer_opts, net.parameters())
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

    run_name = '{}_cifar10_{}'.format(
        opt.model,
        optimizer_run_name
    )

    train_accuracies = []
    test_accuracies = []
    train_loss_data = []

    for epoch in range(opt.epochs):
        train_acc, train_loss = train_epoch(net, epoch, device, train_loader, optimizer, criterion)
        test_acc = test(net, device, test_loader)

        scheduler.step()

        print({
                'Training Loss': train_loss,
                'Training Accuracy': train_acc,
                'Test Accuracy': test_acc,
            })

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        train_loss_data.append(train_loss)

    print({
        'Training Loss': min(train_loss_data),
        'Training Accuracy': max(train_accuracies),
        'Test Accuracy': max(test_accuracies),
    })


    accs = {
        'prune_model_l1_unstructured': {},
        'prune_model_l1_structured': {},
        'prune_model_global_unstructured': {},
    }

    print("===")
    print("prune_model_l1_unstructured")
    for i in np.linspace(0, 1, num=21):
        net_tmp = prune_model_l1_unstructured(deepcopy(net), nn.Conv2d, i)
        acc = test(net_tmp, device, test_loader)
        accs['prune_model_l1_unstructured'][i] = acc

    print("===")
    print("prune_model_l1_structured")
    for i in np.linspace(0, 1, num=21):
        net_tmp = prune_model_l1_structured(deepcopy(net), nn.Conv2d, i)
        acc = test(net_tmp, device, test_loader)
        accs['prune_model_l1_structured'][i] = acc


    print("===")
    print("prune_model_global_unstructured")
    for i in np.linspace(0, 1, num=21):
        net_tmp = prune_model_global_unstructured(deepcopy(net), nn.Conv2d, i)
        acc = test(net_tmp, device, test_loader)
        accs['prune_model_global_unstructured'][i] = acc


    print(accs)
if __name__ == '__main__':
    train_cifar10(*parse_args())
    
