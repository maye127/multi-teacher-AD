import torch
from torch import nn
import torchvision
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn import manifold

# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
class MNIST():
    def get(self, normal_lab=0, batch_size=32):
        if normal_lab < 0 or normal_lab > 9:
            raise ValueError('error normal_lab')
        root = './Data'
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=0.5, std=0.5),
             transforms.Resize(size=(256, 256)),
             transforms.CenterCrop(size=(224, 224))])
        data_set = torchvision.datasets.MNIST(root=root, train=True, download=False, transform=transform)
        print('Normal Label is', normal_lab, ', class', data_set.classes[normal_lab], '.')
        one_index = list()
        for i in range(len(data_set)):
            if data_set.targets[i] == normal_lab:
                one_index.append(i)
        data_set.data = data_set.data[one_index, :]
        data_set.targets = list(np.array(data_set.targets)[one_index])
        data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)

        data_set_test = torchvision.datasets.MNIST(root=root, train=False, download=False, transform=transform)
        data_loader_test = torch.utils.data.DataLoader(data_set_test, batch_size=batch_size, shuffle=False)
        temp = data_set_test.targets
        temp = np.array(temp)
        label_test = np.zeros(temp.shape)
        label_test[temp != normal_lab] = 1
        return data_loader, data_loader_test, label_test

class CIFAR10():
    def get(self, normal_lab=0, batch_size=32):
        if normal_lab < 0 or normal_lab > 9:
            raise ValueError('error normal_lab')
        root = './Data'
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
             transforms.Resize(size=(256, 256)),
             transforms.CenterCrop(size=(224, 224))])
        data_set = torchvision.datasets.CIFAR10(root=root, train=True, download=False, transform=transform)
        print('Normal Label is', normal_lab, ', class', data_set.classes[normal_lab], '.')
        one_index = list()
        for i in range(len(data_set)):
            if data_set.targets[i] == normal_lab:
                one_index.append(i)
        data_set.data = data_set.data[one_index, :]
        data_set.targets = list(np.array(data_set.targets)[one_index])
        data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)

        data_set_test = torchvision.datasets.CIFAR10(root=root, train=False, download=False, transform=transform)
        data_loader_test = torch.utils.data.DataLoader(data_set_test, batch_size=batch_size, shuffle=False)
        temp = data_set_test.targets
        temp = np.array(temp)
        label_test = np.zeros(temp.shape)
        label_test[temp != normal_lab] = 1
        return data_loader, data_loader_test, label_test


class MVTec():
    def __init__(self, path='./Data/mvtec'):
        self.path = path

    def get(self, normal_lab=0, batch_size=32, num_workers=0):
        if normal_lab < 0 or normal_lab > 14:
            raise ValueError('error normal_lab')
        root = './Data/mvtec'
        name = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
                'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
        root = os.path.join(root, name[normal_lab])
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
             transforms.Resize(size=(256, 256)),
             transforms.CenterCrop(size=(224, 224))])

        print('Normal Label is', normal_lab, ', class', name[normal_lab], '.')
        data_set = torchvision.datasets.ImageFolder(root=os.path.join(root, 'train'), transform=transform)
        data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        data_set_test = torchvision.datasets.ImageFolder(root=os.path.join(root, 'test'), transform=transform)
        data_loader_test = torch.utils.data.DataLoader(data_set_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        temp = data_set_test.targets
        temp = np.array(temp)
        label_test = np.zeros(temp.shape)
        label_test[temp != data_set_test.class_to_idx['good']] = 1
        return data_loader, data_loader_test, label_test

    def get_1024(self, normal_lab=0, batch_size=32, num_workers=0):
        if normal_lab < 0 or normal_lab > 14:
            raise ValueError('error normal_lab')
        root = './Data/mvtec'
        name = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
                'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
        root = os.path.join(root, name[normal_lab])
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
             # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
             transforms.Resize(size=(1024, 1024))])

        print('Normal Label is', normal_lab, ', class', name[normal_lab], '.')
        data_set = torchvision.datasets.ImageFolder(root=os.path.join(root, 'train'), transform=transform)
        data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        data_set_test = torchvision.datasets.ImageFolder(root=os.path.join(root, 'test'), transform=transform)
        data_loader_test = torch.utils.data.DataLoader(data_set_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        temp = data_set_test.targets
        temp = np.array(temp)
        label_test = np.zeros(temp.shape)
        label_test[temp != data_set_test.class_to_idx['good']] = 1
        return data_loader, data_loader_test, label_test

    def Randomcut(self, image, size=(224, 224), patch=4):
        n, c, h, w = image.shape
        TF = transforms.RandomCrop(size=size)
        image_patch = torch.zeros(0, c, size[0], size[1])
        for i in range(patch):
            image_patch = torch.cat([image_patch, TF(image)], dim=0)
        return image_patch

    def Fullcut(self, image, size=(224, 224), step=200):
        n, c, h, w = image.shape
        image_list = list()
        for i in range(n):
            image_patch = torch.zeros(0, c, size[0], size[1])
            for ph in range(0, h-size[0]+1, step):
                for pw in range(0, w-size[1]+1, step):
                    # print(pw, ph)
                    image_patch = torch.cat([image_patch, image[i:i+1, :, ph:ph+size[0], pw:pw+size[1]]])
            image_list.append(image_patch)
        return image_list



# def CIFAR10(normal_lab=0, batch_size=32):
#     root = './Data'
#     transform = transforms.Compose(
#         [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), transforms.Resize((256, 256)),
#          transforms.CenterCrop(size=(224, 224))])
#     data_set = torchvision.datasets.CIFAR10(root=root, train=True, download=False, transform=transform)
#     print('Normal Label is', normal_lab, ', class', data_set.classes[normal_lab], '.')
#     one_index = list()
#     for i in range(len(data_set)):
#         if data_set.targets[i] == normal_lab:
#             one_index.append(i)
#     data_set.data = data_set.data[one_index, :]
#     data_set.targets = list(np.array(data_set.targets)[one_index])
#     data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)
#     data_set_test = torchvision.datasets.CIFAR10(root=root, train=False, download=False, transform=transform)
#     data_loader_test = torch.utils.data.DataLoader(data_set_test, batch_size=batch_size, shuffle=False)
#     temp = data_set_test.targets
#     temp = np.array(temp)
#     label_test = np.zeros(temp.shape)
#     label_test[temp != normal_lab] = 1
#     return data_loader, data_loader_test, label_test
