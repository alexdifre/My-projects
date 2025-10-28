import os
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
import socket
import argparse

import socket
from torch.utils.data import DataLoader
from PIL import Image


def get_data_folder():
    """
    Restituisce il percorso alla cartella dei dati.
    """
    hostname = socket.gethostname()
    if hostname.startswith('visiongpu'):
        data_folder = '/data/vision/phillipi/rep-learn/datasets'
    elif hostname.startswith('yonglong-home'):
        data_folder = '/home/yonglong/Data/data'
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_folder = os.path.join(script_dir, "data")

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    return data_folder


class CIFAR100InstanceSample(datasets.CIFAR100):
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, k=4096, mode='exact', is_sample=True, percent=1.0):
        super().__init__(
            root=root, 
            train=train, 
            download=download,
            transform=transform, 
            target_transform=target_transform
        )
        
        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        num_classes = 100
        
        # Usa gli attributi corretti: self.data e self.targets
        num_samples = len(self.data)
        label = self.targets  # <--- Modifica qui

        self.cls_positive = [[] for _ in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for _ in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j != i:
                    self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(arr) for arr in self.cls_positive]
        self.cls_negative = [np.asarray(arr) for arr in self.cls_negative]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [
                np.random.permutation(arr)[:n] 
                for arr in self.cls_negative
            ]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, index):
        # Usa self.data e self.targets invece di train_data/test_data
        img, target = self.data[index], self.targets[index]  # <--- Modifica qui
        
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.is_sample:
            return img, target, index
        else:
            if self.mode == 'exact':
                pos_idx = index
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[target], 1)[0]
            else:
                raise NotImplementedError(self.mode)
                
            replace = self.k > len(self.cls_negative[target])
            neg_idx = np.random.choice(
                self.cls_negative[target], 
                self.k, 
                replace=replace
            )
            
            sample_idx = np.hstack(([pos_idx], neg_idx))
            return img, target, index, sample_idx
        
def get_cifar100_dataloaders_sample(batch_size, num_workers=8, k=4096, mode='exact', is_sample=True, percent=1.0):
    """
    Restituisce i dataloader per CIFAR100 con instance sampling.
    """
    data_folder = get_data_folder()

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_set = CIFAR100InstanceSample(root=data_folder,
                                        train=True,
                                        transform=train_transform,
                                        download=True,
                                        k=k,
                                        mode=mode,
                                        is_sample=is_sample,
                                        percent=percent)
    val_set = CIFAR100InstanceSample(root=data_folder,
                                      train=False,
                                      transform=test_transform,
                                      download=True)

    n_data = len(train_set)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val_set,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers)

    return train_loader, val_loader, n_data

