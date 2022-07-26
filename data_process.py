# -*- coding:utf-8 -*-
"""
@Time: 2022/03/01 11:33
@Author: KI
@File: data_process.py
@Motto: Hungry And Humble
"""
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


def Myloader(path):
    return Image.open(path).convert('RGB')


# get a list of paths and labels.
def init_process(path, lens):
    data = []
    name = find_label(path)
    for i in range(lens[0], lens[1]):
        data.append([path % i, name])

    return data


class MyDataset(Dataset):
    def __init__(self, data, transform, loader):
        self.data = data
        self.transform = transform
        self.loader = loader

    def __getitem__(self, item):
        img, label = self.data[item]
        img = self.loader(img)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)


def find_label(str):
    """
    Find image tags based on file paths.

    :param str: file path
    :return: image label
    """
    first, last = 0, 0
    for i in range(len(str) - 1, -1, -1):
        if str[i] == '%' and str[i - 1] == '.':
            last = i - 1
        if (str[i] == 'c' or str[i] == 'd') and str[i - 1] == '/':
            first = i
            break

    name = str[first:last]
    if name == 'dog':
        return 1
    else:
        return 0


def load_data():
    print('data processing...')
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # normalization
    ])
    path1 = 'data/training_data/cats/cat.%d.jpg'
    data1 = init_process(path1, [0, 500])
    path2 = 'data/training_data/dogs/dog.%d.jpg'
    data2 = init_process(path2, [0, 500])
    path3 = 'data/testing_data/cats/cat.%d.jpg'
    data3 = init_process(path3, [1000, 1200])
    path4 = 'data/testing_data/dogs/dog.%d.jpg'
    data4 = init_process(path4, [1000, 1200])
    data = data1 + data2 + data3 + data4   # 1400
    # shuffle
    np.random.shuffle(data)
    # train, val, test = 900 + 200 + 300
    train_data, val_data, test_data = data[:900], data[900:1100], data[1100:]
    train_data = MyDataset(train_data, transform=transform, loader=Myloader)
    Dtr = DataLoader(dataset=train_data, batch_size=50, shuffle=True, num_workers=0)
    val_data = MyDataset(val_data, transform=transform, loader=Myloader)
    Val = DataLoader(dataset=val_data, batch_size=50, shuffle=True, num_workers=0)
    test_data = MyDataset(test_data, transform=transform, loader=Myloader)
    Dte = DataLoader(dataset=test_data, batch_size=50, shuffle=True, num_workers=0)

    return Dtr, Val, Dte
