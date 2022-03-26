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


def Myloader(path):
    return Image.open(path).convert('RGB')


# get a list of paths and labels
def init_process(path, lens):
    data = []
    name = find_label(path)
    for i in range(lens[0], lens[1]):
        data.append([path % i, name])

    return data


class MyDataset(Dataset):
    def __init__(self, data, transform, loder):
        self.data = data
        self.transform = transform
        self.loader = loder

    def __getitem__(self, item):
        img, label = self.data[item]
        img = self.loader(img)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)


def find_label(str):
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
    # train
    train_data = data1 + data2 + data3[0:150] + data4[0:150]

    train = MyDataset(train_data, transform=transform, loder=Myloader)
    # test
    test_data = data3[150:200] + data4[150:200]
    test = MyDataset(test_data, transform=transform, loder=Myloader)

    train_data = DataLoader(dataset=train, batch_size=10, shuffle=True, num_workers=0)
    test_data = DataLoader(dataset=test, batch_size=1, shuffle=True, num_workers=0)

    return train_data, test_data
