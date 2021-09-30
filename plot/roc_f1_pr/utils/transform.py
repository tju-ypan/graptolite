from PIL import Image, ImageOps
from torchvision import datasets, transforms, utils
import torch


# [0.948078, 0.93855226, 0.9332005] [0.14589554, 0.17054074, 0.18254866]
def get_transform_for_train(mean=(0.5, 0.5, 0.5), var=(0.5, 0.5, 0.5)):
    transform_list = [
        # transforms.Resize((448, 448)),
        # transforms.RandomCrop(448),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, var)
    ]
    return transforms.Compose(transform_list)


def get_transform_for_test(mean=(0.5, 0.5, 0.5), var=(0.5, 0.5, 0.5)):
    transform_list = [
        transforms.Resize((448, 448)),
        # transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize(mean, var)
    ]
    return transforms.Compose(transform_list)
