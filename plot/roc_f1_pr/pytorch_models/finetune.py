import time
import torch
import torchvision.models as models
from torchvision.datasets import ImageFolder
import torch.nn as nn


class FineTuneResnet50(nn.Module):
    def __init__(self, num_class=10, pretrained=True):
        super(FineTuneResnet50, self).__init__()
        self.num_class = num_class
        resnet50_net = models.resnet50(pretrained=pretrained)
        self.features = nn.Sequential(*list(resnet50_net.children())[:-1])
        self.classifier = nn.Linear(2048, self.num_class)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
        

class FineTuneVGG16(nn.Module):
    def __init__(self, num_class=10, pretrained=True):
        super(FineTuneVGG16, self).__init__()
        vgg16_net = models.vgg16_bn(pretrained=pretrained)
        self.num_class = num_class
        self.features = vgg16_net.features
        self.avgpool = vgg16_net.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, self.num_class),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class FineTuneInceptionv3(nn.Module):
    def __init__(self, num_class=10, pretrained=True):
        super(FineTuneInceptionv3, self).__init__()
        self.num_class = num_class
        inception_v3 = models.inception_v3(pretrained=pretrained, aux_logits=False, transform_input=False)
        inception_v3.fc = nn.Linear(2048, self.num_class)
        self.model = inception_v3

    def forward(self, x):
        x = self.model(x)
        return x
