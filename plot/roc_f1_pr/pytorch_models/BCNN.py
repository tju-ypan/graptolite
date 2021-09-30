import torch
import torchvision.models as models
from torchvision.datasets import ImageFolder
import torch.nn as nn


class BCNN(nn.Module):
    def __init__(self, num_class=10):
        super(BCNN, self).__init__()
        # Convolution and pooling layers of VGG-16.
        vgg16_net = models.vgg16(pretrained=True)
        # state_dict = torch.load("./models/vgg16_bn-6c64b313.pth")
        # vgg16_net.load_state_dict(state_dict)
        self.features = vgg16_net.features
        # Remove pool5.
        self.features = torch.nn.Sequential(*list(self.features.children())[:-1])
        self.classifier = torch.nn.Linear(512**2, num_class)

    def forward(self, x):
        # batch size
        bs = x.size()[0]
        x = self.features(x)
        # channel size
        cs = x.size()[1]
        # spatial size
        ss = x.size()[2]
        x = x.view(bs, cs, ss ** 2)
        x = torch.bmm(x, torch.transpose(x, 1, 2)) / (ss ** 2)  # Bilinear
        x = x.view(bs, cs ** 2)
        x = torch.sqrt(x + 1e-5)
        x = torch.nn.functional.normalize(x)
        x = self.classifier(x)
        return x


class BCNN_resnet50(nn.Module):
    def __init__(self, num_class=10):
        super(BCNN_resnet50, self).__init__()
        # Convolution and pooling layers of VGG-16.
        resnet50_net = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet50_net.children())[:-1])
        # Remove pool5.
        self.features = torch.nn.Sequential(*list(self.features.children())[:-1])
        self.classifier = torch.nn.Linear(2048**2, num_class)
        # init Linear
        for m in self.classifier.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # batch size
        bs = x.size()[0]
        x = self.features(x)
        # channel size
        cs = x.size()[1]
        # spatial size
        ss = x.size()[2]
        x = x.view(bs, cs, ss ** 2)
        x = torch.bmm(x, torch.transpose(x, 1, 2)) / (ss ** 2)  # Bilinear
        x = x.view(bs, cs ** 2)
        x = torch.sqrt(x + 1e-5)
        x = torch.nn.functional.normalize(x)
        x = self.classifier(x)
        return x


#if __name__ == '__main__':
#    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#    input_test = torch.ones(1, 3, 448, 448).to(device)
#    vgg16 = BCNN_resnet50(num_class=10).to(device)
#    output_test = vgg16(input_test)
#    print(output_test.shape)
