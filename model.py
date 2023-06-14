# modified from torchvision
import torch.nn as nn
from torch.nn import BatchNorm2d
import torch
import torch.nn.functional as F
from utils import MixStyle as MixStyle
import torchvision



class domain_classifier(nn.Module):
    def __init__(self, cls_num):
        super(domain_classifier, self).__init__()
        self.fc1 = nn.Linear(64 * 8 * 4, 100)
        self.fc2 = nn.Linear(100, cls_num) 
        self.drop = nn.Dropout2d(0.5)

    def forward(self, x):
        x = GradReverse.apply(x)
        x = F.leaky_relu(self.drop(self.fc1(x)))
        x = self.fc2(x)
        return x

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output 

def grad_reverse(x):
    return GradReverse()(x)

resnet = torchvision.models.resnet18(pretrained=True)
class BaseNet(nn.Module):
    def __init__(self, class_num):
        super(BaseNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=5, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer1 = nn.Sequential(
            resnet.layer1[0].conv1,
            resnet.layer1[0].bn1,
            nn.ReLU(),
            resnet.layer1[0].conv2,
            resnet.layer1[0].bn2,
            nn.ReLU()
        ) 
        # self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm2d(64)
        # self.relu2 = nn.ReLU()

        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        # self.bn3 = nn.BatchNorm2d(64)
        # self.relu3 = nn.ReLU()

        self.droppout = nn.Dropout(0.5)
        self.cls = nn.Linear(64*8*4, class_num)
        self.id_cls = domain_classifier(10)
        self.d_cls = domain_classifier(2)
        self.mixstyle = MixStyle(p=0.5, alpha=0.1)

    def forward(self, x, mix=False, mode='train'):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.layer1(x)
        # if mix:
        #     x = self.mixstyle(x)

        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.relu2(x)

        # if mix:
        #     x = self.mixstyle(x)

        # x = self.conv3(x)
        # x = self.bn3(x)
        # x = self.relu3(x)
        x_d = x
        x = x.view(x.size(0), -1)
        feat = x
        x = self.droppout(x)
        x = self.cls(x)
        if self.training:
            if mode == 'train':
                x_d = self.id_cls(x_d.view(x_d.size(0), -1))
            else:
                x_d = self.d_cls(x_d.view(x_d.size(0), -1))
            return x, x_d
        if mode=='feat':
            return feat
        return x


