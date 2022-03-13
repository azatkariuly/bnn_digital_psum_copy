import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Function
from .binarized_modules import  BinarizeConv2d

__all__ = [
    'vgg_small_binary'
]

class VGG_Cifar10(nn.Module):

    def __init__(self, num_classes=1000, nbits_OA=32):
        super(VGG_Cifar10, self).__init__()
        self.infl_ratio=3;

        self.conv1 = nn.Conv2d(3, 128*self.infl_ratio, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(128*self.infl_ratio)
        self.hardtanh1 = nn.Hardtanh(inplace=True)

        self.conv2 = BinarizeConv2d(128*self.infl_ratio, 128*self.infl_ratio, kernel_size=3, padding=1, bias=False, nbits_OA=nbits_OA)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batchnorm2 = nn.BatchNorm2d(128*self.infl_ratio)
        self.hardtanh2 = nn.Hardtanh(inplace=True)

        self.conv3 = BinarizeConv2d(128*self.infl_ratio, 256*self.infl_ratio, kernel_size=3, padding=1, bias=False, nbits_OA=nbits_OA)
        self.batchnorm3 = nn.BatchNorm2d(256*self.infl_ratio)
        self.hardtanh3 = nn.Hardtanh(inplace=True)

        self.conv4 = BinarizeConv2d(256*self.infl_ratio, 256*self.infl_ratio, kernel_size=3, padding=1, bias=False, nbits_OA=nbits_OA)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batchnorm4 = nn.BatchNorm2d(256*self.infl_ratio)
        self.hardtanh4 = nn.Hardtanh(inplace=True)

        self.conv5 = BinarizeConv2d(256*self.infl_ratio, 512*self.infl_ratio, kernel_size=3, padding=1, bias=False, nbits_OA=nbits_OA)
        self.batchnorm5 = nn.BatchNorm2d(512*self.infl_ratio)
        self.hardtanh5 = nn.Hardtanh(inplace=True)

        self.conv6 = BinarizeConv2d(512*self.infl_ratio, 512, kernel_size=3, padding=1, bias=False, nbits_OA=nbits_OA)
        self.maxpool6 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batchnorm6 = nn.BatchNorm2d(512)
        self.hardtanh6 = nn.Hardtanh(inplace=True)

        self.linear = nn.Linear(512*4*4, num_classes, bias=False)
        self.softmax = nn.LogSoftmax()

        '''
        self.classifier = nn.Sequential(
            BinarizeLinear(512 * 4 * 4, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.Hardtanh(inplace=True),
            #nn.Dropout(0.5),
            BinarizeLinear(1024, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.Hardtanh(inplace=True),
            #nn.Dropout(0.5),
            ,
            nn.BatchNorm1d(num_classes, affine=False),

        )
        '''

    def forward(self, x):

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.hardtanh1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.batchnorm2(x)
        x = self.hardtanh2(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.hardtanh3(x)

        x = self.conv4(x)
        x = self.maxpool4(x)
        x = self.batchnorm4(x)
        x = self.hardtanh4(x)

        x = self.conv5(x)
        x = self.batchnorm5(x)
        x = self.hardtanh5(x)

        x = self.conv6(x)
        x = self.maxpool6(x)
        x = self.batchnorm6(x)
        x = self.hardtanh6(x)

        x = x.view(-1, 512 * 4 * 4)

        x = self.linear(x)
        x = self.softmax(x)
        #x = self.classifier(x)
        return x


def vgg_small_binary(**kwargs):
    num_classes = kwargs.get( 'num_classes', 10)
    return VGG_Cifar10(num_classes, nbits_OA=kwargs['nbits_OA'])
