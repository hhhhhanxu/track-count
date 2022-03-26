'''
Author: hanxu
Date: 2022-03-21 09:30:19
LastEditors:  
LastEditTime: 2022-03-21 19:10:09
FilePath: /resnet/model.py
Description: a resnet-101 model for langqiao person classify

Copyright (c) 2022 by bip_hx, All Rights Reserved. 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

class My_resnet(nn.Module):
    def __init__(self,model_size = 101,num_classes=3):
        super(My_resnet,self).__init__()
        self.conv = nn.Conv2d(3,3,kernel_size=1)
        self.resnet = self._init_model(model_size,num_classes)


    def forward(self,x):
        x = self.conv(x)
        x = self.resnet(x)

        return x

    def _init_model(self,model_size,num_classes):
        if model_size == 101:
            resnet = torchvision.models.resnet101(pretrained=False,num_classes=num_classes)
        elif model_size == 50:
            resnet = torchvision.models.resnet50(pretrained=False,num_classes=num_classes)
        elif model_size == 34:
            resnet = torchvision.models.resnet50(pretrained=False,num_classes=num_classes)
        elif model_size == 152:
            resnet = torchvision.models.resnet152(pretrained=False,num_classes=num_classes)

        return resnet

if __name__ == "__main__":
    model = My_resnet(model_size=50)