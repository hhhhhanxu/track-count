'''
Author: hanxu
Date: 2022-03-21 09:04:29
LastEditors: Please set LastEditors
LastEditTime: 2022-03-21 21:07:30
FilePath: /resnet/dataset.py
Description: a dataset for langqiao person classify

Copyright (c) 2022 by bip_hx, All Rights Reserved. 
'''

import torch
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable 
from torchvision import transforms

from utils.print_color_txt import colorstr
from utils.letter_box_for_pseron import letter_resize

import numpy as np
import cv2 as cv
import os

class langqiao_dataset(Dataset):
    def __init__(self,root_path,letter=letter_resize,img_size=384,train=True,transform=None,):
        self.root_path = os.path.join(root_path,'train' if train else 'test')
        self.img_list,self.label_list = self._get_img_list()
        self.img_size = img_size
        self.transform = transform
        self.letter = letter
        
    
    def __getitem__(self,index):

        img = cv.imread(os.path.join(self.root_path,str(self.label_list[index] if self.label_list[index]!=2 else 3),self.img_list[index])) #BGR
        assert img is not None,'Can\'t find img:'+ os.path.join(self.root_path,str(self.label_list[index] if self.label_list[index]!=2 else 3),self.img_list[index])
        h,w = img.shape[:2]
        if h!=w:
            #add letter_box, and update img & h & w 
            img = letter_resize.cv2_letterbox_image_by_warp(img, (self.img_size,self.img_size))
            h,w = img.shape[:2]
        if h/self.img_size !=1:
            # 实际上letter box输出的就是目的size,这里应该是没用的
            img = cv.resize(img,(self.img_size,self.img_size))
        if self.transform != None:
            img = self.transform(img)
        
        label = self.label_list[index]
        return img,torch.from_numpy(np.array(label))
        

    def __len__(self):
        if len(self.img_list)==len(self.label_list):
            return len(self.img_list)
        else:  
            #不过就这种文件的储存结构根本不会出现这种问题ahhh
            print(colorstr('warning! ')+'imgs:{} and labels{} don\'t match !'.format(len(self.img_list,self.label_list)))
            return len(self.img_list) if len(self.img_list)<len(self.label_list) else len(self.label_list) 


    def _get_img_list(self):
        img_list = []
        label_list = []
        for i,j,k in os.walk(self.root_path):
            if i.split('/')[-1] in ['0','1','3']:
                label = i.split('/')[-1] if i.split('/')[-1] != '3' else '2'
                img_list.extend(k)
                label_list.extend([item*int(label) for item in len(k)*[1]])
        return img_list,label_list



if __name__ == "__main__":
    
    a = langqiao_dataset('/Users/hanxu/code/dataset/my',train=False,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                        ]))
    dataloader = DataLoader(a,batch_size=1,shuffle=False,num_workers=2) #一定要开shuffle=True，否则就变成顺序采样了（列表里就是顺序排列的）
    for iter,(img,label) in enumerate(dataloader):
        print(iter,img,label)
