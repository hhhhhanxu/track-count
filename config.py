'''
Author: your name
Date: 2022-03-25 19:37:10
LastEditTime: 2022-03-25 21:06:00
LastEditors: Please set LastEditors
Description: 
FilePath: /langqiao_node/config.py
'''
from easydict import EasyDict as edict
import numpy as np

np.random.seed(15) 


__C = edict()
cfg = __C

cfg.yolo              = edict()
cfg.yolo.conf_thres   = 0.25
cfg.yolo.iou_thres    = 0.45
cfg.yolo.augment      = False
cfg.yolo.agnostic_nms = False

cfg.weights           = edict()
cfg.weights.yolo      = 'about_detect/weights/langqiaobest.pt'
cfg.weights.resnet    = 'about_classify/weights/best.pth'

cfg.plot              = edict()
cfg.plot.categories   = ['crew','clean','other']
cfg.plot.colors       = np.random.randint(0,255,(3,3)).tolist()


