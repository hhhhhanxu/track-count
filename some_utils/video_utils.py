'''
Author: your name
Date: 2022-03-25 20:48:20
LastEditTime: 2022-03-25 20:48:20
LastEditors: Please set LastEditors
Description: 
FilePath: /langqiao_node/utils/ img_utils.py
'''
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


class Video(object):
    """视频读取
    """    
    def __init__(self, path):
        self.videoCapture = cv2.VideoCapture(path)
        self.success = True
        self.fps = int(self.videoCapture.get(cv2.CAP_PROP_FPS))
        self.width = int(self.videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_next_frame(self):
        success, frame = self.videoCapture.read()
        self.success = success
        if success:
            frame = cv2.resize(frame,(1920,1080))
            return frame
        else:
            print("Fail to get next frame.\nTake it easy! Maybe the video has finished.")
            self._release()
            return None

    def _release(self):
        self.videoCapture.release()