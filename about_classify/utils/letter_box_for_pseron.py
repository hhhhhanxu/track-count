'''
Author: hanxu
Date: 2022-03-21 11:00:47
LastEditors: Please set LastEditors
LastEditTime: 2022-03-21 21:05:08
FilePath: /resnet/utils/letter_box_for_pseron.py
Description: a letter box resize method for person(w/h<<1)

Copyright (c) 2022 by bip_hx, All Rights Reserved. 
'''
import cv2
import numpy as np

class letter_resize:
    def cv2_letterbox_image_by_warp(img, expected_size):
        ih, iw = img.shape[0:2]
        ew, eh = expected_size
        scale = min(eh / ih, ew / iw)
        nh = int(ih * scale)
        nw = int(iw * scale)
        smat = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], np.float32)
        top = (eh - nh) // 2
        bottom = eh - nh - top
        left = (ew - nw) // 2
        right = ew - nw - left
        tmat = np.array([[1, 0, left], [0, 1, top], [0, 0, 1]], np.float32)
        amat = np.dot(tmat, smat)
        amat = amat[:2, :]
        dst = cv2.warpAffine(img, amat, expected_size,borderValue=(128,128,128))
        return dst
    
    
if __name__ == "__main__":
    import os
    from tqdm import tqdm 
    root_dataset = '/Users/hanxu/code/dataset/detect/voc'
    for i in tqdm(os.listdir(root_dataset)):
        if i.endswith('.jpg'):
            img = cv2.imread(os.path.join(root_dataset,i))
            new_img = letter_resize.cv2_letterbox_image_by_warp(img, (384,384))
            cv2.imwrite('letter_box/'+i,new_img)