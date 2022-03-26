'''
Author: hanxu
Date: 2022-03-25 18:36:30
LastEditTime: 2022-03-25 21:16:15
LastEditors: Please set LastEditors
Description: 
FilePath: i don't know
'''
import cv2 
import numpy as np
import torch
from torchvision import transforms
# detect
from about_detect.models.experimental import attempt_load
from about_detect.utils.general import check_img_size, non_max_suppression, scale_coords
from about_detect.utils.torch_utils import select_device
from about_detect.utils.plots import plot_images
from utils.datasets import letterbox
#classify
from about_classify.model import My_resnet
from about_classify.utils.letter_box_for_pseron import letter_resize

from config import cfg
from some_utils.video_utils import Video



class Detection(object):
    def __init__(self,model_name,device,cfg,mask=None,imgsz=640):
        #create model
        self.cfg = cfg
        if model_name =='yolo4':
            assert "now we don't support yolov4 !"

        elif model_name=='yolo5':
            self.device = select_device(device)
            self.mask   = mask
            self.half   = self.device.type != 'cpu'
            self.model  = attempt_load(cfg.weights.yolo, map_location=self.device)  # load FP32 model
            stride      = int(self.model.stride.max())  # model stride
            self.imgsz  = check_img_size(imgsz, s=stride)  # check img_size

            if self.half:
                self.model.half()  # to FP16
        else:
            assert "we don't spuuort model:{}".format(model_name)
        # classify model
        self.modelc = My_resnet() #use default para
        if self.device.type != 'cpu':
            self.modelc.load_state_dict(torch.load(cfg.weights.resnet))
            self.modelc.to(self.device)
            self.modelc.half()
        else:
            self.modelc.load_state_dict(torch.load(cfg.weights.resnet,map_location=torch.device('cpu')))
        self.modelc.eval()

        self.transfrom = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                        ])
        self.letter = letter_resize

    def detect(self,frame):
        '''
        输入原始图片，内部调用预处理函数
        input:an original video frame (np.array)
        return : detect result,including bboxes and labels
        '''
        width   = frame.shape[1]
        height  = frame.shape[0]
        if type(self.mask) != type(None):
            input_frame = frame & self.mask 
        else:
            input_frame = frame
        img = self._img_preprocess(input_frame)
        pred = self.model(img, augment=self.cfg.yolo.augment)[0]

        pred = non_max_suppression(pred, self.cfg.yolo.conf_thres, self.cfg.yolo.iou_thres, agnostic=self.cfg.yolo.agnostic_nms)
        # 还要把框resize到原尺寸

        for i, det in enumerate(pred):  # detections per image
            im0 = frame
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if det is not None and len(det)>0:
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
        # 得到det
        if det is not None:
            bboxes = det[:,:4].cpu().numpy()
            scores = det[:,4].cpu().numpy()
            labels = det[:,5].cpu().numpy()
            box_and_score = det[:,:5].cpu().numpy()

            new_labels = self._add_classify(bboxes, input_frame)
            new_labels = new_labels.reshape((len(new_labels),1))
            # print('ori labels:',labels)  # 旧label里面3是other，新的是2
            # print('new labels:',new_labels)
            return bboxes,new_labels,scores,box_and_score
        else:
            return None,None,None,None

    def _img_preprocess(self,frame):
        '''
        图片缩放，保证input是640*640，同时纵横比不变
        '''
        img_ori = frame.copy()
        #尺度缩放，保证纵横比不变
        img = letterbox(img_ori, new_shape=self.imgsz)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        # 图片也设置为Float16
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # 没有batch_size的话则在最前面添加一个轴
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img

    def _add_classify(self,bboxes,frame):
        new_label = np.zeros(len(bboxes))
        for index,i in enumerate(bboxes) :
            i = list(map(int,i))
            x1,x2,y1,y2 = i[0],i[2],i[1],i[3]
            img_patch = frame[y1:y2,x1:x2]
            img_patch = self.letter.cv2_letterbox_image_by_warp(img_patch,(384,384))
            cv2.imwrite('a.jpg', img_patch)
            img_patch = self.transfrom(img_patch)
            img_patch = torch.unsqueeze(img_patch,0)  # 这里也可以考虑把所有框放到一起一批处理
            if self.device.type !='cpu':
                img_patch.to(self.device)
            output = self.modelc(img_patch)
            if self.device.type !='cpu':
                pred = output.data.max(1,keepdim=True)[1].cpu().numpy()[0][0]
            else:
                pred = output.data.max(1,keepdim=True)[1].numpy()[0][0]
            new_label[index] = pred
        return new_label

    def draw_result(self, img:np.ndarray, result:tuple,line_width=None):
        im = img.copy()
        boxes,classes,scores = result
        lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2) 
        for i,box in enumerate(boxes):
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            score = float(scores[i])
            class_id = int(classes[i])
            label = self.cfg.plot.categories[class_id]
            color = self.cfg.plot.colors[class_id]
            cv2.rectangle(im, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
            text = label+f':{score:.2f}'
            tf = max(lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(text, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(im, p1, p2, (128,128,128), -1, cv2.LINE_AA)  # filled
            cv2.putText(im, text, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 3, (255,255,255),
                        thickness=tf, lineType=cv2.LINE_AA)
        return im



if __name__ == "__main__":
    import time
    # img_path = 'WechatIMG4638.jpeg'
    # img = cv2.imread(img_path)
    videoWriter = cv2.VideoWriter('video.mp4', cv2.VideoWriter_fourcc('M','P','E','G'), 25, (1920,1080))

    detector = Detection('yolo5', 'cpu', cfg)

    video_path = ''
    video = Video(video_path)
    frame = video.get_next_frame()

    t1 = time.time()
    if frame is not None:
        bboxes,labels,scores,_ = detector.detect(frame)
        if bboxes is not None:
            #plot 
            result_img = detector.draw_result(frame,(bboxes,labels,scores))
        else:
            result_img = frame
            
        t2 = time.time()
        fps = round(1/(t2-t1),2)
        result_img = cv2.putText(result_img, "fps:{}".format(str(fps)), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255))
        videoWriter.write(result_img)
    videoWriter.release()
    # bboxes,labels,scores,_ = detector.detect(img)
    # n = detector.draw_result(img, (bboxes,labels,scores))
    # cv2.imwrite('a.jpg', n)
    