#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 * Ltd. All rights reserved.
#
#   Editor      : VSCode
#   File name   : count.py
#   Author      : hx
#   Created date: 2021-12-28 13:52:53
#   Description : A module for pedestrian crossing detection
#
#================================================================
import cv2 as cv
import numpy as np


class Compute(object):
    # 计算两个向量的余弦角度
    def get_cos_similar(self,v1,v2):
        num = np.dot(v1,v2)
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        return (num/denom) if denom !=0 else 0

    def get_normal_vector(self,v):
        tana = v[1]/v[0]
        a = np.arctan(tana) #为弧度值
        new_a = a + 3.14/2
        new_v = []
        new_v.append(np.cos(new_a))
        new_v.append(np.sin(new_a))
        return new_v


class CrossDetection(object):
    # 目标撞线判断，line为界限，in_direction_point为靠近舱门一侧
    def __init__(self,line:list,in_direction_point:list):
        # line's param
        self.line = line # [[x1,y1],[x2,y2]]
        self.line_xy = [int((self.line[0][0]+self.line[1][0])/2),int((self.line[0][1]+self.line[1][1])/2)] # center point coordinates of line
        self.a,self.b = self._solve_equation_of_line()
        self.in_direction_point = in_direction_point
        self.computer = Compute()
        self.in_direction = self._init_direction() # a two-dimensional vector 
        self.count_dict={
            'in':{
                'jizu':0,'qingjie':0
            },
            'out':{
                'jizu':0,'qingjie':0
            }
        }
        

    def _init_direction(self):
        # 通过计算in_vector与法线之间的关系，确定法线的某个方向是进入方向,与法线cos为正即为进入
        in_vertor = np.array([self.in_direction_point[0]-self.line_xy[0],self.in_direction_point[1]-self.line_xy[1]])
        zero_normal_vertor = self.computer.get_normal_vector(np.array([self.line[0][0]-self.line[1][0],self.line[0][1]-self.line[1][1]])) 
        # 
        vector_cos_value = self.computer.get_cos_similar(in_vertor, zero_normal_vertor)
        result = [None,None]

        if vector_cos_value <0:
            zero_normal_vertor[0] = -zero_normal_vertor[0]
            zero_normal_vertor[1] = -zero_normal_vertor[1]
        if vector_cos_value != 0:
            result[0] = zero_normal_vertor[0]
            result[1] = zero_normal_vertor[1]
            # norm
            result[0] = result[0]/np.sqrt(result[0]*result[0]+result[1]*result[1])
            result[1] = result[1]/np.sqrt(result[0]*result[0]+result[1]*result[1])
        # 最终如果返回none的话，应当有一些告警或提示
        return result
            

    def detect(self,point1,point2,class_id,track_id,img=None):
        # 判断当前目标的方向
        move_vector = [point2[0]-point1[0],point2[1]-point1[1]]
        vector_cos_value = self.computer.get_cos_similar(move_vector,self.in_direction)
        with open('a.txt','a') as f:
            f.write("tarck_id:{} class:{}".format(str(track_id),str(class_id))+str(vector_cos_value)+'\n')
        if vector_cos_value>0:
            # 进入
            if class_id==0:
                self.count_dict['in']['jizu']+=1
            elif class_id==1:
                self.count_dict['in']['qingjie']+=1
            else:
                pass
        elif vector_cos_value<0:
            if class_id==0:
                self.count_dict['out']['jizu']+=1
            elif class_id==1:
                self.count_dict['out']['qingjie']+=1
            else:
                pass
        else:
            pass
        cv.putText(img,"fa vec:{} \n cos :{}".format(self.in_direction,vector_cos_value),(50,500),cv.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),2)
    

    def pass_line_or_not(self,point1,point2,class_id,track_id,img=None):
        # y=ax+b 如果两个点的数值异号的话，就是撞线了
        if self.a is not None:
            temp_1 = point1[1]-self.a*point1[0] - self.b
            temp_2 = point2[1]-self.a*point2[0] - self.b
            if temp_1*temp_2>0:
                result=False
            elif temp_1*temp_2<0:
                result=True
            else:
                if abs(temp_1-temp_2)>0:
                    result=True
                else:
                    result=False
        else:
            #对于x=a的这种线，只能用这种判断
            if (point1[0]-self.line[0][0])*(point2[0]-self.line[0][0])<0:
                result=True
            else:
                result=False

        if result:
            self.detect(point1, point2, class_id, track_id,img)

        return result

    def _solve_equation_of_line(self):
        # 求解y=ax+b的参数
        if self.line[0][0]==self.line[1][0]:
            # y = b
            answers = [None,None]
            print('球球别画竖直线')
        elif self.line[0][1]==self.line[1][1]:
            answers = [0,self.line[0][1]]
        else:
            coefficients=[[self.line[0][0],1],[self.line[1][0],1]]
            dependents=[self.line[0][1],self.line[1][1]]
            print(coefficients,dependents)
            answers = np.linalg.solve(coefficients,dependents)
        return [answers[0],answers[1]]

        

if __name__ == "__main__":
    cross = CrossDetection([[0,100],[100,95]],[10,10])
    print(cross.in_direction)
    # print(cross.a,cross.b)

    
    