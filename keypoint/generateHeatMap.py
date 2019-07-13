# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 15:08:33 2019
参考部分 https://blog.csdn.net/qq_21033779/article/details/84840307
@author: KaiZhu
"""

import pandas as pd;
import os;
import cv2;
from PIL import Image;
import matplotlib.pyplot as plt
import glob
import numpy as np
import math
import scipy.misc

dataPath = r'D:\various project\root_analysis\rootImage';
savePath = r'D:\various project\root_analysis\rootImage\keypoint';
maskPath = dataPath + '\mask' ;       #图像库路径
imgPath = dataPath + '\root';       # 图像库路径
HeatMapPath = savePath+'\heatmap';
keyPointPath = savePath + '\keypointsCSV'; 

def getKeyPoints(name):
    endPoint_csv = pd.read_csv(name + '_endpoint.csv',usecols=['region_shape_attributes']);
    bifurcationPoint_csv = pd.read_csv(name + '_bifurcationPoint.csv',usecols=['region_shape_attributes']);
    crossPoint_csv = pd.read_csv(name + '_crossPoint.csv',usecols=['region_shape_attributes']);
    
    endPoints = np.zeros((endPoint_csv.shape[0],2), dtype=np.int)
    bifurcationPoints = np.zeros((bifurcationPoint_csv.shape[0],2), dtype=np.int)
    crossPoints = np.zeros((crossPoint_csv.shape[0],2), dtype=np.int)
    for index, row in endPoint_csv.iterrows():
        point = eval(row["region_shape_attributes"]);
        endPoints[index][0] = point['cx'];
        endPoints[index][1] = point['cy'];    
    for index, row in bifurcationPoint_csv.iterrows():
        point = eval(row["region_shape_attributes"]);
        bifurcationPoints[index][0] = point['cx'];
        bifurcationPoints[index][1] = point['cy'];
    for index, row in crossPoint_csv.iterrows():
        point = eval(row["region_shape_attributes"]);
        crossPoints[index][0] = point['cx'];
        crossPoints[index][1] = point['cy'];
        
    return {'endPoints':endPoints, 'bifurcationPoints':bifurcationPoints, 'crossPoints':crossPoints}

def get_heatmap(annos, height, width):
    """
    Parameters
    - annos： 关键点列表 [
                            [[12,10],[10,30],....19个]，#某一个人的
                            [[xx,xx],[aa,aa],....19个]，#另外一个人的
                        ]
    - heigth：图像的高
    - width: 图像的宽
    Returns
    - heatmap: 热图
    """
 
    # 3种特征点
    num_joints = ['endPoints' , 'bifurcationPoints' , 'crossPoints' ];
 
    # the heatmap for every joints takes the maximum over all people
    joints_heatmap = np.zeros((len(num_joints), height, width), dtype=np.float32)
 
    # among all people
    for channel, joint in enumerate(annos):
        # generate heatmap for every keypoints
        # loop through all people and keep the maximum
 
        for i, points in enumerate(annos[joint]):
            if points[0] < 0 or points[1] < 0:
                continue
            joints_heatmap = put_heatmap(joints_heatmap, channel, points, 8.0)
      
    # 0: joint index, 1:y, 2:x
    joints_heatmap = joints_heatmap.transpose((1, 2, 0))
    return joints_heatmap 
#    #保存为图片
#    scipy.misc.imsave('outfile.jpg', joints_heatmap);
#    
#    # background
#    joints_heatmap[:, :, -1] = np.clip(1 - np.amax(joints_heatmap, axis=2), 0.0, 1.0)
# 
#    mapholder = []
#    for i in range(0, len(num_joints)):
#        a = cv2.resize(np.array(joints_heatmap[:, :, i]), (height, width))
#        mapholder.append(a)
#    mapholder = np.array(mapholder)
#    joints_heatmap = mapholder.transpose(1, 2, 0)

#    return joints_heatmap.astype(np.float16)
 
 
def put_heatmap(heatmap, plane_idx, center, sigma):
    """
    Parameters
    -heatmap: 热图（heatmap）
    - plane_idx：关键点列表中第几个关键点（决定了在热图中通道）
    - center： 关键点的位置
    - sigma: 生成高斯分布概率时的一个参数
    Returns
    - heatmap: 热图
    """
 
    center_x, center_y = center  #mou发
    _, height, width = heatmap.shape[:3]
 
    th = 4.6052
    delta = math.sqrt(th * 2)#3 sigma区间
    #掩模左上角
    x0 = int(max(0, center_x - delta * sigma + 0.5))
    y0 = int(max(0, center_y - delta * sigma + 0.5))
    #掩模右下角
    x1 = int(min(width - 1, center_x + delta * sigma + 0.5))
    y1 = int(min(height - 1, center_y + delta * sigma + 0.5))
 
    exp_factor = 1 / 2.0 / sigma / sigma
 
    ## fast - vectorize
    arr_heatmap = heatmap[plane_idx, y0:y1 + 1, x0:x1 + 1]#掩模范围现有值
    y_vec = (np.arange(y0, y1 + 1) - center_y)**2  # y1 included
    x_vec = (np.arange(x0, x1 + 1) - center_x)**2
    xv, yv = np.meshgrid(x_vec, y_vec)
    arr_sum = exp_factor * (xv + yv)
    arr_exp = np.exp(-arr_sum)
    arr_exp[arr_sum > th] = 0
    heatmap[plane_idx, y0:y1 + 1, x0:x1 + 1] = np.maximum(arr_heatmap, arr_exp)#取两者较大的
    return heatmap

def main():
    for pidImage in glob.glob(maskPath + "/*.jpg"):
        print(pidImage)
        rootImg = Image.open(pidImage);
        
        keyPoints = getKeyPoints(keyPointPath + pidImage[47:-4]);
        heatMap = get_heatmap(keyPoints, rootImg.size[1], rootImg.size[0]);
        scipy.misc.imsave(HeatMapPath + pidImage[47:], heatMap);

if __name__ == '__main__':   
    main()






    
    