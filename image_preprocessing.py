import pandas as pd
import numpy as np
import os
import pickle as pkl
import re
import matplotlib.pyplot as plt
import cv2
from skimage import io
import skimage.io
import skimage.measure as skmeas
import sys
import numpy as np
#import SimpleITK as sitk

def calc_size(bbox):
    #print("bbox in calc_size", bbox)
    a = bbox[2] - bbox[0]
    b = bbox[3] - bbox[1]
    return a*b


def stitch_images(x,y):
    diff = abs(x.shape[0] - y.shape[0])
    if x.shape[0] < y.shape[0]:
        z = np.zeros((diff, x.shape[1]))
        x = np.concatenate((x, z), axis=0)
    else:
        if x.shape[0] > y.shape[0]:
            z = np.zeros((diff, y.shape[1]))
            y = np.concatenate((y, z), axis=0)
    img = np.concatenate((x,y), axis=1)
    
    if img.shape[1]>img.shape[0]:
        #print('wide')
        diff = img.shape[1]-img.shape[0]
        z = np.zeros((diff, img.shape[1]))
        img = np.concatenate((img, z), axis=0)
        img = np.array(img, dtype='uint8')

    elif img.shape[0]>img.shape[1]:
        #print('tall')
        diff = img.shape[0]-img.shape[1]
        z1 = np.zeros((img.shape[0], int(diff/2)))
        z2 = np.zeros((img.shape[0], diff - int(diff/2)))
        img = np.concatenate((z2, img, z1), axis=1)
        img = np.array(img, dtype='uint8')
    else:
        #print('square')
        pass
        
    return img

def segment_breast(image_np):
    thresh = 0
    mask = image_np > thresh
    object_labels = skmeas.label(mask)

    some_props = skmeas.regionprops(object_labels)
    bboxes = [{'bbox':one_prop['bbox'], 'centroid':one_prop['centroid'], 'size':calc_size(one_prop['bbox'])} for one_prop in some_props]
    bbox_sizes = [bbox['size'] for bbox in bboxes]
    bbox = bboxes[np.argmax(bbox_sizes)]['bbox']
    #print("bbox", bbox)

    img_crop = image_np[bbox[0]:bbox[2], bbox[1]:bbox[3]]
 

    return img_crop