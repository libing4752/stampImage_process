#!/usr/bin/env python
"""this program is for cutting the backgram edge in splited Stamp pics,and rotate image to right angles
   warning : This program is a demo program with no checks
   using method :
       image_split_and_rotate.py input_img_path output_img_path
       example: image_split_and_rotate.py ./data/1/split_1.jpg ./res.jpg"""
    

import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np

MAX_ANGLE = 15

def read_image(img_path):
    raw_img = cv2.imread(img_path)
    return raw_img

def find_bound(img):
    thread = 0.2
    w,h = img.shape[:2]
    window_width = 3
    w_bound = []
    h_bound = []
    for i in range(w-window_width):
        if np.sum(img[i:i+window_width,:] > 0) > w * thread * window_width:
            w_bound.append(i + window_width / 2)
            break
        w_bound.append(0)
    for i in range(w-1,window_width,-1):
        if np.sum(img[i-window_width:i,:] > 0) > w * thread * window_width:
            w_bound.append(i -window_width / 2)
            break
        w_bound.append(w-1)
    for i in range(h-window_width):
        if np.sum(img[:,i:i+window_width] > 0) > h * thread * window_width:
            h_bound.append(i +window_width / 2)
            break
        h_bound.append(0)
    for i in range(h-1, window_width, -1):
        if np.sum(img[:,i-window_width:i] > 0) > h * thread * window_width:
            h_bound.append(i -window_width / 2)
            break
        h_bound.append(h-1)
    return w_bound, h_bound

def binary_image(image_orig):
    w, h = image_orig.shape[:2]
    image = cv2.cvtColor(image_orig,cv2.COLOR_BGR2GRAY)
    window_size = (min(11, (2 * (w / 100) + 1)), min(11, (2 * (h / 100) + 1)))
    var = int(sum(window_size))
    image = cv2.GaussianBlur(image, window_size, 3, var)
    sobelX = cv2.Sobel(image,cv2.CV_64F,1,0)
    sobelY = cv2.Sobel(image,cv2.CV_64F,0,1)
    image = np.abs((sobelX + sobelY) / 2)
    image = image.astype("uint8")
    thres,image = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return image

def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = w // 2,  h // 2
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    image_res = cv2.warpAffine(image, M, (w, h))
    return image_res

def clip_bound(image_org):
    image = binary_image(image_org)
    w_bound, h_bound = find_bound(image)
    angle = cv2.minAreaRect(np.column_stack(np.where(image > 0)))
    angle = -angle[-1]
    if angle > MAX_ANGLE :
        angle = 0
    image = rotate_image(image, angle)
    image_res = rotate_image(image_orig, angle)
    w_bound, h_bound = find_bound(image)
    image_res = image_res[w_bound[0]:w_bound[1] + 1, h_bound[0]:h_bound[1] + 1]
    image_binary = image
    return image_res, image_binary

def debug_image():
    img_path = sys.argv[1]
    image_orig = read_image(img_path)
    image_res, image_binary = clip_bound(image_orig) 
    plt.subplot(221)
    plt.imshow(image_res)
    plt.subplot(222)
    plt.imshow(image_orig)
    plt.subplot(223)
    plt.imshow(image_binary)
    plt.subplot(224)
    plt.imshow(binary_image(image_res))

    plt.show()


if __name__ == "__main__":
    img_path_orig = sys.argv[1]
    img_path_output = sys.argv[2]
    image_orig = read_image(img_path_orig)
    image_res, image_binary = clip_bound(image_orig) 
    cv2.imwrite(img_path_output, image_res)  
 
