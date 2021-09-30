# -*- coding: utf-8 -*-
"""
对原图像和CAM热力图按权相加
"""
import os
import cv2
import glob
import numpy as np

resize_dir = r"D:\ESSD2021_CAM\resize"
resize_crop_dir = r"D:\ESSD2021_CAM\resize_crop"
save_dir = r"D:\ESSD2021_CAM\image_add"
category_list = os.listdir(resize_dir)

for category_name in category_list:
    resize_category = os.path.join(resize_dir, category_name)
    resize_crop_category = os.path.join(resize_crop_dir, category_name)
    save_category = os.path.join(save_dir, category_name)
    if not os.path.exists(save_category):
        os.mkdir(save_category)

    for cam_img_path in glob.glob(os.path.join(resize_crop_category, "*layer4.jpg")):
        resize_img_path = ".".join(cam_img_path.split("_crop_layer4.")).replace(resize_crop_category, resize_category)
        cam_img = cv2.imread(cam_img_path)
        resize_img = cv2.imread(resize_img_path)
        merge_img = cv2.addWeighted(cam_img, 0.6, resize_img, 0.4, 0)
        save_img_path = "_cam.".join(resize_img_path.split("_448.")).replace(resize_dir, save_dir)
        print(save_img_path)
        cv2.imwrite(save_img_path, merge_img)
