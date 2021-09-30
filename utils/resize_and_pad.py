import cv2
import os
import numpy as np


def resize_and_pad(img, out_size, rgb=(255, 255, 255)):
    """
    # 保持纵横比resize图像，并pad
    :param img: numpy 3D array
    :param out_size: target img size
    :param rgb: default (255,255,255)
    :return: resized and padded img
    """
    h,w = img.shape[0], img.shape[1]
    m = max(w, h)
    ratio = out_size / m
    new_w, new_h = int(ratio * w), int(ratio * h)
    assert new_w > 0 and new_h > 0
    resized = cv2.resize(img, (new_w, new_h))

    # padding
    top = bottom = left = right = 0
    if new_h > new_w:
        left = right = int((new_h - new_w) / 2)
    elif new_h < new_w:
        top = bottom = int((new_w - new_h) / 2)
    else:
        pass
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return padded


def pad_and_resize(img, out_size, rgb=(255, 255, 255)):
    """
    # 保持纵横比resize图像，并pad
    :param img: numpy 3D array
    :param out_size: target img size
    :param rgb: default (255,255,255)
    :return: resized and padded img
    """
    # padding
    h, w = img.shape[0], img.shape[1]
    print(w, h)
    top = bottom = left = right = 0
    if h > w:
        left = right = (h - w) // 2
    elif h < w:
        top = bottom = (w - h) // 2
    else:
        pass
    padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    resized = cv2.resize(padded, (out_size, out_size))
    return resized


if __name__ == '__main__':
    out_size = 448
    img_path = r'D:\set100-70_ori\test_bbox\annotated_images_backup\3Climacograptus pauperatus\image_copy_1.jpg'
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    padded = resize_and_pad(img, out_size)
    cv2.imshow('pad', padded)
    cv2.waitKey(0)
