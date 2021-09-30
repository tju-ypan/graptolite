"""
img_tools.py
"""
import numpy as np
import cv2
from math import ceil


def standard_resize(image, max_side):
    if image is None:
        return None, None, None
    original_h, original_w, _ = image.shape
    # if all(side < max_side for side in [original_h, original_w]):
    #     return image, original_h, original_w
    aspect_ratio = float(np.amax((original_w, original_h)) / float(np.amin((original_h, original_w))))

    if original_w >= original_h:
        new_w = max_side
        new_h = max_side / aspect_ratio
    else:
        new_h = max_side
        new_w = max_side / aspect_ratio

    new_h = int(new_h)
    new_w = int(new_w)
    resized_image = cv2.resize(image, (new_w, new_h))
    return resized_image, new_w, new_h


def get_image(img_path, image_new_size):

    np_img = cv2.imread(img_path)
        # np_img = np.array(img)

    if np_img is None:
        return None, None, None, None, None

    small_image, x1, y1 = standard_resize(np_img, image_new_size)
    if small_image is None:
        return None, None, None, None, None

    dx = int(ceil((image_new_size - x1) / 2))
    dy = int(ceil((image_new_size - y1) / 2))
    return small_image, x1, y1, dx, dy


if __name__ == '__main__':
    img_path = r'D:\ESSD2021_CAM\resize\1Dicellograptus bispiralis\103201_448.jpg'
    result = get_image(img_path, 50)
