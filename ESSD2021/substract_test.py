import cv2 as cv
import numpy as np

img1 = cv.imread(r"D:\ESSD2021\9Glyptograptus elegantulus\79398.jpg")
img2 = cv.imdecode(np.fromfile(r"D:\标注过的图片已审核\2020.08.17第九批已审核\Glyptograptus elegantulus\79398\HXU_3646.jpg", dtype=np.uint8), -1)
if img1.shape[0] == img2.shape[1] and img1.shape[1] == img2.shape[0]:
    print(img1.sum(), img2.sum())
    img1 = img1.reshape(img1.shape[1], img1.shape[0], -1)
    print(img1.shape, img2.shape)
    print(img1.sum(), img2.sum())

