import numpy as np
import cv2
from pathlib import Path
import json
import matplotlib.pyplot as plt
import os


# 将标注框格式转换为坐标点格式供opencv处理
def convert_coordinate_format(input_list):
    original_list = np.array(input_list)
    original_list = np.array(original_list)
    coordinates = []
    for row in range(original_list.shape[0]):  # 遍历二维矩阵每一行
        list_row = original_list[row]  # 每一组坐标原始格式
        coordinate = []
        for j in range(len(list_row)):  # 遍历坐标点，组成二维坐标
            tem = int(list_row[j])
            if j == 0:
                coordinate.append(tem)
                continue
            if j % 2 == 0 and j != 0:
                coordinate.append(tem)
                continue
            else:
                coordinate.append(tem)
                coordinates.append(coordinate)
                coordinate = []
                continue
    return coordinates


category_name = "7Pseudoclimacograptus formosus"
img_name = "HXU_1205.jpg"
img_path = Path(r'D:\标注过的图片已审核\2020.07.13SET-10第七批 已审核\Pseudoclimacograptus formosus\9771') / img_name
json_file_path = Path(r'D:\set100-70_ori\annotation_json')


if __name__ == '__main__':
    # 标注框json文件路径
    source_json_file = json_file_path / (category_name + ".json")
    txt = source_json_file.read_text(encoding='utf-8')
    json_txt = json.loads(txt)

    for image_info in json_txt:
        if image_info['filename'] == img_name:
            seg_list = image_info['segmentation']
            seg_cache = []  # 缓存空心多边形框
            max_seg = []

            image = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), -1)
            (r, g, b) = cv2.split(image)
            image = cv2.merge([b, g, r])  # plt和opencv读取的图像颜色通道是相反的
            width = image.shape[0]
            height = image.shape[1]
            # 和原始图像一样大小的0矩阵，作为mask
            im = np.zeros(image.shape[:2], dtype="uint8")
            # print(pts)
            for seg in seg_list:
                if len(seg) > len(max_seg):  # 标注点过少的话，认为是误操作，不予考虑
                    max_seg = seg
            for seg in seg_list:
                if not seg == max_seg:
                    seg_cache.append(seg)
            pts = np.array([max_seg])
            cv2.polylines(im, pts, 1, 255)
            cv2.fillPoly(im, pts, 255)
            print(img_name, len(seg))
            for seg1 in seg_cache:
                pts = np.array([seg1])
                cv2.polylines(im, pts, 1, 0)
                cv2.fillPoly(im, pts, 0)
            mask = im
            masked = cv2.bitwise_and(image, image, mask=mask)
            array = np.zeros((masked.shape[0], masked.shape[1], 4), np.uint8)

            array[:, :, 0:3] = masked
            array[:, :, 3] = 0
            array[:, :, 3][np.where(array[:, :, 0] > 2)] = 255
            array[:, :, 3][np.where(array[:, :, 1] > 2)] = 255
            array[:, :, 3][np.where(array[:, :, 2] > 2)] = 255

            image_1 = np.array(array)
            plt.rcParams['figure.dpi'] = 1
            plt.rcParams['figure.figsize'] = (width, height)
            plt.imshow(image_1)
            plt.axis('off')
            plt.savefig(os.path.join(r'C:\Users\admin\Desktop\test', img_name))
