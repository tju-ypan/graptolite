# coding=utf-8
"""
根据bbox标注框对图像进行裁剪
"""
import matplotlib.pyplot as plt
import re
import cv2
from PIL import Image
import numpy as np
import json
from pathlib import Path
import os
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

category_list3 = [  # set100-30
    '3Climacograptus pauperatus',
    '3Cryptograptus arcticus',
    '3Cryptograptus marcidus',
    '3Cryptograptus tricornis',
    '3Glossograptus briaros',
    '3Glossograptus robustus',
    '3Glyptograptus plurithecatus wuningensis',
    '3Glyptograptus teretiusculus siccatus',
    '3Pseudoclimacograptus parvus jiangxiensis',
    '3Pseudoclimacograptus wannanensis'
]
category_list7 = [  # set100-70
    '7Amplexograptus orientalis',
    '7Climacograptus angustatus',
    '7Climacograptus leptothecalis',
    '7Climacograptus minutus',
    '7Climacograptus normalis',
    '7Climacograptus tianbaensis',
    '7Colonograptus praedeubeli',
    '7Diplograptus angustidens',
    '7Diplograptus diminutus',
    '7Rectograptus pauperatus'
]

category_list = category_list3 + category_list7
outsize = 448.0


def resize_and_pad(img, out_size, rgb=(255, 255, 255)):
    """
    # 保持纵横比resize图像，并pad
    :param img: numpy 3D array
    :param out_size: target img size
    :param rgb: default (255,255,255)
    :return: resized and padded img
    """
    h, w = img.shape[0], img.shape[1]
    m = max(w, h)
    ratio = out_size / m
    new_w, new_h = int(ratio * w), int(ratio * h)
    assert new_w > 0 and new_h > 0
    resized = cv2.resize(img, (new_w, new_h))

    # padding
    top = bottom = left = right = 0
    if new_h > new_w:
        left = right = (new_h - new_w) // 2
    elif new_h < new_w:
        top = bottom = (new_w - new_h) // 2
    else:
        pass
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return resized


def crop_function(category_name):
    target_category = category_name
    """标注框json文件的路径"""
    source_json_file = Path(r"D:\set113_ori\annotation_json") / (target_category + ".json")  # 首次
    source_json_file = Path(r"D:\set113_ori\re_annotation1\annotation_json") / (target_category + ".json")   # 重标注1
    source_json_file = Path(r"D:\set113_ori\re_annotation2\annotation_json") / (target_category + ".json")  # 重标注2
    source_json_file = Path(r"D:\set113_ori\re_annotation3\annotation_json") / (target_category + ".json")  # 重标注3
    txt = source_json_file.read_text(encoding='utf-8')
    json_txt = json.loads(txt)
    """被裁剪图像的路径"""
    convert_dir = Path(r"D:\set113_ori\images") / target_category  # 首次
    convert_dir = Path(r"D:\set113_ori\re_annotation1\images") / target_category  # 重标注1
    convert_dir = Path(r"D:\set113_ori\re_annotation2\images") / target_category  # 重标注2
    convert_dir = Path(r"D:\set113_ori\re_annotation3\images") / target_category  # 重标注3
    """裁剪后图像的存储路径"""
    save_dir = r'D:\set113_ori\test_bbox\annotated_images_backup'  # 首次
    save_dir = r'D:\set113_ori\test_bbox\annotated_images_backup_re1'    # 重标注
    save_dir = r'D:\set113_ori\test_bbox\annotated_images_backup_re2'  # 重标注
    save_dir = r'D:\set113_ori\test_bbox\annotated_images_backup_re3'  # 重标注

    if not (Path(save_dir) / target_category).exists():
        (Path(save_dir) / target_category).mkdir()

    try:
        if not source_json_file.exists():
            raise FileNotFoundError(str(source_json_file))
        if not convert_dir.exists():
            raise FileNotFoundError(str(convert_dir))
        if not Path(save_dir).exists():
            raise FileNotFoundError(str(save_dir))
    except FileNotFoundError as e:
        print(repr(e))

    for image in convert_dir.iterdir():
        img_name = image.name
        # if re.match("HXU", img_name):   # 匹配单反图片
        #     continue
        num = 1
        for image_info in json_txt:
            if image_info['filename'] == img_name:
                img_bbox = image_info['bbox']
                # bbox的左上顶点横坐标，纵坐标，宽，高
                x, y, w, h = img_bbox
                x, y, w, h = int(x), int(y), int(w), int(h)
                img_relative_path = image_info['relative_path']

                # 读取包含中文路径的图像
                img = cv2.imdecode(np.fromfile(str(convert_dir / img_name), dtype=np.uint8), -1)

                # 画标注框
                # image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255))

                # 裁剪
                new_width = w
                new_high = h
                margin = int(abs(new_width - new_high))
                leftmost = x
                rightmost = x + w
                highest = y
                lowest = y + h
                # h,w变为整体图像的高和宽
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                h, w = img_gray.shape

                # 笔石区域的高>宽
                if new_high > new_width:
                    new_img = img[highest:lowest,
                              leftmost - 10 if leftmost >= 10 else 0:rightmost + 10 if rightmost + 10 <= w else w]

                # 笔石区域的高<宽
                elif new_high < new_width:
                    new_img = img[highest - 10 if highest >= 10 else 0:lowest + 10 if lowest + 10 <= h else h,
                              leftmost:rightmost]
                # 笔石区域的高宽刚好相等
                else:
                    new_img = img[highest:lowest, leftmost:rightmost]

                # reshape
                new_img = resize_and_pad(new_img, outsize)
                # if not new_img.shape[0] == new_img.shape[1] == outsize:
                #     new_img = cv2.resize(new_img, (int(outsize), int(outsize)))

                # save
                save_path = os.path.join(save_dir, img_relative_path)
                if os.path.exists(save_path):
                    save_path = os.path.join(save_dir, target_category,
                                             img_name.split('.')[0] + '_copy_' + str(num) + '.jpg')
                    num += 1
                    print(len(image_info['segmentation']), "重名，裁剪保存：", save_path)
                    cv2.imwrite(save_path, new_img)
                else:
                    print("标注框数目： %d" % len(image_info['segmentation']), "裁剪保存：", save_path)
                    cv2.imwrite(save_path, new_img)
            else:
                continue
    print("进程{}结束".format(category_name))


if __name__ == '__main__':
    # 多进程
    p = Pool(10)
    for path in category_list:
        p.apply_async(crop_function, args=(path,))
    p.close()
    p.join()
