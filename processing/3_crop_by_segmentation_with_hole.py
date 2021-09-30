# coding=utf-8
"""
根据转换格式后的标注数据对原图像进行裁剪（镂空图像），需要修改的参数：
1. category_list代表图像的批次
"""
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np
import json
from pathlib import Path
import os
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

category_list1 = [  # set100-10
    '1Dicellograptus bispiralis',         # re1, hole
    '1Dicellograptus caduceus',           # re1, hole
    '1Dicellograptus divaricatus salopiensis',
    '1Dicellograptus smithi',             # re1, re2
    '1Dicellograptus undatus',
    '1Dicranograptus irregularis',        # re1
    '1Dicranograptus sinensis',           # re1
    '1Didymograptus jiangxiensis',        # re1
    '1Didymograptus latus tholiformis',   # re1
    '1Didymograptus miserabilis'
]
category_list2 = [  # set100-20
    '2Amplexograptus acusiformis',          # re1
    '2Amplexograptus fusiformis',
    '2Cryptograptus arcticus sinensis',     # re1
    '2Cryptograptus gracilicornis',         # re1
    '2Dicellograptus divaricatus',
    '2Dicranograptus nicholsoni parvangulus',
    '2Dicranograptus ramosus',
    '2Didymograptus euodus',
    '2Didymograptus linearis longus',
    '2Didymograptus saerganensis'
]
category_list3 = [  # set100-30
    '3Climacograptus pauperatus',                   # re1, re2
    '3Cryptograptus arcticus',                      # re1
    '3Cryptograptus marcidus',
    '3Cryptograptus tricornis',                     # re1, re2
    '3Glossograptus briaros',
    '3Glossograptus robustus',
    '3Glyptograptus plurithecatus wuningensis',     # re1
    '3Glyptograptus teretiusculus siccatus',        # re1
    '3Pseudoclimacograptus parvus jiangxiensis',    # re1
    '3Pseudoclimacograptus wannanensis'             # re1, re2, re3
]
category_list4 = [  # set100-40
    '4Diplograptus proelongatus',       # re1
    '4Glyptograptus teretiusculus',     # re1
    '4Jiangxigraptus inculus',
    '4Jishougraptus mui',
    '4Leptograptus flaccidus trentonensis',
    '4Monoclimacis neimengolensis',
    '4Pseudoclimacograptus angulatus',
    '4Pseudoclimacograptus longus',
    '4Pseudoclimacograptus modestus',
    '4Pseudoclimacograptus parvus'
]
category_list5 = [  # set100-50
    '5Amplexograptus disjunctus yangtzensis',
    '5Amplexograptus suni',             # re1
    '5Climacograptus miserabilis',      # re1
    '5Climacograptus supernus',         # re1
    '5Dicellograptus ornatus',          # re1, re2
    '5Diplograptus modestus',
    '5Glyptograptus incertus',
    '5Petalolithus elongatus',
    '5Petalolithus folium',
    '5Streptograptus runcinatus'
]
category_list6 = [  # set100-60
    '6Dicellograptus szechuanensis',    # hole
    '6Diplograptus bohemicus',
    '6Glyptograptus austrodentatus',    # re1, re2
    '6Glyptograptus gracilis',          # re1
    '6Glyptograptus lungmaensis',
    '6Glyptograptus tamariscus',        # re1
    '6Glyptograptus tamariscus linealis',
    '6Glyptograptus tamariscus magnus',
    '6Reteograptus uniformis',
    '6Retiolites geinitzianus'
]
category_list7 = [  # set100-70
    '7Amplexograptus orientalis',
    '7Climacograptus angustatus',       # re1
    '7Climacograptus leptothecalis',    # re1
    '7Climacograptus minutus',          # re1
    '7Climacograptus normalis',
    '7Climacograptus tianbaensis',
    '7Colonograptus praedeubeli',
    '7Diplograptus angustidens',
    '7Diplograptus diminutus',
    '7Rectograptus pauperatus'
]
category_list8 = [  # set100-80
    '8Amplexograptus confertus',
    '8Climacograptus angustus',                 # re1
    '8Climacograptus textilis yichangensis',    # re1
    '8Colonograptus deubeli',
    '8Dicellograptus cf. complanatus',
    '8Diplograptus concinnus',
    '8Pristiograptus variabilis',
    '8Pseudoclimacograptus demittolabiosus',
    '8Pseudoclimacograptus formosus',
    '8Rectograptus abbreviatus'
]
category_list9 = [  # set100-90
    '9Akidograptus ascensus',
    '9Amplexograptus cf. maxwelli',
    '9Cardiograptus amplus',
    '9Climacograptus bellulus',
    '9Climacograptus hastatus',
    '9Glyptograptus dentatus',
    '9Glyptograptus elegans',
    '9Glyptograptus elegantulus',
    '9Orthograptus calcaratus',
    '9Trigonograptus ensiformis'
]
category_list10 = [  # set100-96
    '10Demirastrites triangulatus',
    '10Dicellograptus tumidus',
    '10Dicellograptus turgidus',
    '10Paraorthograptus pacificus',     # re1
    '10Paraorthograptus simplex',
    '10Spirograptus turriculatus'
]
category_list11 = [  # set100
    '11Appendispinograptus venustus',   # re1
    '11Nicholsonograptus fasciculatus',
    '11Nicholsonograptus praelongus',
    '11Paraorthograptus longispinus'
]
category_list12 = [  # set105
    '12Cryptograptus tricornis (Juvenile)',
    '12Phyllograptus anna',
    '12Rastrites guizhouensis',     # re1, re2
    '12Tangyagraptus typicus',      # re1
    '12Yinograptus grandis'
]
category_list13 = [  # set110
    '13Coronograptus cyphus',               # re1
    '13Cystograptus vesiculosus',           # re1
    '13Normalograptus extraordinarius',     # re1, re2
    '13Normalograptus persculptus',         # re1, re2
    '13Parakidograptus acuminatus'
]
category_list14 = [  # set114
    '14Diceratograptus mirus',
    '14Lituigraptus convolutus',
    '14Paraplegmatograptus connectus',    # re1
    '14Pararetiograptus regularis',
]
category_list = category_list1 + category_list2 + category_list3 + category_list4 + category_list5 + category_list6 + \
                category_list7 + category_list8 + category_list9 + category_list10 + category_list11 + category_list12 +\
                category_list13 + category_list14
category_list = category_list1


def crop_function(category_name):
    target_category = category_name
    # 标注框json文件的路径
    source_json_file = Path(r"D:\set113_ori\annotation_json") / (target_category + ".json")  # 首次
    # source_json_file = Path(r"D:\set113_ori\re_annotation1\annotation_json") / (target_category + ".json")   # 重标注
    # source_json_file = Path(r"D:\set113_ori\re_annotation2\annotation_json") / (target_category + ".json")  # 重标注二
    txt = source_json_file.read_text(encoding='utf-8')
    json_txt = json.loads(txt)
    # 被裁剪图像的路径
    convert_dir = Path(r"D:\set113_ori\images") / target_category  # 首次
    # convert_dir = Path(r"D:\set113_ori\re_annotation1\images") / target_category  # 重标注
    # convert_dir = Path(r"D:\set113_ori\re_annotation2\images") / target_category  # 重标注二
    # 裁剪后图像的存储路径
    save_dir = r'D:\set113_ori\annotated_images_backup'  # 首次
    # save_dir = r'D:\set113_ori\re_annotation1\annotated_images'    # 重标注
    # save_dir = r'D:\set113_ori\re_annotation2\annotated_images'  # 重标注二

    if not (Path(save_dir) / target_category).exists():
        (Path(save_dir) / target_category).mkdir()

    assert os.path.isfile(str(source_json_file)) is True, print(source_json_file)
    assert os.path.isdir(str(convert_dir)) is True, print(convert_dir)
    assert os.path.isdir(str(save_dir)) is True, print(save_dir)

    for image in convert_dir.iterdir():
        img_name = image.name
        num = 1
        for image_info in json_txt:
            if image_info['filename'] == img_name:
                seg_list = image_info['segmentation']
                seg_cache = []  # 缓存镂空标注框
                max_seg = []  # 缓存最大的标注框，即为外边缘框
                img_relative_path = image_info['relative_path']

                # opencv读取包含中文路径的图像
                crop_num = 0
                image = cv2.imdecode(np.fromfile(str(convert_dir / img_name), dtype=np.uint8), -1)
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                height = image.shape[0]
                width = image.shape[1]
                # 和原始图像一样大小的0矩阵，作为mask掩模
                mask = np.zeros(image.shape[:2], dtype="uint8")
                # print(pts)
                for seg in seg_list:
                    if len(seg) > len(max_seg):  # 对于镂空的笔石图像，找到其最大的标注框
                        max_seg = seg
                for seg in seg_list:  # 缓存除了最大的标注以外的标注框，用于图像相减
                    if not seg == max_seg:
                        seg_cache.append(seg)
                # 对于外标注框，在mask上将多边形区域填充为白色（255）
                pts = np.array([max_seg])
                cv2.polylines(mask, pts, 1, 255)
                cv2.fillPoly(mask, pts, 255)
                # 对于镂空标注框，在mask上将多边形区域填充为黑色（0）
                for seg1 in seg_cache:
                    pts = np.array([seg1])
                    cv2.polylines(mask, pts, 1, 0)
                    cv2.fillPoly(mask, pts, 0)
                # print(img_name, len(seg_list))
                # 裁剪后的目标图像destination
                dst = cv2.bitwise_and(image, image, mask=mask)
                # 添加白色背景
                bg = np.ones_like(image, np.uint8) * 255  # bg的多边形区域为0，背景区域为255
                cv2.bitwise_not(bg, bg, mask=mask)
                dst2 = bg + dst

                # plt.rcParams['figure.dpi'] = 1
                # plt.rcParams['figure.figsize'] = (width, height)
                # plt.imshow(dst2)
                # plt.show()
                # plt.axis('off')

                save_path = os.path.join(save_dir, img_relative_path)
                if os.path.exists(save_path):
                    save_path = os.path.join(save_dir, target_category,
                                             img_name.split('.')[0] + '_copy_' + str(num) + '.jpg')
                    num += 1
                    print("标注框数目： %d" % len(image_info['segmentation']), "重名，裁剪保存：", save_path)
                    # plt.imsave(save_path, dst2)
                    is_success, im_buf_arr = cv2.imencode('.jpg', dst2)
                    im_buf_arr.tofile(save_path)
                else:
                    print("标注框数目： %d" % len(image_info['segmentation']), "裁剪保存：", save_path)
                    # plt.imsave(save_path, dst2)
                    is_success, im_buf_arr = cv2.imencode('.jpg', dst2)
                    im_buf_arr.tofile(save_path)
                plt.close()

            else:
                continue


if __name__ == '__main__':
    # 多进程同时处理多个属种
    p = Pool(10)
    for path in category_list:
        p.apply_async(crop_function, args=(path,))
    p.close()
    p.join()
