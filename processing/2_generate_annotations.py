# -*- coding: utf-8 -*-
"""
转换coco annotator的标注数据格式，并生成图像对应的标注信息的json文件(一张图像的所有标注放在一起裁剪)，需要修改的参数：
1. batch_name代表图像的批次
2. source_json_file，target_json_file代表图像的重标注批次
"""
import re
import os
import numpy as np
import json
from pathlib import Path
from multiprocessing import Pool

category_list1 = [  # set100-10
    '1Dicellograptus bispiralis',         # re1
    '1Dicellograptus caduceus',           # re1
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
    '6Dicellograptus szechuanensis',
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
category_list = category_list1
batch_name_list = [path for path in os.listdir(r"D:\标注过的图片已审核") if os.path.isdir(os.path.join(r"D:\标注过的图片已审核", path))]

batch_name = batch_name_list[0]


# 将coco-annotator标注点格式转换为二维坐标点
def convert_coordinate_format(input_list):
    # 转换为二维矩阵
    original_list = np.array(input_list)
    original_list = np.array(original_list)
    coordinates = []    # 缓存整行坐标点
    # 遍历矩阵每一行
    for row in range(original_list.shape[0]):
        list_row = original_list[row]  # 每一组标注点
        coordinate = []    # 缓存单个坐标点
        # 遍历标注点，两两成组
        for j in range(len(list_row)):
            tmp = int(list_row[j])  # 单个坐标点
            # 该组第一个坐标点
            if j == 0:
                coordinate.append(tmp)
                continue
            if j % 2 == 0 and j != 0:
                coordinate.append(tmp)
                continue
            else:
                coordinate.append(tmp)
                coordinates.append(coordinate)
                coordinate = []
                continue
    return coordinates


def generate_function(target_category):
    # 无批次号的类别名
    pure_target_category = target_category
    if re.match(r'^\d{2}', target_category):
        pure_target_category = target_category[2:]
    elif re.match(r'^\d', target_category):
        pure_target_category = target_category[1:]
    # 原始json文件路径
    source_json_file = Path(r"D:\标注过的图片已审核", batch_name, "json文件") / (pure_target_category + ".json")  # 首次
    # source_json_file = Path(r"D:\标注过的图片已审核", batch_name, r"re-annotate-re1\json文件") / (pure_target_category + "_re1.json")    # 重标注1
    # source_json_file = Path(r"D:\标注过的图片已审核", batch_name, r"re-annotate-re2\json文件") / (pure_target_category + "_re2.json")    # 重标注2
    # source_json_file = Path(r"D:\标注过的图片已审核", batch_name, r"re-annotate-re3\json文件") / (pure_target_category + "_re3.json")  # 重标注3
    # 生成的标注框json文件的存储路径
    target_json_file = Path(r"D:\set113_ori\annotation_json") / (target_category + ".json")  # 首次
    # target_json_file = Path(r"D:\set113_ori\re_annotation1\annotation_json") / (target_category + ".json")   # 重标注1
    # target_json_file = Path(r"D:\set113_ori\re_annotation2\annotation_json") / (target_category + ".json")   # 重标注2
    # target_json_file = Path(r"D:\set113_ori\re_annotation3\annotation_json") / (target_category + ".json")  # 重标注3
    txt = source_json_file.read_text(encoding='utf-8')
    json_txt = json.loads(txt)

    # 包含所有被标注为target_category的图片信息
    images = json_txt["images"]
    annotations = json_txt["annotations"]
    # 生成json格式数据并存储
    json_output = []
    for image in images:
        img_path = image["path"]
        img_id = image["id"]
        img_name = image["file_name"]

        # if not re.match("HXU", img_name):  # 匹配单反图片（已舍弃）
        #     continue
        # else:

        img_path = img_path.split("/")[2:5]  # 截取图片相对路径
        img_relative_path = os.path.join(target_category, img_path[-1])  # 图片最终存储的相对路径[类别名, 文件名]

        for annotation in annotations:  # 每个图像的所有多边形标注框一起存储
            single_img = {}
            if annotation['image_id'] == img_id:
                single_img['id'] = img_id
                single_img['relative_path'] = img_relative_path
                single_img['filename'] = img_name
                single_img['bbox'] = annotation['bbox']
                seg = []
                for biaozhu in annotation['segmentation']:
                    seg.append(convert_coordinate_format([biaozhu]))
                    # break
                single_img['segmentation'] = seg  # [[[x1,y1],[x2,y2]], [[x3,y3],[x4,y4]]]
                json_output.append(single_img)
            else:
                continue
        # break

    json_output = json.dumps(json_output)
    with open(str(target_json_file), "w") as f:
        f.write(json_output)


if __name__ == '__main__':
    p = Pool(10)
    for category in category_list:
        p.apply_async(generate_function, args=(category,))
    p.close()
    p.join()
