# -*- coding: utf-8 -*-
"""
将标注好的图像按照类别、分批次提取划分到target_dir中，需要修改三个参数：
1. category_list和batch_name代表图像的批次
2. target_dir，source_json_file，img_path代表图像的重标注批次
"""
import os
import re
import json
import shutil

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
batch_name_list = [path for path in os.listdir(r"D:\标注过的图片已审核") if os.path.isdir(os.path.join(r"D:\标注过的图片已审核", path))]

# 需要修改的变量
category_list = category_list1
batch_name = category_list[0]

def helper(target_category):
    # 无批次号的类别名
    pure_target_category = target_category
    if re.match(r'^\d{2}', target_category):
        pure_target_category = target_category[2:]
    elif re.match(r'^\d', target_category):
        pure_target_category = target_category[1:]
    # 图像的目标存储路径
    target_dir = Path(r"D:\set113_ori\images") / target_category  # 首次
    # target_dir = Path(r"D:\set113_ori\re_annotation1\images") / target_category  # 重标注1
    # target_dir = Path(r"D:\set113_ori\re_annotation2\images") / target_category  # 重标注2
    # target_dir = Path(r"D:\set113_ori\re_annotation3\images") / target_category  # 重标注3
    if not target_dir.exists():
        target_dir.mkdir()
    # coco annotator生成的标注数据（json文件）路径
    source_json_file = Path(r"D:\标注过的图片已审核", batch_name, "json文件") / (pure_target_category + ".json")  # 首次
    # source_json_file = Path(r"D:\标注过的图片已审核", batch_name, r"re-annotate-re1\json文件") / (pure_target_category + "_re1.json")  # 重标注1
    # source_json_file = Path(r"D:\标注过的图片已审核", batch_name, r"re-annotate-re2\json文件") / (pure_target_category + "_re2.json")  # 重标注2
    # source_json_file = Path(r"D:\标注过的图片已审核", batch_name, r"re-annotate-re3\json文件") / (pure_target_category + "_re3.json")  # 重标注3
    try:
        if not os.path.isfile(source_json_file):
            raise Exception("原始json文件不存在：", source_json_file)
    except Exception as e:
        print(str(e))
    txt = source_json_file.read_text(encoding='utf-8')
    json_txt = json.loads(txt)
    # 包含所有被标注为target_category的图片信息
    images = json_txt["images"]

    # img_list包含该类别的所有图片路径：类别名+文件夹名+文件名
    img_list = []
    for image in images:
        img_path = image["path"]

        # img_name = image["file_name"]     # 过滤单反图片或显微镜图片（已舍弃）
        # if not re.match("HXU", img_name):
        #     continue

        img_path = img_path.split("/")[2:5]  # 截取图片相对路径 [类别名,标本号,文件名]

        img_path_str = ""
        for i in range(len(img_path)):
            path_index = img_path[i]

            img_path_str = os.path.join(img_path_str, path_index)  # 最终的图片相对路径 [类别名\标本号\文件名]
        img_list.append(img_path_str)

    for img_path in img_list:
        # 图像的初始存储路径
        img_path = os.path.join(r"D:\标注过的图片已审核", batch_name, img_path)  # 首次
        # img_path = os.path.join(r"D:\标注过的图片已审核", batch_name, "re-annotate-re1", img_path)  # 重标注1
        # img_path = os.path.join(r"D:\标注过的图片已审核", batch_name, "re-annotate-re2", img_path)  # 重标注2
        # img_path = os.path.join(r"D:\标注过的图片已审核", batch_name, "re-annotate-re3", img_path)  # 重标注3
        img_name = img_path.split("\\")[-1]
        target_path = os.path.join(str(target_dir), img_name)
        if os.path.exists(img_path):
            print("正在复制：", img_path)
            shutil.copy(img_path, target_path)
            print("复制成功：", target_path)
        else:
            continue


if __name__ == '__main__':
    p = Pool(10)
    for path in category_list:
        p.apply_async(helper, args=(path,))
    p.close()
    p.join()
