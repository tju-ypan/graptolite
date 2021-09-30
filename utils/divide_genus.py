"""
将按种划分的数据集重新按属进行划分
"""
import os
import re
import copy
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

category_list1 = [  # set100-10
    '1Dicellograptus bispiralis',
    '1Dicellograptus caduceus',
    '1Dicellograptus divaricatus salopiensis',
    '1Dicellograptus smithi',
    '1Dicellograptus undatus',
    '1Dicranograptus irregularis',
    '1Dicranograptus sinensis',
    '1Didymograptus jiangxiensis',
    '1Didymograptus latus tholiformis',
    '1Didymograptus miserabilis'
]
category_list2 = [  # set100-20
    '2Amplexograptus acusiformis',
    '2Amplexograptus fusiformis',
    '2Cryptograptus arcticus sinensis',
    '2Cryptograptus gracilicornis',
    '2Dicellograptus divaricatus',
    '2Dicranograptus nicholsoni parvangulus',
    '2Dicranograptus ramosus',
    '2Didymograptus euodus',
    '2Didymograptus linearis longus',
    '2Didymograptus saerganensis'
]
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
category_list4 = [  # set100-40
    '4Diplograptus proelongatus',
    '4Glyptograptus teretiusculus',
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
    '5Amplexograptus suni',
    '5Climacograptus miserabilis',
    '5Climacograptus supernus',
    '5Dicellograptus ornatus',
    '5Diplograptus modestus',
    '5Glyptograptus incertus',
    '5Petalolithus elongatus',
    '5Petalolithus folium',
    '5Streptograptus runcinatus'
]
category_list6 = [  # set100-60
    '6Dicellograptus szechuanensis',
    '6Diplograptus bohemicus',
    '6Glyptograptus austrodentatus',
    '6Glyptograptus gracilis',
    '6Glyptograptus lungmaensis',
    '6Glyptograptus tamariscus',
    '6Glyptograptus tamariscus linealis',
    '6Glyptograptus tamariscus magnus',
    '6Reteograptus uniformis',
    '6Retiolites geinitzianus'
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
category_list8 = [  # set100-80
    '8Amplexograptus confertus',
    '8Climacograptus angustus',
    '8Climacograptus textilis yichangensis',
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
category_list10 = [  # set100-100
    '10Demirastrites triangulatus',
    '10Dicellograptus tumidus',
    '10Dicellograptus turgidus',
    '10Paraorthograptus pacificus',
    '10Spirograptus turriculatus'
]
category_list11 = [  # set102
    '11Appendispinograptus venustus',
    '11Nicholsonograptus fasciculatus',  # multi-instance
    '11Nicholsonograptus praelongus',  # multi-instance
    '11Paraorthograptus longispinus'
]
category_list12 = [  # set107
    '12Cryptograptus tricornis (Juvenile)',  # multi-instance
    '12Phyllograptus anna',  # multi-instance
    '12Rastrites guizhouensis',
    '12Tangyagraptus typicus',
    '12Yinograptus grandis'
]
category_list13 = [  # set112
    '13Coronograptus cyphus',
    '13Cystograptus vesiculosus',
    '13Normalograptus extraordinarius',
    '13Normalograptus persculptus',
    '13Parakidograptus acuminatus'
]
category_list14 = [  # set112
    '14Diceratograptus mirus',
    '14Lituigraptus convolutus',
    '14Paraplegmatograptus connectus',
    '14Pararetiograptus regularis',
]
multi_category_list = ['11Nicholsonograptus fasciculatus', '11Nicholsonograptus praelongus',
                       '12Cryptograptus tricornis (Juvenile)', '12Phyllograptus anna']
name_list = category_list1 + category_list2 + category_list3 + category_list4 + category_list5 + category_list6 \
                + category_list7 + category_list8 + category_list9 + category_list10 + category_list11 \
                + category_list12 + category_list13 + category_list14
name_list_ori = copy.copy(name_list)
# 去掉开头的批次号
for index, name in enumerate(name_list):
    if re.match(r'^\d{2}', name):
        name_list[index] = name[2:]
    elif re.match(r'^\d', name):
        name_list[index] = name[1:]
# 解决编码问题
name_list = [name[:].replace(u'\xa0', u' ') for name in name_list]
# pandas读取文件
# file_path = os.path.abspath(r'.\标注属种信息汇总_1125.xlsx')
# df = pd.read_excel(file_path, sheet_name=0, header=1, encoding='utf-8')
# name_dict = {name: {"phylum": "", "class": "", "order": "", "family": ""} for name in name_list}
# # 遍历dataframe每行
# for index, row in df.iterrows():
#     english_name = row["english_name"].strip().replace(u'\xa0', u' ')
#     pattern = re.compile(r'[a-zA-Z]+', re.S)
#     phylum_name = re.search(pattern, row["phylum"].strip()).group()
#     class_name = re.search(pattern, row["class"].strip()).group()
#     order_name = re.search(pattern, row["order"].strip()).group()
#     family_name = re.search(pattern, row["family"].strip()).group()
#     # 判断类名列表中是否存在
#     assert english_name in name_dict, print('种名列表中不存在: ', english_name)
#     # 更新字典，记录每个属种的门、纲、目、科名
#     name_dict[english_name]["phylum"] = phylum_name
#     name_dict[english_name]["class"] = class_name
#     name_dict[english_name]["order"] = order_name
#     name_dict[english_name]["family"] = family_name

genus_name_list = []
for name in name_list:
    if not name.split(' ')[0] in genus_name_list:
        genus_name_list.append(name.split(' ')[0])

test_index = 'test4'
temp_list = ["train_images", "test_images"]
save_dir = r'D:\set113\all_images_genus'

for genus_name in genus_name_list:
    # 训练集保存路径
    target_train_images_folder = os.path.join(save_dir, test_index, "train_images", genus_name)
    if not os.path.exists(target_train_images_folder):
        os.makedirs(target_train_images_folder)
    # 测试集保存路径
    target_test_images_folder = os.path.join(save_dir, test_index, "test_images", genus_name)
    if not os.path.exists(target_test_images_folder):
        os.makedirs(target_test_images_folder)

for species_name in name_list_ori:
    species_name_clean = species_name
    if re.match(r'^\d{2}', species_name):
        species_name_clean = species_name[2:]
    elif re.match(r'^\d', species_name):
        species_name_clean = species_name[1:]
    genus_name = species_name_clean.split(' ')[0]
    for t in temp_list:
        ori_dir = os.path.join(r'D:\set113\all_images', test_index, t, species_name)
        target_dir = os.path.join(save_dir, test_index, t, genus_name)
        for image_name in os.listdir(ori_dir):
            image_path = os.path.join(ori_dir, image_name)
            target_image_name = species_name.split(' ')[0] + '_' + species_name.split(' ')[1] + image_name
            target_img_path = os.path.join(target_dir, target_image_name)
            shutil.copy(image_path, target_img_path)
            print("复制成功：", target_img_path)
