"""
根据挑选后的图像，找出其对应的裁剪图
"""
import os
import re
import glob
import shutil
import cv2 as cv
import numpy as np
from concurrent.futures import ThreadPoolExecutor

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
category_list = category_list1 + category_list2 + category_list3 + category_list4 + category_list5 + category_list6 \
                + category_list7 + category_list8 + category_list9 + category_list10 + category_list11 \
                + category_list12 + category_list13 + category_list14
multi_category_list = ['11Nicholsonograptus fasciculatus', '11Nicholsonograptus praelongus',
                       '12Cryptograptus tricornis (Juvenile)', '12Phyllograptus anna']

DATASET_PATH = r"D:\ESSD2021"
TARGET_PATH = r"D:\ESSD2021_crop"
MATCH_PATH = r"D:\标注过的图片已审核"
CROPED_PATH = r"D:\set113_ori\annotated_images_backup"
match_dir_list = [path for path in os.listdir(MATCH_PATH) if os.path.isdir(os.path.join(MATCH_PATH, path))]
match_dir_dict = {str(index+1): path for index, path in enumerate(match_dir_list)}


def match_by_category(category_name):
    if re.search(r"^\d{2}", category_name):
        pure_category = category_name[2:]

        index = category_name[:2]
    else:
        pure_category = category_name[1:]
        index = category_name[:1]
    pure_category_re1 = pure_category + "_re1"
    pure_category_re2 = pure_category + "_re2"
    pure_category_re3 = pure_category + "_re3"

    category_path = os.path.join(DATASET_PATH, category_name)
    target_category_path = os.path.join(TARGET_PATH, category_name)
    if not os.path.exists(target_category_path):
        os.mkdir(target_category_path)
    # 遍历每一张待匹配图像
    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        curr_img = cv.imread(img_path)
        img_instance = img_name.split(".")[0]
        match_path = os.path.join(MATCH_PATH, match_dir_dict[index], pure_category, img_instance)
        # match_path = os.path.join(MATCH_PATH, match_dir_dict[index], "re-annotate-re1", pure_category_re1, img_instance)
        # match_path = os.path.join(MATCH_PATH, match_dir_dict[index], "re-annotate-re2", pure_category_re2, img_instance)
        # match_path = os.path.join(MATCH_PATH, match_dir_dict[index], "re-annotate-re3", pure_category_re3, img_instance)
        match_list = glob.glob(os.path.join(match_path, "*.jpg"))
        if not os.path.exists(match_path):
            continue
        # 对于每一张待匹配图像，遍历原标本号文件夹一一匹配
        for match_img_path in match_list:
            match_img_name = match_img_path.split("\\")[-1]
            match_img = cv.imdecode(np.fromfile(match_img_path, dtype=np.uint8), -1)
            if curr_img.shape[0] == match_img.shape[1] and curr_img.shape[1] == match_img.shape[0]:
                match_img = match_img.reshape(match_img.shape[1], match_img.shape[0], -1)
            if curr_img.shape == match_img.shape:
                # if np.sum(cv.subtract(curr_img, match_img)) == 0:
                if curr_img.sum() == match_img.sum():
                    print("匹配成功: {}, {}".format(img_name, match_img_path.split("\\")[-1]))
                    croped_img_path = os.path.join(CROPED_PATH, category_name, match_img_name)
                    if not os.path.exists(croped_img_path):     # 裁剪图像不存在
                        continue
                    target_croped_img_name = os.path.join(target_category_path, img_name)
                    if os.path.exists(target_croped_img_name):
                        continue
                    shutil.copy(croped_img_path, target_croped_img_name)
                    print(croped_img_path, target_croped_img_name)
                    break


if __name__ == '__main__':
    # name = "9Glyptograptus elegantulus"
    # print(name)
    # match_by_category(name)

    with ThreadPoolExecutor(10) as executor:
        results = executor.map(match_by_category, multi_category_list)
