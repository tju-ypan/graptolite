"""
将专家挑选的100张图像作为测试集，并删除这些图像所在标本的所有图像
"""
import os
import json
from pathlib import Path
import shutil
import pandas as pd
from multiprocessing import Pool
import re

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
# category_list = category_list13


def get_json_images(parent_json_folder: str, pure_category_name: str):
    source_json_file = Path(parent_json_folder) / (pure_category_name + '.json')
    if not source_json_file.exists():
        return None
    txt = source_json_file.read_text(encoding='utf-8')  # str
    json_txt = json.loads(txt)  # dict
    return json_txt["images"]


def get_test_images_info():
    # pandas读取文件
    file_path = os.path.abspath(r'C:\Users\admin\Desktop\TEST SET-1225-答案定稿(1).xlsx')
    df = pd.read_excel(file_path, sheet_name=0, header=0, encoding='utf-8')
    test_species_images = {}  # 存储每个属种对应的测试图像
    test_species_instance = {}  # 存储每个属种对应的测试图像的标本号
    curr_species_name = None
    for index, row in df.iterrows():
        species_name = str(row["属种拉丁名"]).replace(u'\xa0', u' ').strip()
        image_name = str(row["原照片号"]).strip()
        instance_name = str(row["标本编号"]).strip()
        if not species_name == 'nan':
            curr_species_name = species_name
            test_species_images[species_name] = []
            test_species_instance[species_name] = []
            test_species_images[species_name].append(image_name)
            if instance_name not in test_species_instance[species_name]:
                test_species_instance[species_name].append(instance_name)
        else:
            test_species_images[curr_species_name].append(image_name)
            if instance_name not in test_species_instance[curr_species_name]:
                test_species_instance[curr_species_name].append(instance_name)
    return test_species_images, test_species_instance


def cross_valid(category_name):
    if re.match(r'^\d{2}', category_name):
        pure_category_name = category_name[2:]
    else:
        pure_category_name = category_name[1:]
    test_species_images, test_species_instance = get_test_images_info()
    # print(test_species_images[pure_category_name])
    # print(test_species_instance[pure_category_name])

    annotated_images_folder = Path(r'D:\set113_final\annotated_images') / category_name
    test_images_dir = os.path.join(r'D:\set113_final\test_images', category_name)
    discarded_images_dir = os.path.join(r'D:\set113_final\discarded_images', category_name)
    if not os.path.exists(test_images_dir):
        os.makedirs(test_images_dir)
    if not os.path.exists(discarded_images_dir):
        os.makedirs(discarded_images_dir)

    # 读取json文件
    json_images_all = []
    parent_json_folder1 = r"D:\set113_ori\original_json"
    json_images1 = get_json_images(parent_json_folder1, pure_category_name)
    if json_images1 is not None:
        json_images_all.extend(json_images1)

    parent_json_folder2 = r"D:\set113_ori\re_annotation1\original_json"
    json_images2 = get_json_images(parent_json_folder2, pure_category_name + '_re1')
    if json_images2 is not None:
        json_images_all.extend(json_images2)

    parent_json_folder3 = r"D:\set113_ori\re_annotation2\original_json"
    json_images3 = get_json_images(parent_json_folder3, pure_category_name + '_re2')
    if json_images3 is not None:
        json_images_all.extend(json_images3)

    parent_json_folder4 = r"D:\set113_ori\re_annotation3\original_json"
    json_images4 = get_json_images(parent_json_folder4, pure_category_name + '_re3')
    if json_images4 is not None:
        json_images_all.extend(json_images4)

    for image in annotated_images_folder.glob("*.jpg"):
        img_name = image.name
        img_name_copy = img_name
        img_path = image.parent / img_name
        if re.search('_copy_', img_name):
            s_index = re.search('_copy_', img_name).span()[0]
            e_index = re.search('_copy_', img_name).span()[1] + 1
            extra_str = img_name[s_index: e_index]
            img_name_copy = img_name.replace(extra_str, '')
        # 第一步，复制测试集图像
        if img_name in test_species_images[pure_category_name]:
            test_image_path = os.path.join(test_images_dir, img_name)
            shutil.copy(img_path, test_image_path)
            print("复制测试图像：{}".format(test_image_path))

        # 第二步，删除测试图像所在标本的所有图像
        for json_image in json_images_all:
            if not img_path.exists():   # 对于一张图像上有多个标注的，遍历第一个标注时图像已被舍弃
                continue
            name_in_json = json_image["file_name"]

            if name_in_json == img_name or name_in_json == img_name_copy:
                json_img_path = json_image["path"]
                img_instance_number = json_img_path.split('/')[-2]

                if img_instance_number in test_species_instance[pure_category_name]:
                    discarded_image_path = os.path.join(discarded_images_dir, img_name)
                    shutil.copy(img_path, discarded_image_path)
                    print("复制舍弃图像：{}".format(discarded_image_path))
                    img_path.unlink()


if __name__ == '__main__':
    p = Pool(10)
    for category_name in category_list:
        p.apply_async(cross_valid, args=(category_name,))
    p.close()
    p.join()
    print("总类别数: ", len(category_list))
