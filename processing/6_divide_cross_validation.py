"""
交叉验证划分，无需分批次，仅需修改以下参数：
selected_number，max_threshold，min_threshold
"""
import os
import re
import json
import shutil
from multiprocessing import Pool
from pathlib import Path

selected_number = 1  # 分组号
max_threshold = 80  # 测试图像最大数
min_threshold = 15  # 测试图像最小数

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
category_list = category_list1 + category_list2 + category_list3 + category_list4 + category_list5 + category_list6 \
                + category_list7 + category_list8 + category_list9 + category_list10 + category_list11 \
                + category_list12 + category_list13 + category_list14


def get_json_images(parent_json_folder: str, pure_category_name: str):
    source_json_file = Path(parent_json_folder) / (pure_category_name + '.json')
    if not source_json_file.exists():
        return None
    txt = source_json_file.read_text(encoding='utf-8')  # str
    json_txt = json.loads(txt)  # dict
    return json_txt["images"]


# 根据笔石图像的样本号划分训练集和测试集，进行交叉验证
def cross_valid(category_name):
    global max_threshold
    global min_threshold
    if re.match(r'^\d{2}', category_name):
        pure_category_name = category_name[2:]
    else:
        pure_category_name = category_name[1:]

    save_dir = r'D:\set113\cross_validation'
    annotated_images_folder = Path(r'D:\set113_ori\annotated_images_cleaning\annotated_images') / category_name
    dirty_images_folder = Path(r'D:\set113_ori\annotated_images_cleaning\dirty_images') / category_name
    discarded_images_folder = Path(r'D:\set113_final\discarded_images') / category_name
    species_images_sum = len(os.listdir(str(annotated_images_folder)))

    # 训练集保存路径
    target_train_images_folder = Path(save_dir) / ("test" + str(selected_number)) / "train_images" / category_name
    if not target_train_images_folder.exists():
        target_train_images_folder.mkdir(exist_ok=True, parents=True)
    # 测试集保存路径
    target_test_images_folder = Path(save_dir) / ("test" + str(selected_number)) / "test_images" / category_name
    if not target_test_images_folder.exists():
        target_test_images_folder.mkdir(exist_ok=True, parents=True)

    # 1 筛选出该类别中图像数符合阈值范围的标本号并生成列表
    # 1.1 读取并整合所有json文件的标注信息
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
    instance_list_temp = []  # 保存当前属种的所有图像对应的标本号

    # 1.2 获取当前类别图像数符合阈值区间的所有样本的样本号
    for image in annotated_images_folder.glob("*.jpg"):
        img_name = image.name
        img_name_copy = img_name
        if re.search('_copy_', img_name):  # 对于一张图像上的多个标注，还原图像名
            s_index = re.search('_copy_', img_name).span()[0]
            e_index = re.search('_copy_', img_name).span()[1] + 1
            extra_str = img_name[s_index: e_index]
            img_name_copy = img_name.replace(extra_str, '')
        for json_image in json_images_all:
            name_in_json = json_image["file_name"]
            img_path = json_image["path"]
            if len(img_path.split('/')) != 5:  # 过滤掉没有标本号的标注图像
                continue
            if name_in_json == img_name or name_in_json == img_name_copy:
                json_img_path = json_image["path"]
                img_instance_number = json_img_path.split('/')[-2]
                instance_list_temp.append(img_instance_number)
                break  # 对于一张图像，只记录一次它的样本号

    # 1.3 根据阈值范围进行筛选
    instance_dict = {}  # 保存每个标本对应的图像数
    for key in instance_list_temp:
        instance_dict[key] = instance_dict.get(key, 0) + 1
    # min_threshold = species_images_sum * 0.2
    # max_threshold = species_images_sum * 0.4
    if category_name in multi_category_list:
        min_threshold = 1
    instance_list = [instance for instance in instance_dict if
                     min_threshold <= instance_dict[instance] <= max_threshold]
    length = len(instance_list)
    print('类别名:{}:'.format(category_name))
    print('筛选前:', instance_dict)
    print('筛选后:', len(instance_list), instance_list)

    # 2 在标本本号列表中选择第selected_number个样本号作为测试集标本。其余样本号作为训练集标本
    # 遍历裁剪后图像的类别文件夹，对于每一张图像，遍历json文件找到它的标本号，然后进行划分。
    for image in annotated_images_folder.glob("*.jpg"):
        is_use = False  # 记录第一批原始图像是否被分组
        img_name = image.name
        img_name_copy = img_name
        if re.search('_copy_', img_name):  # 对于一张图像上的多个标注，还原图像名
            s_index = re.search('_copy_', img_name).span()[0]
            e_index = re.search('_copy_', img_name).span()[1] + 1
            extra_str = img_name[s_index: e_index]
            img_name_copy = img_name.replace(extra_str, '')
        img_path = image.parent / img_name

        for json_image in json_images_all:
            name_in_json = json_image["file_name"]
            if len(json_image["path"].split('/')) != 5:  # 过滤掉没有标本号的标注图像
                continue
            if name_in_json == img_name or name_in_json == img_name_copy:
                is_use = True
                json_img_path = json_image["path"]
                img_instance_number = json_img_path.split('/')[-2]
                # train_images文件夹对应关键字'not'，即划分第selected_number个样本的所有图像为测试集，其余为训练集
                if category_name not in multi_category_list:
                    if not (img_instance_number == instance_list[(selected_number - 1) % length]):
                        target_images_path = target_train_images_folder / img_name
                        print('train images: ', target_images_path)
                        shutil.copy(img_path, target_images_path)
                    else:
                        target_images_path = target_test_images_folder / img_name
                        print('test images: ', target_images_path)
                        shutil.copy(img_path, target_images_path)
                    break  # 一张图像无论有多少个标注，都仅匹配一次标本号
                else:  # 特殊类别，马譞论文中的图像
                    if not (img_instance_number == instance_list[(selected_number - 1) % length]
                            or img_instance_number == instance_list[(selected_number + 0) % length]
                            or img_instance_number == instance_list[(selected_number + 1) % length]
                            or img_instance_number == instance_list[(selected_number + 2) % length]
                            or img_instance_number == instance_list[(selected_number + 3) % length]
                            or img_instance_number == instance_list[(selected_number + 4) % length]
                            or img_instance_number == instance_list[(selected_number + 5) % length]
                            or img_instance_number == instance_list[(selected_number + 6) % length]
                            or img_instance_number == instance_list[(selected_number + 7) % length]
                            or img_instance_number == instance_list[(selected_number + 8) % length]
                            or img_instance_number == instance_list[(selected_number + 9) % length]
                            or img_instance_number == instance_list[(selected_number + 10) % length]
                            or img_instance_number == instance_list[(selected_number + 11) % length]
                    ):
                        target_images_path = target_train_images_folder / img_name
                        print('train images: ', target_images_path)
                        shutil.copy(img_path, target_images_path)
                    else:
                        target_images_path = target_test_images_folder / img_name
                        print('test images: ', target_images_path)
                        shutil.copy(img_path, target_images_path)
                    break  # 一张图像无论有多少个标注，都仅匹配一次标本号
            else:
                continue
        if not is_use:
            target_images_path = target_train_images_folder / img_name
            print('train images: ', target_images_path)
            shutil.copy(img_path, target_images_path)


if __name__ == '__main__':
    p = Pool(10)
    for category_name in category_list:
        p.apply_async(cross_valid, args=(category_name,))
    p.close()
    p.join()
    # for name in category_list:
    #     cross_valid(name)
    print("总类别数: ", len(category_list))
