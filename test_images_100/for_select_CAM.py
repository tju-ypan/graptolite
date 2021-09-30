"""
整合100张测试图像的原始图像及裁剪图像
"""
import os
import sys
import glob
import shutil

from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

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

test100_crop_resize_dir = r"D:\set113_final\test_images"
test100_crop_dir = r"D:\set113_ori\annotated_images_backup"
test100_ori_dir = r"D:\set113_ori\images"
test100_ori_dir1 = r"D:\set113_ori\re_annotation1\images"
test100_ori_dir2 = r"D:\set113_ori\re_annotation2\images"
test100_ori_dir3 = r"D:\set113_ori\re_annotation3\images"
target_dir = r"D:\set113_forCAM"

crop_img_list = []
ori_img_list = []


def select_images(category_name):
    global crop_img_list, ori_img_list
    crop_resize_category_path = os.path.join(test100_crop_resize_dir, category_name)
    for crop_resize_category_img in glob.glob(os.path.join(crop_resize_category_path, "*.jpg")):
        crop_img_path = crop_resize_category_img.replace(test100_crop_resize_dir, test100_crop_dir)
        crop_img_list.append(crop_img_path)

        ori_img_path = crop_resize_category_img.replace(test100_crop_resize_dir, test100_ori_dir)
        ori_img_path1 = crop_resize_category_img.replace(test100_crop_resize_dir, test100_ori_dir1)
        ori_img_path2 = crop_resize_category_img.replace(test100_crop_resize_dir, test100_ori_dir2)
        ori_img_path3 = crop_resize_category_img.replace(test100_crop_resize_dir, test100_ori_dir3)
        if os.path.exists(ori_img_path):
            ori_img_list.append(ori_img_path)
        elif os.path.exists(ori_img_path1):
            ori_img_list.append(ori_img_path1)
        elif os.path.exists(ori_img_path2):
            ori_img_list.append(ori_img_path2)
        elif os.path.exists(ori_img_path3):
            ori_img_list.append(ori_img_path3)
        else:
            sys.exit(0)

        # ori_img_list.append(ori_img_path3)


def copy_images():
    global crop_img_list, ori_img_list
    for i, (crop_img_path, ori_img_path) in enumerate(zip(crop_img_list, ori_img_list)):
        assert crop_img_path.split("\\")[-1] == ori_img_path.split("\\")[-1]
        save_dir = os.path.join(target_dir, str(i + 1))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        ori_img_name = crop_img_path.split("\\")[-1]
        crop_img_name = "_crop.".join(ori_img_name.split("."))
        ori_save_path = os.path.join(save_dir, ori_img_name)
        crop_save_path = os.path.join(save_dir, crop_img_name)

        shutil.copy(ori_img_path, ori_save_path)
        shutil.copy(crop_img_path, crop_save_path)
        print(i+1, ori_save_path, crop_save_path)


if __name__ == '__main__':
    for category_name in category_list:
        select_images(category_name)

    # print(len(crop_img_list), len(ori_img_list))
    # for n1, n2 in zip(crop_img_list, ori_img_list):
    #     print(n1.split("\\")[-1], n2.split("\\")[-1])
    #     if n1.split("\\")[-1] != n2.split("\\")[-1]:
    #         print("fuck")
    #         os._exit(0)
    copy_images()
