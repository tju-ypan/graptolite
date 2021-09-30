"""
根据图像人工清理的结果，对数据集进行筛选
"""
from pathlib import Path
from multiprocessing import Pool
import re
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

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
category_list = category_list1 + category_list2 + category_list3 + category_list4 + category_list5 + category_list6 \
                + category_list7 + category_list8 + category_list9 + category_list10 + category_list11 \
                + category_list12 + category_list13 + category_list14

wait_clean_dir = Path(r'D:\set113_ori\annotated_images_cleaning\annotated_images')
dirty_image_dir = Path(r'D:\set113_ori\annotated_images_cleaning\dirty_images')


def cleaning(category_name):
    dirty_dir = dirty_image_dir / category_name
    if not os.path.exists(str(dirty_dir)):
        os.makedirs(str(dirty_dir))

    wait_clean_path = wait_clean_dir / category_name

    for image_path in wait_clean_path.glob('*.jpg'):
        image_name = image_path.name
        image_path = str(image_path)
        check_path = image_path.replace('annotated_images_cleaning', 'annotated_images_template')
        check_path = check_path.replace('annotated_images\\', 'dirty_images\\')
        # if re.search('_copy_', check_path):
        #     check_path = check_path.replace('_copy_', '_')

        if os.path.exists(check_path):
            dirty_path = dirty_image_dir / category_name / image_name

            try:
                shutil.copy(image_path, dirty_path)
                os.remove(image_path)
                print("图像:{}清理至:{}".format(image_path, dirty_path))
            except Exception as e:
                print(str(e))


if __name__ == '__main__':
    # p = Pool(10)
    # for path in category_list:
    #     p.apply_async(cleaning, args=(path,))
    # p.close()
    # p.join()

    with ThreadPoolExecutor(10) as executor:
        results = executor.map(cleaning, category_list)
