import os
import re
import glob
import pandas as pd

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
category_list_all = [category_list1, category_list2, category_list3, category_list4, category_list5, category_list6,
                     category_list7, category_list8, category_list9, category_list10, category_list11, category_list12,
                     category_list13, category_list14]

save_path = r'./标本号.csv'
batch_path_list = [
    r"D:\标注过的图片已审核\2019.08.23第一批已审核",
    r"D:\标注过的图片已审核\2020.04.16第二批已审核",
    r"D:\标注过的图片已审核\2020.04.24第三批已审核",
    r"D:\标注过的图片已审核\2020.05.15第四批已审核",
    r"D:\标注过的图片已审核\2020.06.05第五批已审核",
    r"D:\标注过的图片已审核\2020.06.12第六批已审核",
    r"D:\标注过的图片已审核\2020.06.18第七批已审核",
    r"D:\标注过的图片已审核\2020.07.13第八批已审核",
    r"D:\标注过的图片已审核\2020.08.17第九批已审核",
    r"D:\标注过的图片已审核\2020.09.27第十批已审核",
    r"D:\标注过的图片已审核\2020.10.12第十一批已审核",
    r"D:\标注过的图片已审核\2020.11.12第十二批已审核",
    r"D:\标注过的图片已审核\2020.11.16第十三批已审核",
    r"D:\标注过的图片已审核\2020.11.20第十四批已审核"
]
class_name_list = []
label_list = []
image_name_list = []


def helper(batch_path, class_name):
    class_path = os.path.join(batch_path, class_name)
    label_path_list = [os.path.join(class_path, label) for label in os.listdir(class_path) if label not in ['.exports', '.thumbnail']]
    for label_path in label_path_list:
        image_list = glob.glob(os.path.join(label_path, "*.jpg"))
        for image_path in image_list:
            image_name = image_path.split("\\")[-1]
            label = image_path.split("\\")[-2]
            class_name_list.append(class_name)
            label_list.append(label)
            image_name_list.append(image_name)


if __name__ == '__main__':
    for batch_path, category_list in zip(batch_path_list, category_list_all):
        for class_name in category_list:
            if re.match(r'^\d{2}', class_name):
                class_name = class_name[2:]
            elif re.match(r'^\d', class_name):
                class_name = class_name[1:]
            helper(batch_path, class_name)
    print(len(class_name_list), len(label_list), len(image_name_list))
    data = {
        'label': label_list,
        'image': image_name_list
    }
    df = pd.DataFrame(data, index=class_name_list, columns=['label', 'image'])
    if os.path.exists(save_path):
        df.to_csv(r'./标本号.csv', mode='a', header=False)
    else:
        df.to_csv(r'./标本号.csv')
