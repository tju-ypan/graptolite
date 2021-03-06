# 计算每个类别对应的门、纲、目、科、属、种的准确率
import os
import re
import numpy as np
import pandas as pd

name_list = ['10Demirastrites triangulatus', '10Dicellograptus tumidus', '10Dicellograptus turgidus',
             '10Paraorthograptus pacificus', '10Spirograptus turriculatus', '11Appendispinograptus venustus',
             '11Nicholsonograptus fasciculatus', '11Nicholsonograptus praelongus', '11Paraorthograptus longispinus',
             '12Cryptograptus tricornis (Juvenile)', '12Phyllograptus anna', '12Rastrites guizhouensis',
             '12Tangyagraptus typicus', '12Yinograptus grandis', '13Coronograptus cyphus', '13Cystograptus vesiculosus',
             '13Normalograptus extraordinarius', '13Normalograptus persculptus', '13Parakidograptus acuminatus',
             '14Diceratograptus mirus', '14Lituigraptus convolutus', '14Paraplegmatograptus connectus',
             '14Pararetiograptus regularis', '1Dicellograptus bispiralis', '1Dicellograptus caduceus',
             '1Dicellograptus divaricatus salopiensis', '1Dicellograptus smithi', '1Dicellograptus undatus',
             '1Dicranograptus irregularis', '1Dicranograptus sinensis', '1Didymograptus jiangxiensis',
             '1Didymograptus latus tholiformis', '1Didymograptus miserabilis', '2Amplexograptus acusiformis',
             '2Amplexograptus fusiformis', '2Cryptograptus arcticus sinensis', '2Cryptograptus gracilicornis',
             '2Dicellograptus divaricatus', '2Dicranograptus nicholsoni parvangulus', '2Dicranograptus ramosus',
             '2Didymograptus euodus', '2Didymograptus linearis longus', '2Didymograptus saerganensis',
             '3Climacograptus pauperatus', '3Cryptograptus arcticus', '3Cryptograptus marcidus',
             '3Cryptograptus tricornis', '3Glossograptus briaros', '3Glossograptus robustus',
             '3Glyptograptus plurithecatus wuningensis', '3Glyptograptus teretiusculus siccatus',
             '3Pseudoclimacograptus parvus jiangxiensis', '3Pseudoclimacograptus wannanensis',
             '4Diplograptus proelongatus', '4Glyptograptus teretiusculus', '4Jiangxigraptus inculus',
             '4Jishougraptus mui', '4Leptograptus flaccidus trentonensis', '4Monoclimacis neimengolensis',
             '4Pseudoclimacograptus angulatus', '4Pseudoclimacograptus longus', '4Pseudoclimacograptus modestus',
             '4Pseudoclimacograptus parvus', '5Amplexograptus disjunctus yangtzensis', '5Amplexograptus suni',
             '5Climacograptus miserabilis', '5Climacograptus supernus', '5Dicellograptus ornatus',
             '5Diplograptus modestus', '5Glyptograptus incertus', '5Petalolithus elongatus', '5Petalolithus folium',
             '5Streptograptus runcinatus', '6Dicellograptus szechuanensis', '6Diplograptus bohemicus',
             '6Glyptograptus austrodentatus', '6Glyptograptus gracilis', '6Glyptograptus lungmacensis',
             '6Glyptograptus tamariscus', '6Glyptograptus tamariscus linealis', '6Glyptograptus tamariscus magnus',
             '6Reteograptus uniformis', '6Retiolites geinitzianus', '7Amplexograptus orientalis',
             '7Climacograptus angustatus', '7Climacograptus leptothecalis', '7Climacograptus minutus',
             '7Climacograptus normalis', '7Climacograptus tianbaensis', '7Colonograptus praedeubeli',
             '7Diplograptus angustidens', '7Diplograptus diminutus', '7Rectograptus pauperatus',
             '8Amplexograptus confertus', '8Climacograptus angustus', '8Climacograptus textilis yichangensis',
             '8Colonograptus deubeli', '8Dicellograptus\xa0cf.\xa0complanatus', '8Diplograptus concinnus',
             '8Pristiograptus variabilis', '8Pseudoclimacograptus demittolabiosus', '8Pseudoclimacograptus formosus',
             '8Rectograptus abbreviatus', '9Akidograptus ascensus', '9Amplexograptus\xa0cf.\xa0maxwelli',
             '9Cardiograptus amplus', '9Climacograptus bellulus', '9Climacograptus hastatus', '9Glyptograptus dentatus',
             '9Glyptograptus elegans', '9Glyptograptus elegantulus', '9Orthograptus calcaratus',
             '9Trigonograptus ensiformis']

# 去掉开头的批次号
for index, name in enumerate(name_list):
    if re.match(r'^\d{2}', name):
        name_list[index] = name[2:]
    elif re.match(r'^\d', name):
        name_list[index] = name[1:]
# 解决编码问题
name_list = [name[:].replace(u'\xa0', u' ') for name in name_list]
name_dict = {name: {"phylum": "", "class": "", "order": "", "family": ""} for name in name_list}

# pandas读取文件
file_path = os.path.abspath(r'D:\TJU\GBDB\set113\标注属种信息汇总_1125.xlsx')
df = pd.read_excel(file_path, sheet_name=0, header=1, encoding='utf-8')

# 遍历dataframe每行
for index, row in df.iterrows():
    english_name = row["english_name"].strip().replace(u'\xa0', u' ')
    pattern = re.compile(r'[a-zA-Z]+', re.S)
    phylum_name = re.search(pattern, row["phylum"].strip()).group()
    class_name = re.search(pattern, row["class"].strip()).group()
    order_name = re.search(pattern, row["order"].strip()).group()
    family_name = re.search(pattern, row["family"].strip()).group()
    # 判断类名列表中是否存在
    assert english_name in name_dict, print(english_name)
    # 更新字典
    name_dict[english_name]["phylum"] = phylum_name
    name_dict[english_name]["class"] = class_name
    name_dict[english_name]["order"] = order_name
    name_dict[english_name]["family"] = family_name

nclasses = 113
test_results = \
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 97, 73, 97, 67, 67, 97, 1, 67, 73, 67, 67, 67, 1, 67, 1, 97, 1, 1, 1, 1, 1, 67, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 52, 2, 2, 2, 19, 19, 28, 28, 1, 24, 2, 2, 2, 2, 28, 2, 28, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 67, 24, 19, 67, 67, 19, 73, 67, 2, 67, 24, 67, 67, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 21, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 19, 6, 6, 6, 6, 6, 7, 6, 6, 6, 6, 6, 7, 6, 7, 6, 6, 7, 7, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 3, 3, 3, 3, 3, 3, 3, 3, 8, 3, 3, 3, 8, 8, 8, 8, 8, 3, 8, 8, 8, 8, 9, 46, 9, 9, 46, 46, 46, 46, 46, 46, 9, 46, 9, 46, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 21, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 72, 73, 14, 14, 73, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 17, 88, 65, 15, 65, 65, 65, 17, 17, 17, 17, 17, 17, 17, 17, 15, 17, 17, 17, 17, 17, 16, 16, 16, 64, 64, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 65, 65, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 99, 99, 72, 63, 73, 90, 72, 90, 90, 65, 90, 90, 90, 90, 63, 90, 63, 90, 90, 63, 72, 63, 56, 63, 63, 18, 63, 63, 63, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 0, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 3, 22, 22, 22, 22, 22, 21, 22, 22, 22, 22, 22, 22, 22, 22, 22, 49, 52, 22, 3, 21, 22, 15, 49, 21, 21, 22, 3, 21, 21, 21, 22, 22, 49, 15, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 30, 24, 24, 24, 24, 24, 24, 24, 24, 24, 56, 25, 24, 25, 24, 24, 24, 97, 33, 24, 27, 24, 73, 24, 24, 73, 29, 25, 28, 28, 28, 28, 28, 28, 25, 28, 28, 29, 25, 25, 29, 29, 29, 24, 25, 25, 25, 24, 24, 24, 24, 25, 25, 25, 24, 25, 25, 25, 29, 25, 25, 29, 29, 29, 27, 55, 27, 27, 55, 27, 27, 27, 27, 55, 55, 28, 28, 28, 27, 29, 29, 29, 28, 28, 29, 28, 28, 28, 28, 29, 28, 28, 35, 28, 29, 28, 28, 28, 28, 28, 28, 28, 35, 29, 28, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 35, 29, 29, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 35, 30, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 30, 30, 30, 30, 30, 30, 30, 32, 32, 32, 32, 32, 30, 30, 32, 30, 30, 30, 32, 32, 32, 30, 32, 32, 30, 99, 32, 30, 32, 32, 32, 30, 32, 32, 42, 32, 32, 32, 42, 42, 43, 65, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 34, 34, 34, 34, 34, 75, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 57, 57, 35, 57, 57, 35, 35, 35, 35, 46, 65, 35, 35, 35, 35, 35, 35, 43, 43, 50, 35, 50, 43, 43, 94, 36, 86, 36, 36, 85, 52, 36, 36, 49, 37, 37, 37, 40, 37, 37, 40, 44, 37, 40, 37, 37, 37, 37, 37, 53, 53, 98, 38, 98, 53, 53, 98, 110, 39, 38, 38, 38, 38, 38, 39, 38, 37, 109, 38, 54, 37, 57, 46, 46, 46, 44, 39, 39, 39, 39, 39, 57, 39, 39, 39, 39, 39, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 42, 40, 40, 40, 40, 39, 1, 39, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 56, 41, 41, 41, 41, 41, 41, 42, 25, 41, 56, 56, 42, 40, 56, 42, 42, 42, 56, 56, 42, 42, 42, 33, 35, 43, 43, 35, 43, 35, 35, 35, 35, 35, 35, 35, 51, 35, 35, 35, 35, 35, 35, 43, 55, 44, 44, 55, 43, 44, 44, 55, 44, 44, 44, 44, 44, 44, 44, 45, 45, 45, 45, 45, 45, 108, 45, 45, 108, 45, 45, 45, 45, 45, 45, 45, 45, 44, 44, 44, 112, 44, 44, 112, 46, 112, 4, 46, 46, 46, 46, 46, 46, 46, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 50, 75, 50, 50, 50, 69, 69, 69, 109, 78, 78, 103, 76, 76, 78, 100, 51, 100, 51, 100, 51, 51, 51, 43, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 52, 52, 52, 52, 52, 52, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 53, 54, 92, 68, 54, 53, 31, 53, 53, 53, 53, 53, 53, 53, 53, 54, 53, 53, 53, 53, 53, 53, 53, 53, 54, 53, 53, 53, 53, 53, 53, 53, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 57, 57, 57, 57, 57, 57, 57, 72, 57, 57, 41, 57, 57, 57, 57, 72, 72, 72, 72, 72, 58, 72, 72, 72, 62, 58, 58, 62, 58, 58, 58, 58, 58, 58, 59, 59, 59, 59, 62, 59, 62, 62, 59, 62, 59, 63, 63, 63, 63, 66, 79, 79, 84, 103, 84, 79, 103, 79, 60, 60, 60, 60, 60, 60, 60, 60, 86, 84, 60, 66, 60, 103, 61, 61, 60, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 62, 62, 62, 62, 59, 62, 62, 62, 61, 62, 62, 64, 62, 93, 93, 62, 87, 64, 64, 63, 64, 64, 64, 63, 64, 63, 64, 64, 64, 63, 59, 63, 64, 63, 63, 63, 63, 63, 63, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 101, 84, 59, 79, 79, 60, 59, 79, 59, 78, 72, 78, 84, 78, 58, 76, 75, 18, 78, 18, 66, 66, 63, 70, 66, 66, 66, 66, 90, 90, 66, 66, 66, 66, 66, 66, 66, 66, 66, 67, 67, 67, 41, 67, 67, 67, 67, 65, 67, 67, 67, 67, 67, 67, 67, 67, 67, 1, 67, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 82, 68, 49, 69, 91, 82, 68, 68, 68, 68, 68, 109, 84, 109, 69, 63, 109, 88, 88, 87, 69, 79, 86, 84, 84, 69, 50, 79, 79, 69, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 71, 15, 71, 71, 71, 71, 71, 71, 71, 71, 71, 15, 71, 15, 15, 71, 71, 15, 71, 15, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 97, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 74, 74, 74, 74, 74, 63, 74, 74, 74, 74, 92, 69, 74, 74, 74, 74, 92, 80, 74, 74, 69, 69, 91, 69, 52, 69, 92, 91, 91, 91, 69, 52, 91, 75, 75, 75, 75, 75, 75, 76, 76, 76, 92, 92, 92, 92, 76, 92, 92, 76, 76, 76, 69, 76, 92, 111, 111, 76, 76, 50, 84, 50, 77, 50, 77, 77, 108, 50, 77, 108, 77, 108, 77, 108, 77, 108, 80, 77, 76, 76, 78, 78, 78, 78, 78, 78, 78, 99, 78, 87, 78, 79, 78, 78, 87, 87, 111, 79, 111, 111, 111, 109, 79, 79, 111, 108, 79, 79, 79, 79, 79, 79, 79, 79, 79, 80, 80, 80, 80, 80, 80, 80, 80, 80, 79, 80, 80, 80, 80, 80, 80, 80, 80, 80, 81, 81, 81, 81, 16, 81, 81, 81, 16, 81, 81, 16, 16, 81, 81, 22, 22, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 83, 83, 86, 83, 83, 83, 86, 83, 83, 83, 16, 91, 91, 91, 91, 91, 91, 91, 86, 16, 86, 86, 16, 16, 88, 88, 65, 88, 88, 109, 109, 87, 88, 86, 86, 87, 87, 88, 88, 87, 109, 87, 90, 87, 87, 87, 70, 85, 85, 85, 63, 85, 85, 27, 85, 63, 85, 85, 106, 106, 85, 85, 85, 106, 85, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 86, 104, 104, 104, 104, 104, 104, 77, 77, 77, 77, 77, 77, 31, 77, 77, 77, 77, 77, 31, 44, 77, 31, 77, 77, 77, 90, 77, 77, 77, 90, 77, 77, 77, 77, 44, 65, 44, 88, 44, 65, 88, 88, 88, 44, 88, 88, 74, 74, 69, 86, 86, 61, 52, 74, 69, 99, 96, 96, 89, 96, 96, 96, 96, 96, 96, 96, 89, 96, 96, 89, 89, 96, 96, 99, 89, 96, 89, 89, 89, 90, 90, 63, 90, 70, 70, 70, 70, 70, 63, 70, 70, 70, 63, 70, 70, 90, 90, 70, 90, 90, 90, 90, 70, 90, 90, 90, 90, 90, 90, 70, 90, 90, 90, 90, 70, 102, 102, 91, 91, 91, 91, 91, 91, 91, 91, 91, 91, 91, 91, 91, 69, 91, 102, 102, 91, 91, 102, 91, 102, 92, 92, 92, 92, 92, 72, 92, 92, 99, 99, 96, 92, 96, 92, 96, 96, 96, 99, 22, 22, 22, 22, 22, 70, 17, 22, 22, 104, 93, 93, 93, 93, 93, 59, 93, 93, 93, 93, 93, 66, 84, 87, 79, 84, 84, 66, 88, 78, 74, 66, 84, 66, 86, 84, 84, 84, 94, 86, 86, 92, 86, 86, 79, 76, 78, 85, 101, 85, 85, 87, 101, 85, 85, 85, 92, 66, 108, 85, 85, 85, 85, 85, 85, 66, 85, 66, 85, 85, 85, 66, 66, 66, 66, 85, 66, 85, 66, 66, 96, 96, 96, 96, 96, 96, 96, 76, 96, 96, 89, 99, 96, 96, 96, 96, 96, 96, 96, 96, 96, 97, 97, 97, 97, 97, 85, 97, 97, 85, 85, 85, 85, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 98, 76, 98, 98, 98, 98, 98, 98, 98, 110, 98, 98, 110, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 99, 99, 99, 99, 99, 99, 96, 99, 99, 66, 99, 99, 99, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 84, 100, 100, 84, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 101, 101, 101, 101, 101, 101, 101, 101, 101, 51, 101, 101, 101, 101, 101, 101, 101, 40, 101, 43, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 76, 101, 101, 101, 101, 101, 101, 101, 101, 102, 102, 102, 102, 92, 92, 92, 102, 102, 102, 91, 102, 91, 70, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 103, 108, 103, 110, 109, 103, 108, 103, 103, 33, 97, 108, 45, 108, 103, 103, 108, 103, 103, 103, 108, 66, 104, 104, 104, 104, 104, 106, 44, 85, 46, 104, 104, 104, 104, 104, 104, 104, 104, 15, 15, 15, 15, 15, 22, 15, 15, 71, 15, 15, 15, 15, 15, 15, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 52, 107, 107, 107, 85, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 95, 91, 102, 102, 111, 108, 102, 75, 108, 108, 108, 108, 108, 108, 65, 108, 108, 93, 93, 93, 93, 93, 93, 45, 93, 108, 93, 108, 59, 59, 63, 63, 103, 59, 109, 63, 109, 63, 63, 70, 63, 63, 103, 91, 103, 91, 103, 91, 110, 103, 110, 110, 110, 110, 110, 110, 110, 110, 110, 101, 110, 108, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 103, 108, 108, 108, 108, 74, 95, 111, 54, 111, 95, 95, 111, 111, 111, 75, 83, 111, 111, 111, 111, 111, 111, 80, 111, 111, 112, 86, 112, 13, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112]
test_labels = \
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 91, 91, 91, 91, 91, 91, 91, 91, 91, 91, 91, 91, 91, 91, 91, 91, 91, 91, 91, 91, 91, 91, 91, 91, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112]
print('种平均准确率为: {:.4f}'.format(np.equal(test_results, test_labels).astype(np.float32).mean()))

species_acc_list = []
genus_acc_list = []
family_acc_list = []
order_acc_list = []
class_acc_list = []
phylum_acc_list = []

for class_id in range(nclasses):
    # 保存当前类别的ground truth label的索引
    gt_index_list = []
    for index, class_number in enumerate(test_labels):
        if class_number == class_id:
            gt_index_list.append(index)

    species_acc_num = 0
    genus_acc_num = 0
    family_acc_num = 0
    order_acc_num = 0
    class_acc_num = 0
    phylum_acc_num = 0

    for gt_index in gt_index_list:
        # 种
        if test_results[gt_index] == class_id:
            species_acc_num += 1
        # 属
        if name_list[test_results[gt_index]].split(' ')[0] == name_list[class_id].split(' ')[0]:
            genus_acc_num += 1
        # 科
        if name_dict[name_list[test_results[gt_index]]]["family"] == name_dict[name_list[class_id]]["family"]:
            family_acc_num += 1
        # 目
        if name_dict[name_list[test_results[gt_index]]]["order"] == name_dict[name_list[class_id]]["order"]:
            order_acc_num += 1
        # 纲
        if name_dict[name_list[test_results[gt_index]]]["class"] == name_dict[name_list[class_id]]["class"]:
            class_acc_num += 1
        # 门
        if name_dict[name_list[test_results[gt_index]]]["phylum"] == name_dict[name_list[class_id]]["phylum"]:
            phylum_acc_num += 1
    # 当前类别的各级别准确率
    species_acc = float(format(species_acc_num / len(gt_index_list), '.2f'))
    genus_acc = float(format(genus_acc_num / len(gt_index_list), '.2f'))
    family_acc = float(format(family_acc_num / len(gt_index_list), '.2f'))
    order_acc = float(format(order_acc_num / len(gt_index_list), '.2f'))
    class_acc = float(format(class_acc_num / len(gt_index_list), '.2f'))
    phylum_acc = float(format(phylum_acc_num / len(gt_index_list), '.2f'))
    # 添加到总体类别中
    species_acc_list.append(species_acc)
    genus_acc_list.append(genus_acc)
    family_acc_list.append(family_acc)
    order_acc_list.append(order_acc)
    class_acc_list.append(class_acc)
    phylum_acc_list.append(phylum_acc)

print('种被识别为正确的种: %.2f' % np.mean(species_acc_list))
print('种被识别为正确的属: %.2f' % np.mean(genus_acc_list))
print('种被识别为正确的科: %.2f' % np.mean(family_acc_list))
print('种被识别为正确的目: %.2f' % np.mean(order_acc_list))
print('种被识别为正确的纲: %.2f' % np.mean(class_acc_list))
print('种被识别为正确的门: %.2f' % np.mean(phylum_acc_list))
for i in genus_acc_list:
    print(i)
