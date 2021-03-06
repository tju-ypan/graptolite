import os
import re
import uuid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, fbeta_score

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
             '6Glyptograptus austrodentatus', '6Glyptograptus gracilis', '6Glyptograptus lungmaensis',
             '6Glyptograptus tamariscus', '6Glyptograptus tamariscus linealis', '6Glyptograptus tamariscus magnus',
             '6Reteograptus uniformis', '6Retiolites geinitzianus', '7Amplexograptus orientalis',
             '7Climacograptus angustatus', '7Climacograptus leptothecalis', '7Climacograptus minutus',
             '7Climacograptus normalis', '7Climacograptus tianbaensis', '7Colonograptus praedeubeli',
             '7Diplograptus angustidens', '7Diplograptus diminutus', '7Rectograptus pauperatus',
             '8Amplexograptus confertus', '8Climacograptus angustus', '8Climacograptus textilis yichangensis',
             '8Colonograptus deubeli', '8Dicellograptus cf. complanatus', '8Diplograptus concinnus',
             '8Pristiograptus variabilis', '8Pseudoclimacograptus demittolabiosus', '8Pseudoclimacograptus formosus',
             '8Rectograptus abbreviatus', '9Akidograptus ascensus', '9Amplexograptus cf. maxwelli',
             '9Cardiograptus amplus', '9Climacograptus bellulus', '9Climacograptus hastatus', '9Glyptograptus dentatus',
             '9Glyptograptus elegans', '9Glyptograptus elegantulus', '9Orthograptus calcaratus',
             '9Trigonograptus ensiformis']
# ????????????????????????
for index, name in enumerate(name_list):
    if re.match(r'^\d{2}', name):
        name_list[index] = name[2:]
    elif re.match(r'^\d', name):
        name_list[index] = name[1:]
# ??????????????????
name_list = [name[:].replace(u'\xa0', u' ') for name in name_list]
# pandas????????????
# file_path = os.path.abspath(r'C:\Users\admin\Desktop\???????????????-?????????.xlsx')
# file_path = os.path.abspath(r'C:\Users\admin\Desktop\????????? for fig3.xlsx')
file_path = os.path.abspath(r'C:\Users\admin\Desktop\expert tables for fig3.xlsx')
df = pd.read_excel(file_path, sheet_name=0, header=0)[:100]
valid_score = [0, 1]
label_list = []
after_to_before_dict = {}


def calculate_points(people_name):
    print(people_name)
    species_name_before_list = []   # ?????????????????????
    species_name_after_list = []    # ?????????????????????
    genus_name_before_list = []     # ?????????????????????
    genus_name_after_list = []      # ?????????????????????
    category_name_before_list = []  # ????????????????????????
    category_name_after_list = []  # ????????????????????????

    predict_species_name_list = []  # ????????????????????????
    predict_genus_name_list = []  # ????????????????????????
    predict_category_name_list = []    # ???????????????????????????

    # ?????????????????????????????????????????????
    for index, row in df.iterrows():
        pattern = re.compile(r'[a-zA-Z]*.*', re.S)
        genus_name_before = re.search(pattern, str(row["??????"]).strip()).group().replace(u'\xa0', u' ')
        genus_name_after = re.search(pattern, str(row["???????????????"]).strip()).group().replace(u'\xa0', u' ')
        species_name_before = re.search(pattern, str(row["??????"]).strip()).group().replace(u'\xa0', u' ')
        species_name_after = re.search(pattern, str(row["???????????????"]).strip()).group().replace(u'\xa0', u' ')

        species_score = row[people_name + "??????????????????"]
        predict_species_name = str(row[people_name + "????????????"]).strip().replace(u'\xa0', u' ')
        predict_genus_name = str(row[people_name + "????????????"]).strip().replace(u'\xa0', u' ')

        # ?????????????????????????????????????????????????????????????????????
        # if predict_species_name == "nan":
        #     predict_species_name = str(uuid.uuid1())
        # if predict_genus_name == "nan":
        #     predict_genus_name = str(uuid.uuid1())

        if species_score not in valid_score:
            species_score = 0.0
        if species_score in valid_score:
            species_name_before_list.append(species_name_before)
            species_name_after_list.append(species_name_after)
            genus_name_before_list.append(genus_name_before)
            genus_name_after_list.append(genus_name_after)
            predict_species_name_list.append(predict_species_name)
            predict_genus_name_list.append(predict_genus_name)

            category_name_before = genus_name_before + ' ' + species_name_before
            category_name_before_list.append(category_name_before)
            category_name_after = genus_name_after + ' ' + species_name_after
            category_name_after_list.append(category_name_after)
            category_name_predict = predict_genus_name + ' ' + predict_species_name
            predict_category_name_list.append(category_name_predict)

            after_to_before_dict[category_name_after] = category_name_before

    # ??????????????????????????????set113+??????????????????????????????
    expert_category_scope_all = name_list + predict_category_name_list
    expert_category_scope = []
    for name in expert_category_scope_all:
        if name not in expert_category_scope:
            expert_category_scope.append(name)
    # ?????????????????????????????????????????????????????????????????????????????????
    matrix_length = len(expert_category_scope)
    expert_confuse_matrix = np.zeros((matrix_length, matrix_length), dtype=np.int)
    print("?????????????????????{}".format(matrix_length), expert_category_scope)

    for index, row in df.iterrows():
        pattern = re.compile('[a-zA-Z]*.*', re.S)
        genus_name_before = re.search(pattern, str(row["??????"]).strip()).group().replace(u'\xa0', u' ')
        genus_name_after = re.search(pattern, str(row["???????????????"]).strip()).group().replace(u'\xa0', u' ')
        species_name_before = re.search(pattern, str(row["??????"]).strip()).group().replace(u'\xa0', u' ')
        species_name_after = re.search(pattern, str(row["???????????????"]).strip()).group().replace(u'\xa0', u' ')
        category_name_before = genus_name_before + ' ' + species_name_before    # gt label1
        category_name_after = genus_name_after + ' ' + species_name_after   # gt label2

        predict_genus_name = re.search(pattern, str(row[people_name + "????????????"]).strip()).group().replace(u'\xa0', u' ')
        predict_species_name = re.search(pattern, str(row[people_name + "????????????"]).strip()).group().replace(u'\xa0', u' ')

        species_score = row[people_name + "??????????????????"]
        if species_score not in valid_score:
            species_score = 0.0
        if species_score in valid_score:
            predict_category_name = predict_genus_name + ' ' + predict_species_name
            # if predict_category_name == "nan nan":
            #     continue
            if predict_category_name in category_name_after_list:
                x_coord = expert_category_scope.index(after_to_before_dict[predict_category_name])
            else:
                x_coord = expert_category_scope.index(predict_category_name)
            y_coord = expert_category_scope.index(category_name_before)
            expert_confuse_matrix[x_coord][y_coord] += 1

    # ????????????????????????????????????
    TPRs_macro, FPRs_macro = [], []
    precision_macro, recall_macro = [], []
    TPs, FPs, TNs, FNs = 0, 0, 0, 0
    for num in range(matrix_length):
        TPR, FPR = 0, 0
        TP, FP, TN, FN = 0, 0, 0, 0
        TP += expert_confuse_matrix[num, num]
        FP += (expert_confuse_matrix[num].sum() - expert_confuse_matrix[num, num])
        for i in range(matrix_length):
            for j in range(matrix_length):
                if i != num and j != num:
                    TN += expert_confuse_matrix[i, j]
        for i in range(matrix_length):
            for j in range(matrix_length):
                if i != num and j == num:
                    FN += expert_confuse_matrix[i, j]
        # for macro
        TPR += TP / (TP + FN)
        FPR += FP / (FP + TN)
        if not np.isnan(TPR):
            TPRs_macro.append(TPR)
        if not np.isnan(FPR):
            FPRs_macro.append(FPR)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        if not np.isnan(precision):
            precision_macro.append(precision)
        if not np.isnan(recall):
            recall_macro.append(recall)

        # for micro
        TPs += TP
        FPs += FP
        TNs += TN
        FNs += FN
    TPRs_micro = TPs / (TPs + FNs)
    FPRs_micro = FPs / (FPs + TNs)

    precision_micro = TPs / (TPs + FPs)
    recall_micro = TPs / (TPs + FNs)

    f1_micro = (2 * precision_micro * recall_micro) / (precision_micro + recall_micro)
    f1_macro = (2 * np.mean(precision_macro) * np.mean(recall_macro)) / \
               (np.mean(precision_macro) + np.mean(recall_macro))

    print("precision_micro:", precision_micro)
    print("recall_micro:", recall_micro)
    print("f1_micro:", f1_micro)
    print("precision_macro:", np.mean(precision_macro))
    print("recall_macro:", np.mean(recall_macro))
    print("f1_macro:", f1_macro)

    # ??????sklearn?????????????????????
    # y_true = []
    # y_pred = []
    # for j in range(expert_confuse_matrix.shape[1]):
    #     column = expert_confuse_matrix[:, j]
    #     for i in range(len(column)):
    #         value = column[i]
    #         for _ in range(value):
    #             y_true.append(j)
    #             y_pred.append(i)
    # precision_macro = precision_score(y_true, y_pred, average='micro')
    # recall_macro = recall_score(y_true, y_pred, average='micro')
    # print("precision_macro:", precision_macro)
    # print("recall_macro:", recall_macro)

    return TPRs_micro, FPRs_micro, precision_micro, recall_micro


if __name__ == '__main__':
    x_points = []
    y_points = []
    pr_x_points = []
    pr_y_points = []
    names = ["??????", "?????????", "?????????", "?????????", "??????"]
    names = ["expert1", "expert2", "expert3", "expert4", "expert5", "expert6", "expert7", "expert8", "expert9",
             "expert10", "expert11", "expert12", "expert13", "expert14", "expert15", "expert16", "expert17", "expert18",
             "expert19", "expert20", "expert21"]
    for name in names:
        TPRs_all, FPRs_all, precision_micro, recall_micro = calculate_points(name)
        x_points.append(np.mean(FPRs_all))
        y_points.append(np.mean(TPRs_all))
        pr_x_points.append(recall_micro)
        pr_y_points.append(precision_micro)
    # for i in range(len(names)):
    #     print(names[i], (pr_x_points[i], pr_y_points[i]))
    print("roc_x_points:", x_points)
    print("roc_y_points:", y_points)
    print("pr_x_points:", pr_x_points)
    print("pr_y_points:", pr_y_points)
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # ??????????????????
    # plt.rcParams['axes.unicode_minus'] = False  # ???????????????????????????
    # plt.plot(x_points, y_points, 'o', color='red')
    # for i in range(len(y_points)):
    #     plt.text(x_points[i], y_points[i], names[i])
    # plt.show()


def check_spell():
    for index, row in df.iterrows():
        pattern = re.compile(r'[a-zA-Z]*.*', re.S)

        genus_name1 = re.search(pattern, str(row["??????"]).strip()).group()
        genus_name2 = re.search(pattern, str(row["???????????????"]).strip()).group()
        species_name1 = re.search(pattern, str(row["??????"]).strip()).group()
        species_name2 = re.search(pattern, str(row["???????????????"]).strip()).group()

        species_score = row["???????????????????????????"]
        genus_score = row["???????????????????????????"]
        predict_genus_name = re.search(pattern, str(row["?????????????????????"]).strip()).group()
        predict_species_name = re.search(pattern, str(row["?????????????????????"]).strip()).group()

        category_name1 = genus_name1 + " " + species_name1
        category_name2 = genus_name2 + " " + species_name2
        category_name_pred = predict_genus_name + " " + predict_species_name

        if species_score in valid_score:
            if species_score == 1 and (category_name_pred not in [category_name1, category_name2]):
            # if genus_score == 1 and (predict_genus_name not in [genus_name1, genus_name2]):
                print(index + 2)
# check_spell()