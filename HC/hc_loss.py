import os
import re

import torch
import pandas as pd


def get_images_info():
    name_list = ['10Demirastrites triangulatus', '10Dicellograptus tumidus', '10Dicellograptus turgidus',
                 '10Paraorthograptus pacificus', '10Spirograptus turriculatus', '11Appendispinograptus venustus',
                 '11Nicholsonograptus fasciculatus', '11Nicholsonograptus praelongus', '11Paraorthograptus longispinus',
                 '12Cryptograptus tricornis (Juvenile)', '12Phyllograptus anna', '12Rastrites guizhouensis',
                 '12Tangyagraptus typicus', '12Yinograptus grandis', '13Coronograptus cyphus',
                 '13Cystograptus vesiculosus',
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
                 '8Pristiograptus variabilis', '8Pseudoclimacograptus demittolabiosus',
                 '8Pseudoclimacograptus formosus',
                 '8Rectograptus abbreviatus', '9Akidograptus ascensus', '9Amplexograptus\xa0cf.\xa0maxwelli',
                 '9Cardiograptus amplus', '9Climacograptus bellulus', '9Climacograptus hastatus',
                 '9Glyptograptus dentatus',
                 '9Glyptograptus elegans', '9Glyptograptus elegantulus', '9Orthograptus calcaratus',
                 '9Trigonograptus ensiformis']
    # ???????????????
    for index, name in enumerate(name_list):
        if re.match(r'^\d{2}', name):
            name_list[index] = name[2:]
        elif re.match(r'^\d', name):
            name_list[index] = name[1:]
    # ??????????????????
    name_list = [name[:].replace(u'\xa0', u' ') for name in name_list]
    # pandas????????????
    file_path = os.path.abspath('/data/userdata/pyh_projects/????????????????????????_1125.xlsx')
    df = pd.read_excel(file_path, sheet_name=0, header=1, encoding='utf-8')
    name_dict = {name: {"family": ""} for name in name_list}
    # ??????dataframe??????
    for index, row in df.iterrows():
        english_name = row["english_name"].strip().replace(u'\xa0', u' ')
        pattern = re.compile(r'[a-zA-Z]+', re.S)
        family_name = re.search(pattern, row["family"].strip()).group()
        # ?????????????????????????????????
        assert english_name in name_dict, print('????????????????????????: ', english_name)
        # ????????????????????????????????????????????????????????????
        name_dict[english_name]["family"] = family_name
    return name_list, name_dict


def graph_rise_loss(logits, targets, name_list, name_dict):
    batch_size = logits.size(0)
    if float(batch_size) % 2 != 0:
        raise Exception('Incorrect batch_size provided')
    # ???????????????????????????
    batch_left = logits[:int(0.5 * batch_size)]
    batch_right = logits[int(0.5 * batch_size):]
    target_left = targets[:int(0.5 * batch_size)]
    target_right = targets[int(0.5 * batch_size):]

    # ???????????????
    # cosine_similarity = torch.cosine_similarity(batch_left, batch_right, 1)
    # loss = 1 - cosine_similarity    # (0.5 * batch_size,)
    # l2????????????loss
    loss = torch.norm((batch_left - batch_right).abs(), 2, 1)

    # ??????????????????????????????????????????????????????0
    target_mask_species = ~ torch.eq(target_left, target_right)
    target_mask_species = target_mask_species.type(torch.cuda.FloatTensor)
    valid_number = target_mask_species.sum()  # ????????????????????????

    target_mask_species = torch.tensor([1. for _ in range(int(0.5 * batch_size))]).type(torch.cuda.FloatTensor)
    target_mask_genus = torch.tensor([1. for _ in range(int(0.5 * batch_size))]).type(torch.cuda.FloatTensor)
    target_mask_family = torch.tensor([1. for _ in range(int(0.5 * batch_size))]).type(torch.cuda.FloatTensor)
    target_mask_others = torch.tensor([1. for _ in range(int(0.5 * batch_size))]).type(torch.cuda.FloatTensor)
    for index in range(int(0.5 * batch_size)):
        # ??????????????????????????????????????????????????????1.
        if target_left[index].item() == target_right[index].item():
            target_mask_species[index] = 1.
        # ?????????????????????????????????????????????????????????1.
        elif name_list[target_left[index].item()].split(' ')[0] == name_list[target_right[index].item()].split(' ')[0]:
            target_mask_genus[index] = 1.
        # ????????????????????????????????????????????????????????????0.4
        elif name_dict[name_list[target_left[index].item()]]["family"] == \
                name_dict[name_list[target_right[index].item()]]["family"]:
            target_mask_family[index] = 0.8
        # ???????????????????????????????????????????????????????????????0.2
        else:
            target_mask_others[index] = 0.5

    # ??????loss genus?????????
    loss = loss * target_mask_species
    # ??????loss genus?????????
    loss = loss * target_mask_genus
    # ??????loss family?????????
    loss = loss * target_mask_family
    # ??????loss others?????????
    loss = loss * target_mask_others

    loss = loss.sum() / int(0.5 * batch_size)
    loss = loss * 0.5
    print("graph_rise_loss", loss)
    return loss

