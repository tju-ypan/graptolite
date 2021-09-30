# coding:utf-8
# 对训练集和测试集进行不同的
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
from pathlib import Path
from multiprocessing import Pool
import random
import glob
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
    '10Paraorthograptus simplex',
    '10Spirograptus turriculatus'
]
category_list11 = [  # set102
    '11Appendispinograptus venustus',
    '11Cryptograptus tricornis',
    '11Nicholsonograptus fasciculatus',
    '11Nicholsonograptus praelongus',
    '11Paraorthograptus longispinus',
    '11Pseudoclimacograptus wannanensis'
]
category_list12 = [  # set107
    '12Cryptograptus tricornis (Juvenile)',
    '12Phyllograptus anna',
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


def gen_datagen_train():
    # for train images
    datagen = ImageDataGenerator(
        # rescale=1./255,
        # featurewise_center=False,
        # featurewise_std_normalization=False,
        width_shift_range=0.05,
        height_shift_range=0.05,
        # zoom_range=0.2,
        rotation_range=45,
        # shear_range=0.1,
        # horizontal_flip=True,
        # vertical_flip=False,
        channel_shift_range=50,
        fill_mode='nearest'
    )
    return datagen


def gen_datagen_test():
    # for test images
    datagen = ImageDataGenerator(
        # rescale=1./255,
        featurewise_center=False,
        featurewise_std_normalization=False,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        rotation_range=40,
        shear_range=0.1,
        horizontal_flip=True,
        vertical_flip=False,
        channel_shift_range=15,
        fill_mode='nearest'
    )
    return datagen


datagen = gen_datagen_train()


def image_augmentation(category):
    global datagen
    org_dir = os.path.join(r'D:\set100-70_ori\annotated_images_augmentation', category)
    # org_dir = os.path.join(r'C:\Users\admin\Desktop\tmp_dir', category)
    for img in Path(org_dir).glob('*.jpg'):
        img_path = str(img)
        if not re.search('enhance', img_path):
            # load_img返回一个PIL对象
            img = load_img(img_path)
            # 将PIL对象转换为np.ndarray对象
            img_array = img_to_array(img)
            # 转换为四维数组(b,h,w,c)
            img_array = img_array.reshape((1,) + img_array.shape)
            # img_array = img_array.astype('float32') / 255
            # test or train images
            # datagen = gen_datagen_train()
            datagen.fit(img_array)
            batch = datagen.flow(img_array, batch_size=1)
            gen_image = batch.next()
            # 将数组转为Image对象
            gen_image = array_to_img(gen_image[0])
            # 覆盖原图像
            save_path = img_path
            gen_image.save(save_path)
        else:
            continue
    print(category, ' end.')


if __name__ == '__main__':
    p = Pool(10)
    for category in category_list:
        p.apply_async(image_augmentation, args=(category,))
    p.close()
    p.join()
    # image_augmentation('tmp_aug')
