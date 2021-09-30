from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
from pathlib import Path
from multiprocessing import Pool
import random

category_list1 = [   # set100-10
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
category_list2 = [   # set100-20
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
category_list3 = [   # set100-30
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
category_list4 = [   # set100-40
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
category_list5 = [   # set100-50
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
category_list6 = [   # set100-60
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
category_list7 = [   # set100-70
                 '7Amplexograptus confertus',
                 '7Climacograptus angustus',
                 '7Climacograptus textilis yichangensis',
                 '7Colonograptus deubeli',
                 '7Dicellograptus cf. complanatus',
                 '7Diplograptus concinnus',
                 '7Pristiograptus variabilis',
                 '7Pseudoclimacograptus demittolabiosus',
                 '7Pseudoclimacograptus formosus',
                 '7Rectograptus abbreviatus'
                 ]
category_list8 = [   # set100-80
                 '8Akidograptus ascensus',
                 '8Amplexograptus cf. maxwelli',
                 '8Cardiograptus amplus',
                 '8Climacograptus bellulus',
                 '8Climacograptus hastatus',
                 '8Glyptograptus dentatus',
                 '8Glyptograptus elegans',
                 '8Glyptograptus elegantulus',
                 '8Orthograptus calcaratus',
                 '8Trigonograptus ensiformis'
                 ]

category_list = category_list1+category_list2+category_list3+category_list4+category_list5+category_list6+category_list7
category_list = category_list8


datagen = ImageDataGenerator(
    # rescale=1./255,
    featurewise_center=False,
    featurewise_std_normalization=False,
    zoom_range=[0.8, 1.2],
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    channel_shift_range=15,
    fill_mode='nearest'
)


def generate_function(category):
    target_number = 1000
    org_dir = os.path.join(r'D:\set100-70_ori\annotated_images_cleaning\annotated_images\test1\train_images', category)
    # org_dir = os.path.join(r'C:\Users\admin\Desktop\test')
    save_dir = os.path.join(r'D:\set100-70\annotated_images_448\test1\train_images', category)
    # save_dir = os.path.join(r'C:\Users\admin\Desktop\test1')

    img_list = os.listdir(org_dir)
    img_num = len(img_list)
    target_num = target_number - len(os.listdir(save_dir))
    for i in range(target_num):
        max_random = img_num - 1
        index = random.randint(0, max_random)  # 随机数字区间, 选择一张被增强的图像
        img_path = Path(org_dir) / img_list[index]
        img_name = img_path.stem
        # load_img返回一个PIL对象
        img = load_img(str(img_path))
        # 将PIL对象转换为np.ndarray对象
        img_array = img_to_array(img)
        # 转换为四维数组(b,h,w,c)
        img_array = img_array.reshape((1,) + img_array.shape)
        # img_array = img_array.astype('float32') / 255
        datagen.fit(img_array)
        batch = datagen.flow(img_array, batch_size=1)
        gen_image = batch.next()
        # 将数组转为Image
        gen_image = array_to_img(gen_image[0])

        save_path = Path(save_dir) / (img_name + "_enhance" + str(i) + ".jpg")
        if save_path.exists():
            save_path = Path(save_dir) / (img_name + "_enhance" + str(i) + "_" + str(i) + ".jpg")
        print("original path :", img_path)
        print("save path:", save_path)
        gen_image.save(save_path)


if __name__ == '__main__':
    p = Pool(10)
    for category in category_list:
        p.apply_async(generate_function, args=(category,))
    p.close()
    p.join()