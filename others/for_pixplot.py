"""
为pixplot整合所有图像并修改格式
"""
import os
import shutil
from PIL import Image

croped_images_dir = r"D:\set113_ori\annotated_images_cleaning\annotated_images"
ori_images_dir1 = r"D:\set113_ori\images"
ori_images_dir2 = r"D:\set113_ori\re_annotation1\images"
ori_images_dir3 = r"D:\set113_ori\re_annotation2\images"
ori_images_dir4 = r"D:\set113_ori\re_annotation3\images"
save_images_dir = r"D:\temp_dir\pixplot\ori_images"

category_paths = [os.path.join(croped_images_dir, category) for category in os.listdir(croped_images_dir)]
for cid, category_path in enumerate(category_paths):
    category_name = category_path.split("\\")[-1]
    for img_name in os.listdir(category_path):
        if os.path.exists(os.path.join(ori_images_dir1, category_name, img_name)):
            ori_img_path = os.path.join(ori_images_dir1, category_name, img_name)
        elif os.path.exists(os.path.join(ori_images_dir2, category_name, img_name)):
            ori_img_path = os.path.join(ori_images_dir2, category_name, img_name)
        elif os.path.exists(os.path.join(ori_images_dir3, category_name, img_name)):
            ori_img_path = os.path.join(ori_images_dir3, category_name, img_name)
        elif os.path.exists(os.path.join(ori_images_dir4, category_name, img_name)):
            ori_img_path = os.path.join(ori_images_dir4, category_name, img_name)
        save_img_name = str(cid) + '_' + img_name
        save_img_path = os.path.join(save_images_dir, save_img_name)
        # img = Image.open(ori_img_path)
        # img = img.resize((224, 224))
        # img.save(save_img_path)
        shutil.copy(ori_img_path, save_img_path)
