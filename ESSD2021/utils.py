import os
import re

# 修改命名不规范问题
def clean_name(dir_path):
    for img_name in os.listdir(dir_path):
        new_name = img_name
        img_path = os.path.join(dir_path, img_name)

        if img_name.endswith("JPG"):
            new_name = new_name.replace("JPG", "jpg")
        if img_name.find("  "):
            new_name = new_name.replace("  ", " ")
        if img_name.find(" ."):
            new_name = new_name.replace(" .", ".")
        if img_name.find(u'\xa0'):
            new_name = new_name.replace(u'\xa0', u' ')
        if img_name.find('Glyptograptus tamariscus linearis'):
            new_name = new_name.replace('Glyptograptus tamariscus linearis', 'Glyptograptus tamariscus linealis')

        new_path = img_path.replace(img_name, new_name)
        if img_name != new_name:
            os.rename(img_path, new_path)
            print(img_name, new_name)


# 删除类别名前的前缀
def delete_pre(dir_path):
    category_list = os.listdir(dir_path)
    for category_name in category_list:
        ori_category_path = os.path.join(dir_path, category_name)
        if re.search(r"^\d{2}", category_name):
            pure_category_name = category_name[2:]
        elif re.search(r"^\d", category_name):
            pure_category_name = category_name[1:]
        pure_category_path = os.path.join(dir_path, pure_category_name)
        os.rename(ori_category_path, pure_category_path)


if __name__ == '__main__':
    # clean_name(r"D:\ESSD2021_ori")
    delete_pre(r"D:\ESSD2021_CAM\image_add")
