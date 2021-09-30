import os
import re
import json
import pandas as pd

# 读取修订的属种名关系{修订前: 修订后}
with open(r"C:\Users\admin\Desktop\set113_revision.txt", 'r') as f:
    revision_dict = json.loads(f.read())
revision_dict_reverse = {}
for key, value in revision_dict.items():
    revision_dict_reverse[value] = key
print(len(revision_dict), revision_dict)
print(len(revision_dict_reverse), revision_dict_reverse)

valid_num = 0
ori_dir = r"D:\set113_ori\annotated_images_cleaning\annotated_images"
category_list = os.listdir(ori_dir)
file_path = r'C:\Users\admin\Desktop\属种图像名.xls'
df = pd.read_excel(file_path, sheet_name=0, header=0)
for index, row in df.iterrows():
    temp_index = 0
    category_name = row["属种"].strip()
    image_name = row["标本编号"]+".jpg"
    for category in category_list:
        category_path = os.path.join(ori_dir, category)
        for img in os.listdir(category_path):
            if img == image_name:
                temp_index += 1
                valid_num += 1
    print(temp_index)
    # break
print(valid_num)
