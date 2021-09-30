import os
import glob
import json
from pathlib import Path

specie_name = 'Phyllograptus anna'
ori_json_path = os.path.join(r'C:\Users\admin\Desktop\待划分属种\第十一批', specie_name, '.exports', specie_name + '.json')
ori_json = json.loads(Path(ori_json_path).read_text(encoding='utf-8'))
json_images = ori_json['images']
json_categories = ori_json['categories']
json_annotations = ori_json['annotations']

modified_json = Path(
    os.path.join(r'C:\Users\admin\Desktop\待划分属种\第十一批', specie_name, '.exports', specie_name + '_new.json'))

wait_divide_path = os.path.join(r'D:\标注过的图片已审核\2020.11.12Set-6第十一批 已审核', specie_name)

check_path = os.path.join(r'C:\Users\admin\Desktop\待划分属种\第十一批', specie_name)
specimens_list = [specimen for specimen in os.listdir(check_path) if specimen[0] != '.']
check_specimen_list = [os.path.join(check_path, specimen) for specimen in specimens_list]

# 存储每张图像所属的标本名
img_specimen = {}

# 遍历每一张图像
for img_path in glob.glob(os.path.join(wait_divide_path, '*.jpg')):
    img_name = img_path.split('\\')[-1]
    img_specimen[img_name] = ''
    # 遍历每一个标本文件夹
    for check_specimen in check_specimen_list:
        specimen_name = check_specimen.split('\\')[-1]
        specimen_images = glob.glob(os.path.join(check_specimen, '*.jpg'))
        # 遍历改标本的每一张图像
        for check_image_path in specimen_images:
            check_image_name = check_image_path.split('\\')[-1]
            if check_image_name == img_name:
                img_specimen[img_name] = specimen_name
                continue

# 遍历json文件中所有images，在img_specimen字典中找到对应的标本号，并添加到path中
for json_image in json_images:
    for img in img_specimen:
        specimen_name = img_specimen[img]
        if json_image['file_name'] == img:
            relative_path = r'/'.join(json_image['path'].split(r'/')[:-1])
            json_image['path'] = os.path.join(relative_path, specimen_name, img).replace('\\', '/')
            print(json_image['path'])
            break

modified_content = {"images": json_images, "categories": json_categories, "annotations": json_annotations}
modified_content = json.dumps(modified_content)
modified_json.write_text(modified_content)
