"""
将多个类别的json文件进行合并
"""
import os
import json
from pathlib import Path

json1 = Path(r'D:\temp_dir\json_merge\Paraorthograptus angustus.json')
json2 = Path(r'D:\temp_dir\json_merge\Paraorthograptus pacificus.json')
json3 = Path(r'D:\temp_dir\json_merge\Paraorthograptus typicus.json')
merge_path = Path(r'D:\temp_dir\json_merge\merged\Paraorthograptus pacificus.json')

content1 = json.loads(json1.read_text())
content2 = json.loads(json2.read_text())
content3 = json.loads(json3.read_text())
merge_content = {}
for k in list(content1.keys()):
    merge_content[k] = []
# dict_keys(['images', 'categories', 'annotations'])
images1 = content1['images']
images2 = content2['images']
images3 = content3['images']

categories1 = content1['categories']
categories2 = content2['categories']
categories3 = content3['categories']

annotations1 = content1['annotations']
annotations2 = content2['annotations']
annotations3 = content3['annotations']

print("整合前：")
print(len(images1), len(images2), len(images3))
print(len(categories1), len(categories2), len(categories3))
print(len(annotations1), len(annotations2), len(annotations3))

images_all = []
categories_all = []
annotations_all = []

images_all.extend(images1)
images_all.extend(images2)
images_all.extend(images3)

categories_all.extend(categories1)
categories_all.extend(categories2)
categories_all.extend(categories3)

annotations_all.extend(annotations1)
annotations_all.extend(annotations2)
annotations_all.extend(annotations3)

print(len(images_all), len(categories_all), len(annotations_all))
merge_content["images"] = images_all
merge_content["categories"] = categories_all
merge_content["annotations"] = annotations_all

print("整合后：")
for k in merge_content.keys():
    print(k, len(merge_content[k]))

merge_content = json.dumps(merge_content)
merge_path.write_text(merge_content)
