import os
import glob
from PIL import Image

dataset_path = r'D:\博物馆图像训练'
save_dataset_path = r'D:\macrofossil'
for category_name in os.listdir(dataset_path):
    category_path = os.path.join(dataset_path, category_name)

    save_category_path = os.path.join(save_dataset_path, category_name)
    if not os.path.exists(save_category_path):
        os.makedirs(save_category_path)

    if os.path.isdir(category_path):
        for img_path in glob.glob(os.path.join(category_path, "*.jpg")):
            img_name = img_path.split("\\")[-1]
            img = Image.open(img_path, 'r')
            img = img.resize((500, 500))
            save_img_path = os.path.join(save_category_path, img_name)
            img.save(save_img_path)
