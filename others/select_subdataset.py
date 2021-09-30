"""
从总数据集中选择一个小的子数据集
"""
import os
import random
import shutil
import numpy as np

import re
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder

from senet.se_resnet import FineTuneSEResnet50

shuffle = False
select_nclass = 10
images_per_class = 6
# 总数据集测试集路径
dataset_path = r'D:\set100-70\annotated_images_448\test1\test_images'
ori_dataset_path = r'D:\set100-70_ori\images'
# 子数据集裁剪图像保存路径
sub_dataset_path_crop = r'D:\set100-70\annotated_images_448\test1\sub_test_images'
# 子数据集原始图像保存路径
sub_dataset_path_ori = r'D:\set100-70\annotated_images_448\test1\sub_test_images_ori'
# 模型保存路径
model_path = r"D:\epoch_0278_top1_70.565_'checkpoint.pth.tar'"
transform_test = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize([0.948078, 0.93855226, 0.9332005], [0.14589554, 0.17054074, 0.18254866])
])

if __name__ == '__main__':
    if shuffle:
        # 随机选择select_nclass个类别
        all_class_name = list(os.listdir(dataset_path))
        np.random.shuffle(all_class_name)
        sub_class_name = all_class_name[:select_nclass]
        sub_class_name.sort()
        # 删除已有的子数据集原始图像和裁剪图像
        if os.path.exists(sub_dataset_path_crop):
            shutil.rmtree(sub_dataset_path_crop)
        if os.path.exists(sub_dataset_path_ori):
            shutil.rmtree(sub_dataset_path_ori)

        # 每个类别随机选择images_per_class张图像
        for index, class_name in enumerate(sub_class_name):
            original_class_path = os.path.join(dataset_path, class_name)    # 子类别裁剪图像原始路径
            target_class_path_cro = os.path.join(sub_dataset_path_crop, class_name)  # 子类别裁剪图像保存路径
            if re.match(r'^\d{2}', class_name):
                tmp_name = class_name[:2] + '_' + str(index)
            elif re.match(r'^\d', class_name):
                tmp_name = class_name[:1] + '_' + str(index)
            target_class_path_ori = os.path.join(sub_dataset_path_ori, tmp_name)  # 子类别原始图像保存路径, 隐藏类名用类索引代替

            os.makedirs(target_class_path_cro)  # 创建子类别裁剪图像文件夹
            os.makedirs(target_class_path_ori)  # 创建子类别原始图像文件夹

            images_per_class = random.randint(0, 5)
            img_name_list = list(os.listdir(original_class_path))
            np.random.shuffle(img_name_list)
            random_img_name_list = img_name_list[:images_per_class]
            # 复制子类别原始图像和裁剪图像
            for img_name in random_img_name_list:
                ori_img_path_ori = os.path.join(ori_dataset_path, class_name, img_name)
                tar_img_path_ori = os.path.join(target_class_path_ori, img_name)

                ori_img_path_crop = os.path.join(original_class_path, img_name)
                tar_img_path_crop = os.path.join(target_class_path_cro, img_name)

                if os.path.exists(ori_img_path_ori) and os.path.exists(ori_img_path_crop):
                    shutil.copy(ori_img_path_ori, tar_img_path_ori)
                    shutil.copy(ori_img_path_crop, tar_img_path_crop)
    # os._exit(0)
    # 记录总数据集类名和对应的id
    all_class_name_index_dict = {}
    for i, name in enumerate(os.listdir(dataset_path)):
        all_class_name_index_dict[i] = name
    # print(all_class_name_index_dict)

    # 记录子数据集类名和对应的id
    sub_class_name_index_dict = {}
    for i, name in enumerate(os.listdir(sub_dataset_path_crop)):
        sub_class_name_index_dict[name] = i
    # print(sub_class_name_index_dict)

    # 加载测试数据集和模型
    test_dataset = ImageFolder(sub_dataset_path_crop, transform=transform_test)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
    model = FineTuneSEResnet50(num_class=113, pretrained=False)
    device = torch.device("cuda:0")
    model = model.to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # 开始预测
    test_results = []
    test_labels = []
    for i, (inputs, labels) in enumerate(testloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        # compute output
        outputs = model(inputs)
        test_results.extend(torch.argmax(outputs, 1).cpu().numpy())
        test_labels.extend(labels.cpu().numpy())    # 此时预测标签为总类别的标签
    # 将总类别的类标签转换为子类别的类标签
    for i, value in enumerate(test_results):
        if all_class_name_index_dict[value] in sub_class_name_index_dict.keys():
            test_results[i] = all_class_name_index_dict[value]
    for i, value in enumerate(test_results):
        if isinstance(value, str):
            test_results[i] = sub_class_name_index_dict[value]
    print(test_labels)
    print(test_results)

    # 计算单个类别准确率
    sub_class_name = list(os.listdir(sub_dataset_path_crop))
    species_acc_list = []
    for class_id in range(select_nclass):
        # 保存当前类别的ground truth label的索引
        gt_index_list = []
        for index, class_number in enumerate(test_labels):
            if class_number == class_id:
                gt_index_list.append(index)
        species_acc_num = 0
        for gt_index in gt_index_list:
            if test_results[gt_index] == class_id:
                species_acc_num += 1
        if len(gt_index_list) > 0:
            species_acc = float(format(species_acc_num / len(gt_index_list), '.2f'))
            species_acc_list.append(species_acc)
        elif len(gt_index_list) == 0:
            species_acc_list.append(0)
    for a, b in zip(sub_class_name, species_acc_list):
        print(a, b)
    print("总体识别准确率：", np.equal(test_results, test_labels).astype('float').mean())

