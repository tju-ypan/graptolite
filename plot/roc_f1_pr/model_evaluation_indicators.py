import os
import torch
import torch.nn as nn
import numpy as np
from torchvision.datasets import ImageFolder
from utils.transform import get_transform_for_test
from senet.se_resnet import FineTuneSEResnet50

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# 属
data_root = r"C:\Users\admin\Desktop\fsdownload\test100\genus"
test_weights_path = r"C:\Users\admin\Desktop\fsdownload\test100\genus\epoch_0059_top1_86.000_'checkpoint.pth.tar'"
num_class = 42
# 种
# data_root = r"C:\Users\admin\Desktop\fsdownload\test100\species"
# test_weights_path = r"C:\Users\admin\Desktop\fsdownload\test100\species\epoch_0041_top1_81.000_'checkpoint.pth.tar'"
# num_class = 113
gpu = "cuda:0"
confuse_matrix = np.zeros((num_class, num_class), dtype=np.int)
matrix_length = len(confuse_matrix)


# mean=[0.948078, 0.93855226, 0.9332005], var=[0.14589554, 0.17054074, 0.18254866]
def test(model, test_path):
    global confuse_matrix
    # 加载测试集和预训练模型参数
    test_dir = os.path.join(data_root, 'test_images')
    transform_test = get_transform_for_test()
    test_dataset = ImageFolder(test_dir, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, drop_last=False, pin_memory=True, num_workers=1)
    checkpoint = torch.load(test_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    for i, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()

        outputs = model(inputs)
        prediction = torch.argmax(outputs, 1)
        confuse_matrix[prediction.item(), labels.item()] += 1

    precision_macro, recall_macro = [], []
    TPs, FPs, TNs, FNs = 0, 0, 0, 0
    for num in range(matrix_length):
        TPR, FPR = 0, 0
        TP, FP, TN, FN = 0, 0, 0, 0
        TP += confuse_matrix[num, num]
        FP += (confuse_matrix[num].sum() - confuse_matrix[num, num])
        for i in range(matrix_length):
            for j in range(matrix_length):
                if i != num and j != num:
                    TN += confuse_matrix[i, j]
        for i in range(matrix_length):
            for j in range(matrix_length):
                if i != num and j == num:
                    FN += confuse_matrix[i, j]
        # for macro
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        if not np.isnan(precision):
            precision_macro.append(precision)
        if not np.isnan(recall):
            recall_macro.append(recall)
        # for micro
        TPs += TP
        FPs += FP
        TNs += TN
        FNs += FN

    precision_micro = TPs / (TPs + FPs)
    recall_micro = TPs / (TPs + FNs)

    f1_micro = (2 * precision_micro * recall_micro) / (precision_micro + recall_micro)
    f1_macro = (2 * np.mean(precision_macro) * np.mean(recall_macro)) / \
               (np.mean(precision_macro) + np.mean(recall_macro))
    print("precision_micro:", precision_micro)
    print("recall_micro:", recall_micro)
    print("f1_micro:", f1_micro)
    print("precision_macro:", np.mean(precision_macro))
    print("recall_macro:", np.mean(recall_macro))
    print("f1_macro:", f1_macro)


if __name__ == '__main__':
    # 加载模型
    seresnet = FineTuneSEResnet50(num_class=num_class)
    device = torch.device(gpu)
    seresnet = seresnet.to(device)
    test(seresnet, test_weights_path)
