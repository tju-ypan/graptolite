import torch
import torch.nn as nn
import os
import numpy as np
from torchvision.datasets import ImageFolder
from utils.transform import get_transform_for_test
from senet.se_resnet import FineTuneSEResnet50
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# 属
data_root = r"C:\Users\admin\Desktop\fsdownload\set113\test100\genus"
test_weights_path = r"C:\Users\admin\Desktop\fsdownload\set113\test100\genus\epoch_0059_top1_86.000_'checkpoint.pth.tar'"
num_class = 42
save_path = "test100_pr_genus.jpg"
x_points = [0.48, 0.51, 0.59, 0.83, 0.72, 0.47, 0.47, 0.62, 0.42, 0.4, 0.64, 0.34, 0.46, 0.35, 0.45, 0.33, 0.25, 0.23, 0.19, 0.22, 0.2]
y_points = [0.48, 0.51, 0.59, 0.83, 0.72, 0.47, 0.47, 0.62, 0.42, 0.4, 0.64, 0.34, 0.46, 0.35, 0.45, 0.33, 0.25, 0.23, 0.19, 0.22, 0.2]

# 种
# data_root = r"C:\Users\admin\Desktop\fsdownload\set113\test100\species"
# test_weights_path = r"C:\Users\admin\Desktop\fsdownload\set113\test100\species\epoch_0041_top1_81.000_'checkpoint.pth.tar'"
# num_class = 113
# save_path = "test100_pr_species.jpg"
# x_points = [0.14, 0.28, 0.22, 0.68, 0.47, 0.31, 0.09, 0.49, 0.08, 0.13, 0.26, 0.02, 0.22, 0.07, 0.19, 0.16, 0.11, 0.07, 0.04, 0.05, 0.06]
# y_points = [0.14, 0.28, 0.22, 0.68, 0.47, 0.31, 0.09, 0.49, 0.08, 0.13, 0.26, 0.02, 0.22, 0.07, 0.19, 0.16, 0.11, 0.07, 0.04, 0.05, 0.06]

gpu = "cuda:0"
font = {'family': "Arial"}


# mean=[0.948078, 0.93855226, 0.9332005], var=[0.14589554, 0.17054074, 0.18254866]
def test(model, test_path):
    # 加载测试集和预训练模型参数
    test_dir = os.path.join(data_root, 'test_images')
    class_list = list(os.listdir(test_dir))
    class_list.sort()
    transform_test = get_transform_for_test()
    test_dataset = ImageFolder(test_dir, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False)
    checkpoint = torch.load(test_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    score_list = []  # 存储预测得分
    label_list = []  # 存储真实标签
    for i, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()

        outputs = model(inputs)
        prob_tmp = torch.nn.Softmax(dim=1)(outputs)  # (batchsize, nclass)
        score_tmp = prob_tmp  # (batchsize, nclass)

        score_list.extend(score_tmp.detach().cpu().numpy())
        label_list.extend(labels.cpu().numpy())

    score_array = np.array(score_list)
    # 将label转换成onehot形式
    label_tensor = torch.tensor(label_list)
    label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
    label_onehot = torch.zeros(label_tensor.shape[0], num_class)
    label_onehot.scatter_(dim=1, index=label_tensor, value=1)
    label_onehot = np.array(label_onehot)
    print("score_array:", score_array.shape)  # (batchsize, classnum) softmax
    print("label_onehot:", label_onehot.shape)  # torch.Size([batchsize, classnum]) onehot

    # 调用sklearn库，计算每个类别对应的precision和recall
    precision_dict = dict()
    recall_dict = dict()
    average_precision_dict = dict()
    for i in range(num_class):
        precision_dict[i], recall_dict[i], _ = precision_recall_curve(label_onehot[:, i], score_array[:, i])
        average_precision_dict[i] = average_precision_score(label_onehot[:, i], score_array[:, i])
        # print(precision_dict[i].shape, recall_dict[i].shape, average_precision_dict[i])

    # micro
    precision_dict["micro"], recall_dict["micro"], _ = precision_recall_curve(label_onehot.ravel(),
                                                                              score_array.ravel())
    average_precision_dict["micro"] = average_precision_score(label_onehot, score_array, average="micro")
    # print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision_dict["micro"]))

    # 绘制所有类别平均的pr曲线
    plt.figure()
    plt.rcParams['savefig.dpi'] = 600  # 图片像素
    plt.rcParams['figure.dpi'] = 600  # 分辨率

    x_point_avg = np.mean(x_points)
    y_point_avg = np.mean(y_points)

    plt.grid(linestyle='--')
    plt.step(recall_dict['micro'], precision_dict['micro'], where='post')

    plt.plot(x_points, y_points, 'o', color='red', markersize=3)
    plt.plot(x_point_avg, y_point_avg, '+', color="green", markersize=12)

    names = ["M Li", "W Wang", "L Li", "X Tong", "X Ma"]
    names = [str(i) for i in range(1, 22)]
    for i in range(len(y_points)):
        plt.text(x_points[i] + 0.015, y_points[i] - 0.015, names[i], fontsize=3, fontdict=font)
    # for i in range(len(y_points)):
        # plt.text(x_points[i] + 0.015, y_points[i] - 0.015, names[i], fontsize=8, fontdict=font)
        # if i != 3 and i != 4:
        #     plt.text(x_points[i] + 0.015, y_points[i] - 0.015, names[i], fontsize=8, fontdict=font)
        # else:
        #     plt.text(x_points[3] + 0.015, y_points[3] - 0.015, names[3] + ", " + names[4], fontsize=8, fontdict=font)
    plt.xlabel('Recall', fontdict=font)
    plt.ylabel('Precision', fontdict=font)
    plt.ylim([-0.05, 1.05])
    plt.xlim([-0.05, 1.05])
    plt.title(
        'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
            .format(average_precision_dict["micro"]), fontdict=font)
    plt.savefig(save_path)
    # plt.show()


if __name__ == '__main__':
    # 加载模型
    seresnet = FineTuneSEResnet50(num_class=num_class)
    device = torch.device(gpu)
    seresnet = seresnet.to(device)
    test(seresnet, test_weights_path)