import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
from torchvision.datasets import ImageFolder
from utils.transform import get_transform_for_test
from senet.se_resnet import FineTuneSEResnet50
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# 属
data_root = r"C:\Users\admin\Desktop\fsdownload\set113\test100\genus"
test_weights_path = r"C:\Users\admin\Desktop\fsdownload\set113\test100\genus\epoch_0059_top1_86.000_'checkpoint.pth.tar'"
num_class = 42
title_name = 'Receiver operating characteristic to multi-class (Genus)'
save_path = "test100_roc_genus.jpg"
x_points = [0.009811320754716982, 0.0098, 0.007735849056603773, 0.003953488372093023, 0.005957446808510639,
            0.010192307692307691, 0.0106, 0.008636363636363636, 0.01183673469387755, 0.01, 0.006666666666666667, 0.0132,
            0.012857142857142857, 0.014130434782608696, 0.013095238095238096, 0.015952380952380954,
            0.018292682926829267, 0.018780487804878048, 0.019285714285714285, 0.018571428571428572, 0.01951219512195122]
y_points = [0.48, 0.51, 0.59, 0.83, 0.72, 0.47, 0.47, 0.62, 0.42, 0.4, 0.64, 0.34, 0.46, 0.35, 0.45, 0.33, 0.25, 0.23,
            0.19, 0.22, 0.2]
# x_points = [0.004545454545454545, 0.006481481481481481, 0.007547169811320755, 0.003953488372093023, 0.005957446808510639, 0.006851851851851852, 0.007222222222222222, 0.0064, 0.008, 0.007586206896551724, 0.0058823529411764705, 0.008035714285714285, 0.010851063829787235, 0.012448979591836735, 0.007380952380952381, 0.009523809523809525, 0.012380952380952381, 0.014523809523809524, 0.015952380952380954, 0.014047619047619048, 0.016428571428571428]
# y_points = [0.75, 0.65, 0.6, 0.83, 0.72, 0.63, 0.61, 0.68, 0.56, 0.56, 0.7, 0.55, 0.49, 0.39, 0.69, 0.6, 0.48, 0.39, 0.33, 0.41, 0.31]


# 种
data_root = r"C:\Users\admin\Desktop\fsdownload\set113\test100\species"
test_weights_path = r"D:\TJU\GBDB\set113\models\test100\species\SE-Resnet50\epoch_0041_top1_81.000_'checkpoint.pth.tar'"
num_class = 113
title_name = 'Receiver operating characteristic to multi-class (Species)'
save_path = "test100_roc_species.jpg"
x_points = [0.005308641975308642, 0.005106382978723404, 0.004588235294117647, 0.002735042735042735,
            0.003955223880597015, 0.0042592592592592595, 0.006232876712328767, 0.004047619047619047,
            0.0060526315789473685, 0.005723684210526316, 0.004836601307189542, 0.007368421052631579,
            0.005909090909090909, 0.006549295774647888, 0.006279069767441861, 0.006614173228346456,
            0.007007874015748031, 0.007322834645669292, 0.007218045112781955, 0.00753968253968254, 0.007768595041322314]
y_points = [0.14, 0.28, 0.22, 0.68, 0.47, 0.31, 0.09, 0.49, 0.08, 0.13, 0.26, 0.02, 0.22, 0.07, 0.19, 0.16, 0.11, 0.07,
            0.04, 0.05, 0.06]
# x_points = [0.0027152317880794704, 0.0038414634146341463, 0.0035428571428571427, 0.002735042735042735, 0.003955223880597015, 0.0035714285714285713, 0.0034838709677419357, 0.003120567375886525, 0.003962264150943396, 0.0042763157894736845, 0.0032679738562091504, 0.004516129032258065, 0.004714285714285714, 0.005410958904109589, 0.003492063492063492, 0.003968253968253968, 0.005, 0.005887096774193548, 0.005813953488372093, 0.00568, 0.006311475409836066]
# y_points = [0.59, 0.37, 0.38, 0.68, 0.47, 0.45, 0.46, 0.56, 0.37, 0.35, 0.5, 0.3, 0.34, 0.21, 0.56, 0.5, 0.36, 0.27, 0.25, 0.29, 0.23]


x_points_academic, x_points_industry, y_points_academic, y_points_industry = [], [], [], []
expert_type_list = ["s", "s", "s", "s", "s", "s", "j", "s", "j", "j", "s", "j", "j", "j", "o", "o", "o", "o", "o", "o", "o"]
for i in range(len(y_points)):
    if expert_type_list[i] in ["j", "s"]:
        x_points_academic.append(x_points[i])
        y_points_academic.append(y_points[i])
    else:
        x_points_industry.append(x_points[i])
        y_points_industry.append(y_points[i])
gpu = "cuda:0"
font = {'family': "arial", "fontsize": 10}
axis_font = {'family': "arial", "fontsize": 12}
title_font = {'family': "arial", "fontsize": 14}


# mean=[0.948078, 0.93855226, 0.9332005], var=[0.14589554, 0.17054074, 0.18254866]
def test(model, test_path):
    global x_points, y_points, x_points_academic, y_points_academic, x_points_industry, y_points_industry
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
    results_list = []   # 存储预测标签
    for i, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()

        outputs = model(inputs)
        results_list.extend(torch.argmax(outputs, 1).cpu().numpy())
        prob_tmp = torch.nn.Softmax(dim=1)(outputs)  # (batchsize, nclass)
        score_tmp = prob_tmp  # (batchsize, nclass)

        score_list.extend(score_tmp.detach().cpu().numpy())
        label_list.extend(labels.cpu().numpy())

    score_array = np.array(score_list)
    print(score_array.shape)
    # 将label转换成onehot形式
    label_tensor = torch.tensor(label_list)
    label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
    label_onehot = torch.zeros(label_tensor.shape[0], num_class)
    # print(label_onehot.shape, label_tensor.shape)
    label_onehot.scatter_(dim=1, index=label_tensor, value=1)
    label_onehot = np.array(label_onehot)

    # print("score_array:", score_array.shape)  # (batchsize, classnum)
    # print("label_onehot:", label_onehot.shape)  # torch.Size([batchsize, classnum])

    # 构建混淆矩阵
    matrix_length = len(class_list)
    model_confusion_matrix = np.zeros((matrix_length, matrix_length), dtype=np.int)
    for index in range(len(results_list)):
        x_coord = results_list[index]
        y_coord = label_list[index]
        model_confusion_matrix[x_coord][y_coord] += 1
    # 根据混淆矩阵计算评估指标
    TPs, FPs, TNs, FNs = 0, 0, 0, 0
    for num in range(matrix_length):
        TP, FP, TN, FN = 0, 0, 0, 0
        TP += model_confusion_matrix[num, num]
        FP += (model_confusion_matrix[num].sum() - model_confusion_matrix[num, num])
        for i in range(matrix_length):
            for j in range(matrix_length):
                if i != num and j != num:
                    TN += model_confusion_matrix[i, j]
        for i in range(matrix_length):
            for j in range(matrix_length):
                if i != num and j == num:
                    FN += model_confusion_matrix[i, j]
        # for micro
        TPs += TP
        FPs += FP
        TNs += TN
        FNs += FN
    TPRs_micro = TPs / (TPs + FNs)
    FPRs_micro = FPs / (FPs + TNs)


    # 调用sklearn库，计算每个类别对应的fpr和tpr
    fpr_dict = dict()
    tpr_dict = dict()
    roc_auc_dict = dict()
    for i in range(num_class):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(label_onehot[:, i], score_array[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
    # micro
    fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(label_onehot.ravel(), score_array.ravel())
    roc_auc_dict["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])

    # macro
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(num_class) if not (np.isnan(tpr_dict[i][0]))]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    invalid_class_num = 0
    for i in range(num_class):
        if (np.isnan(tpr_dict[i][0])):
            invalid_class_num += 1
            continue
        mean_tpr += interp(all_fpr, fpr_dict[i], tpr_dict[i])
    # Finally average it and compute AUC
    mean_tpr /= (num_class - invalid_class_num)
    fpr_dict["macro"] = all_fpr
    tpr_dict["macro"] = mean_tpr
    roc_auc_dict["macro"] = auc(fpr_dict["macro"], tpr_dict["macro"])

    # 绘制所有类别平均的roc曲线
    plt.figure()
    plt.tick_params(labelsize=10)
    lw = 2
    plt.rcParams['savefig.dpi'] = 600  # 图片像素
    plt.rcParams['figure.dpi'] = 600  # 分辨率

    plt.grid(linestyle='--')
    x_points = [1 - x_points[i] for i in range(len(x_points))]
    x_points_academic = [1 - x_points_academic[i] for i in range(len(x_points_academic))]
    x_points_industry = [1 - x_points_industry[i] for i in range(len(x_points_industry))]
    x_point_avg, y_point_avg = np.mean(x_points), np.mean(y_points)
    x_points_academic_avg, y_points_academic_avg = np.mean(x_points_academic), np.mean(y_points_academic)
    x_points_industry_avg, y_points_industry_avg = np.mean(x_points_industry), np.mean(y_points_industry)

    # micro roc
    plt.plot(1 - fpr_dict["micro"], tpr_dict["micro"],
             label='Micro-average ROC of SE-Resnet50: AUC={0:0.3f}'
                   ''.format(roc_auc_dict["micro"]),
             color='#00AEEF', linestyle='-', linewidth=2)
    plt.plot(1 - FPRs_micro, TPRs_micro, '+', color="#00AEEF", alpha=1., markersize=12, markeredgewidth=1.5, label="Average SE-Resnet50")
    plt.plot(x_point_avg, y_point_avg, '+', color="#85C299", alpha=1., markersize=12, markeredgewidth=1.5, label="Average experts ({0})".format(len(x_points)))
    print("1 - FPRs_micro, TPRs_micro:", 1 - FPRs_micro, TPRs_micro)
    print("x_point_avg, y_point_avg:", x_point_avg, y_point_avg)
    # macro roc
    # plt.plot(1 - fpr_dict["macro"], tpr_dict["macro"],
    #          label='macro-average ROC curve (area = {0:0.3f})'
    #                ''.format(roc_auc_dict["macro"]),
    #          color='#E18D65', linestyle='-', linewidth=2)

    # 为每个类别绘制一条曲线
    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    # for i, color in zip(range(num_class), colors):
    # plt.plot(fpr_dict[i], tpr_dict[i], color=color, lw=lw,
    # label='ROC curve of class {0} (area = {1:0.3f})'
    # ''.format(i, roc_auc_dict[i]))
    # plt.plot([0, 1], [0, 1], 'k--', lw=lw)

    # 绘制专家点
    # plt.plot(x_points, y_points, 'o', color='red', markersize=3)
    # plt.plot(x_point_avg, y_point_avg, '+', color="green", markersize=12)
    plt.plot(x_points_academic, y_points_academic, 'o', color='#F15A5B', alpha=1., markersize=3, label="Academic experts ({0})".format(len(x_points_academic)))
    plt.plot(x_points_academic_avg, y_points_academic_avg, '+', color="#F15A5B", alpha=1., markersize=12, markeredgewidth=1.5, label="Average academic experts")
    plt.plot(x_points_industry, y_points_industry, 'o', color='#FDB840', alpha=1., markersize=3, label="Industry experts ({0})".format(len(x_points_industry)))
    plt.plot(x_points_industry_avg, y_points_industry_avg, '+', color="#FDB840", alpha=1., markersize=12, markeredgewidth=1.5, label="Average industry experts")
    print("x_points_academic_avg, y_points_academic_avg:", x_points_academic_avg, y_points_academic_avg)
    print("x_points_industry_avg, y_points_industry_avg:", x_points_industry_avg, y_points_industry_avg)

    # 绘制箱线图
    # data_academic = {'sensitivities': y_points_academic}
    # data_industry = {'sensitivities': y_points_industry}
    # df_academic = pd.DataFrame(data_academic)
    # df_industry = pd.DataFrame(data_industry)
    # plt.boxplot(df_academic, patch_artist=True, widths=0.02,
    #             capprops={"color": "#E18D65"},
    #             flierprops={"markeredgecolor": "#E18D65", "markerfacecolor": "#E18D65"},
    #             boxprops={"color": "#88C0A4", "facecolor": "#88C0A4"},
    #             medianprops={"linestyle": "-", "color": "#E18D65"},
    #             whiskerprops={"color": "#E18D65"})
    #
    # plt.boxplot(df_industry, patch_artist=True, widths=0.02,
    #             capprops={"color": "#E18D65"},
    #             flierprops={"markeredgecolor": "#E18D65", "markerfacecolor": "#E18D65"},
    #             boxprops={"color": "#88C0A4", "facecolor": "#88C0A4"},
    #             medianprops={"linestyle": "-", "color": "#E18D65"},
    #             whiskerprops={"color": "#E18D65"})

    # 标注文字
    # names = ["M Li", "W Wang", "L Li", "X Tong", "X Ma"]
    # names = [str(i) for i in range(1, 22)]
    # for i in range(len(y_points)):
    #     plt.text(x_points[i] + 0.015, y_points[i] - 0.015, names[i], fontsize=3, fontdict=font)
    # if i != 3 and i != 4:
    # plt.text(x_points[i] + 0.015, y_points[i] - 0.015, names[i], fontsize=8, fontdict=font)
    # else:
    # plt.text(x_points[3] + 0.015, y_points[3] - 0.015, names[3] + ", " + names[4], fontsize=8, fontdict=font)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    # plt.xlabel('False Positive Rate', fontdict=axis_font)
    # plt.ylabel('True Positive Rate', fontdict=axis_font)
    plt.xlabel('Specificity', fontdict=axis_font)
    plt.ylabel('Sensitivity', fontdict=axis_font)
    plt.title(title_name, fontdict=title_font)
    plt.legend(loc="lower left", fontsize=10)
    plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    # 加载模型
    seresnet = FineTuneSEResnet50(num_class=num_class)
    device = torch.device(gpu)
    seresnet = seresnet.to(device)
    test(seresnet, test_weights_path)
