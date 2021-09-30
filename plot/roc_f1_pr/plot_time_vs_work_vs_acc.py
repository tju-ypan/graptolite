# x轴是识别一块的时间，y轴是准确率
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# file_path = r'C:\Users\admin\Desktop\笔石鉴定表-修订版.xlsx'
# names_list = ["李明", "王文卉", "李丽霞", "仝晓静", "马譞", "孙翊桐"]
# english_names_list = ["M Li", "W Wang", "L Li", "X Tong", "X Ma", "Y Sun"]

file_path = r'C:\Users\admin\Desktop\鉴定表 for fig3.xlsx'
names_list = ["expert1", "expert2", "expert3", "expert4", "expert5", "expert6", "expert7", "expert8", "expert9",
              "expert10", "expert11", "expert12", "expert13", "expert14", "expert15", "expert16", "expert17",
              "expert18", "expert19", "expert20", "expert21"]
english_names_list = ["expert1", "expert2", "expert3", "expert4", "expert5", "expert6", "expert7", "expert8", "expert9",
                      "expert10", "expert11", "expert12", "expert13", "expert14", "expert15", "expert16", "expert17",
                      "expert18", "expert19", "expert20", "expert21"]

df = pd.read_excel(file_path, sheet_name=0, header=0)
# x轴
time_cost_list = [df[name + "用时"][:100].mean() for name in names_list]
# y轴
species_acc_list = [0.14, 0.3, 0.22, 0.68, 0.57, 0.31, 0.09, 0.49, 0.1, 0.18, 0.27, 0.02, 0.2, 0.16, 0.19, 0.16, 0.12, 0.07, 0.04, 0.05, 0.08]
genus_acc_list = [0.45, 0.58, 0.61, 0.84, 0.82, 0.5, 0.47, 0.62, 0.42, 0.4, 0.66, 0.41, 0.47, 0.41, 0.4, 0.3, 0.27, 0.21, 0.18, 0.19, 0.16]
# z轴
working_time = [20, 10, 5, 2, 8, 1, 10, 2, 5, 5, 30, 22, 20, 15, 15, 5, 5, 2, 1, 1, 1]
expert_type_list = ["s", "s", "s", "s", "s", "s", "j", "s", "j", "j", "s", "j", "j", "j", "o", "o", "o", "o", "o", "o", "o"]
# sort
all_list = [
    (working_time[i], species_acc_list[i], genus_acc_list[i], time_cost_list[i], english_names_list[i], names_list[i], expert_type_list[i])
    for i in range(len(names_list))]
sorted_all_list = sorted(all_list, key=lambda x: x[0], reverse=False)
working_time = [i[0] for i in sorted_all_list]
species_acc_list = [i[1] for i in sorted_all_list]
genus_acc_list = [i[2] for i in sorted_all_list]
time_cost_list = [i[3] for i in sorted_all_list]
english_names_list = [i[4] for i in sorted_all_list]
names_list = [i[5] for i in sorted_all_list]
expert_type_list = [i[6] for i in sorted_all_list]
# avg
time_cost_avg = np.mean(time_cost_list)
species_acc_avg = np.mean(species_acc_list)
genus_acc_avg = np.mean(genus_acc_list)
working_time_avg = np.mean(working_time)


def year2size(year):
    return int(year / 4)


def plot_function(_type):
    font = {'family': "arial"}
    base_size = 4
    if _type == "genus":
        acc_list = genus_acc_list
        model_acc = 0.86
        model_time_cost = 0.02
        model_work_time = 0.008
        title = "Time cost per image vs. Accuracy of genus"
        acc_avg = genus_acc_avg
        y_label = "Accuracy of genus"
    elif _type == "species":
        acc_list = species_acc_list
        model_acc = 0.81
        model_time_cost = 0.02
        model_work_time = 0.008
        title = "Time cost per image vs. Accuracy of species"
        acc_avg = species_acc_avg
        y_label = "Accuracy of species"

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.rcParams['savefig.dpi'] = 600  # 图片像素
    plt.rcParams['figure.dpi'] = 600  # 分辨率

    # 绘制专家坐标点
    legended_size = []
    for i in range(len(names_list)):
        # 圆点大小代表年限
        # if working_time[i] % 5 == 0 and working_time[i] not in legended_size:
        #     ax.plot(time_cost_list[i], acc_list[i], 'o', color='#E18D65',
        #             markersize=base_size + year2size(working_time[i]), label="%d%s" % (working_time[i], " years"))
        #     legended_size.append(working_time[i])
        # else:
        #     ax.plot(time_cost_list[i], acc_list[i], 'o', color='#E18D65',
        #             markersize=base_size + year2size(working_time[i]))

        # academic, industry用两个不同的颜色
        if expert_type_list[i] in ["j", "s"] and expert_type_list[i] not in legended_size:
            ax.plot(time_cost_list[i], acc_list[i], 'o', alpha=0.8, color='#A92334', label="Academic experts")
            legended_size.extend(["j", "s"])
        elif expert_type_list[i] in ["j", "s"]:
            ax.plot(time_cost_list[i], acc_list[i], 'o', alpha=0.8, color='#A92334')
        elif expert_type_list[i] == "o" and expert_type_list[i] not in legended_size:
            ax.plot(time_cost_list[i], acc_list[i], 'o', alpha=0.8, color='#5AA49C', label="Industry experts")
            legended_size.append(expert_type_list[i])
        else:
            ax.plot(time_cost_list[i], acc_list[i], 'o', alpha=0.8, color='#5AA49C')
        ax.text(time_cost_list[i] + 0.8, acc_list[i] - 0.015, english_names_list[i].replace("expert", ""), fontsize=8, fontdict=font)

    # 绘制模型点
    ax.plot(model_time_cost, model_acc, 'o', alpha=0.8, color='#4B79B7', label="SE-Resnet50")
    # ax.text(model_time_cost + 0.8, model_acc - 0.015, "Model", fontsize=10, fontdict=font)

    # 绘制平均
    plt.plot(time_cost_avg, acc_avg, '+', color="green")

    ax.grid(linestyle='--')
    ax.set_title(title, fontdict=font)
    ax.set_xlim([-1, 30])
    ax.set_ylim([0, 1])
    # ax.set_zlim([0, 1])
    ax.set_xlabel('Time cost per image (minutes)', fontdict=font)
    ax.set_ylabel(y_label, fontdict=font)
    plt.legend(fontsize=8)
    # ax.set_zlabel(z_label, fontdict=font)
    plt.savefig("./test100_time_vs_work_vs_acc_genus.jpg")
    plt.show()


if __name__ == '__main__':
    plot_function('genus')
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # x = time_cost_list
    # y = genus_acc_list
    # z = working_hours
    # ax.plot(x, y, z, 'o', label='parametric curve')
    # plt.show()
