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
expert_type_list = ["s", "s", "s", "s", "s", "s", "j", "s", "j", "j", "s", "j", "j", "j", "o", "o", "o", "o", "o", "o", "o"]
# x轴
time_cost_list = [df[name + "用时"][:100].mean() for name in names_list]
time_cost_academic_list = [time_cost for i, time_cost in enumerate(time_cost_list) if expert_type_list[i] in ["j", "s"]]
time_cost_industry_list = [time_cost for i, time_cost in enumerate(time_cost_list) if expert_type_list[i] not in ["j", "s"]]
# y轴
species_acc_list = [0.14, 0.3, 0.22, 0.68, 0.57, 0.31, 0.09, 0.49, 0.1, 0.18, 0.27, 0.02, 0.2, 0.16, 0.19, 0.16, 0.12, 0.07, 0.04, 0.05, 0.08]
genus_acc_list = [0.45, 0.58, 0.61, 0.84, 0.82, 0.5, 0.47, 0.62, 0.42, 0.4, 0.66, 0.41, 0.47, 0.41, 0.4, 0.3, 0.27, 0.21, 0.18, 0.19, 0.16]
species_acc_academic_list = [species_acc for i, species_acc in enumerate(species_acc_list) if expert_type_list[i] in ["j", "s"]]
species_acc_industry_list = [species_acc for i, species_acc in enumerate(species_acc_list) if expert_type_list[i] not in ["j", "s"]]
genus_acc_academic_list = [genus_acc for i, genus_acc in enumerate(genus_acc_list) if expert_type_list[i] in ["j", "s"]]
genus_acc_industry_list = [genus_acc for i, genus_acc in enumerate(genus_acc_list) if expert_type_list[i] not in ["j", "s"]]
# z轴
working_time = [20, 10, 5, 2, 8, 1, 10, 2, 5, 5, 30, 22, 20, 15, 15, 5, 5, 2, 1, 1, 1]

# avg
time_cost_avg = np.mean(time_cost_list)
time_cost_academic_avg = np.mean(time_cost_academic_list)
time_cost_industry_avg = np.mean(time_cost_industry_list)
species_acc_avg = np.mean(species_acc_list)
species_acc_academic_avg = np.mean(species_acc_academic_list)
species_acc_industry_avg = np.mean(species_acc_industry_list)
genus_acc_avg = np.mean(genus_acc_list)
genus_acc_academic_avg = np.mean(genus_acc_academic_list)
genus_acc_industry_avg = np.mean(genus_acc_industry_list)
working_time_avg = np.mean(working_time)


def year2size(year):
    return int(year / 4)


def plot_function(_type):
    font = {'family': "arial", "fontsize": 10}
    axis_font = {'family': "arial", "fontsize": 12}
    title_font = {'family': "arial", "fontsize": 14}
    base_size = 4
    title = "Time cost per image vs. Accuracy of Genus and Species"

    genus_model_acc = 0.86
    genus_model_time_cost = 0.02
    genus_y_label = "Accuracy of genus"
    species_model_acc = 0.81
    species_model_time_cost = 0.02
    species_y_label = "Accuracy of species"

    fig = plt.figure()
    fig.subplots_adjust(
        hspace=0,
        # right=0.7
    )
    plt.rcParams['savefig.dpi'] = 600  # 图片像素
    plt.rcParams['figure.dpi'] = 600  # 分辨率

    ax1 = fig.add_subplot(211)
    ax1.tick_params(labelsize=10)
    ax1.grid(linestyle='--')
    # 绘制模型点
    ax1.plot(species_model_time_cost, species_model_acc, 'o', alpha=1., color='#00AEEF', label="SE-Resnet50")
    ax1.plot(time_cost_avg, species_acc_avg, '+', alpha=1., markersize=8, markeredgewidth=1.5, color="#85C299", label="Average experts ({0})".format(len(time_cost_list)))
    print("time_cost_avg, species_acc_avg:", time_cost_avg, species_acc_avg)
    # 绘制species专家坐标点
    legended_size = []
    # academic
    for i in range(len(names_list)):
        if expert_type_list[i] in ["j", "s"] and expert_type_list[i] not in legended_size:
            ax1.plot(time_cost_list[i], species_acc_list[i], 'o', alpha=1., color='#F15A5B', label="Academic experts ({0})".format(len(time_cost_academic_list)))
            legended_size.extend(["j", "s"])
        elif expert_type_list[i] in ["j", "s"]:
            ax1.plot(time_cost_list[i], species_acc_list[i], 'o', alpha=1., color='#F15A5B')
    # 绘制academic平均
    ax1.plot(time_cost_academic_avg, species_acc_academic_avg, '+', alpha=1., markersize=8, markeredgewidth=1.5, color="#F15A5B", label="Average academic experts")
    print("time_cost_academic_avg, species_acc_academic_avg:", time_cost_academic_avg, species_acc_academic_avg)
    # industry
    for i in range(len(names_list)):
        if expert_type_list[i] == "o" and expert_type_list[i] not in legended_size:
            ax1.plot(time_cost_list[i], species_acc_list[i], 'o', alpha=1., color='#FDB840', label="Industry experts ({0})".format(len(time_cost_industry_list)))
            legended_size.append(expert_type_list[i])
        elif expert_type_list[i] == "o":
            ax1.plot(time_cost_list[i], species_acc_list[i], 'o', alpha=1., color='#FDB840')
    # 绘制industry平均
    ax1.plot(time_cost_industry_avg, species_acc_industry_avg, '+', alpha=1., markersize=8, markeredgewidth=1.5, color="#FDB840", label="Average industry experts")
    print("time_cost_industry_avg, species_acc_industry_avg:", time_cost_industry_avg, species_acc_industry_avg)
    ax1.set_title(title, fontdict=title_font)
    ax1.set_xlim([-1, 26])
    ax1.set_ylim([-0.1, 1.1])
    ax1.set_ylabel(species_y_label, fontdict=axis_font)


    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2.tick_params(labelsize=10)
    ax2.grid(linestyle='--')
    # 绘制模型点
    ax2.plot(genus_model_time_cost, genus_model_acc, 'o', alpha=1., color='#00AEEF', label="SE-Resnet50")
    ax2.plot(time_cost_avg, genus_acc_avg, '+', alpha=1., markersize=8, markeredgewidth=1.5, color="#85C299", label="Average experts ({0})".format(len(time_cost_list)))
    print("time_cost_avg, genus_acc_avg:", time_cost_avg, genus_acc_avg)
    # 绘制genus专家坐标点
    legended_size = []
    # academic
    for i in range(len(names_list)):
        if expert_type_list[i] in ["j", "s"] and expert_type_list[i] not in legended_size:
            ax2.plot(time_cost_list[i], genus_acc_list[i], 'o', alpha=1., color='#F15A5B', label="Academic experts ({0})".format(len(time_cost_academic_list)))
            legended_size.extend(["j", "s"])
        elif expert_type_list[i] in ["j", "s"]:
            ax2.plot(time_cost_list[i], genus_acc_list[i], 'o', alpha=1., color='#F15A5B')
    # 绘制academic平均
    ax2.plot(time_cost_academic_avg, genus_acc_academic_avg, '+', alpha=1., markersize=8, markeredgewidth=1.5, color="#F15A5B",label="Average academic experts")
    print("time_cost_academic_avg, genus_acc_academic_avg:", time_cost_academic_avg, genus_acc_academic_avg)
    # industry
    for i in range(len(names_list)):
        if expert_type_list[i] == "o" and expert_type_list[i] not in legended_size:
            ax2.plot(time_cost_list[i], genus_acc_list[i], 'o', alpha=1., color='#FDB840', label="Industry experts ({0})".format(len(time_cost_industry_list)))
            legended_size.append(expert_type_list[i])
        elif expert_type_list[i] == "o":
            ax2.plot(time_cost_list[i], genus_acc_list[i], 'o', alpha=1., color='#FDB840')
    # 绘制industry平均
    ax2.plot(time_cost_industry_avg, genus_acc_industry_avg, '+', alpha=1., markersize=8, markeredgewidth=1.5, color="#FDB840", label="Average industry experts")
    print("time_cost_industry_avg, genus_acc_industry_avg:", time_cost_industry_avg, genus_acc_industry_avg)

    ax2.set_xlim([-1, 26])
    ax2.set_ylim([-0.1, 1.1])
    ax2.set_xlabel('Time cost per image (minutes)', fontdict=axis_font)
    ax2.set_ylabel(genus_y_label, fontdict=axis_font)
    ax1.legend(fontsize=10, loc="upper right", bbox_to_anchor=(0.995, 0.988), borderaxespad=0.)

    plt.savefig("./test100_time_vs_acc_all.jpg")
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
