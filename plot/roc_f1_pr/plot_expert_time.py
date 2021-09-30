import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
font = {'family': "arial"}
# pandas读取文件
file_path = os.path.abspath(r'C:\Users\admin\Desktop\笔石鉴定表-修订版.xlsx')
names = ["李明", "王文卉", "李丽霞", "仝晓静", "马譞"]
names_e = ["M Li", "W Wang", "L Li", "X Tong", "X Ma"]

file_path = os.path.abspath(r'C:\Users\admin\Desktop\鉴定表 for fig3.xlsx')
names = ["expert1", "expert2", "expert3", "expert4", "expert5", "expert6", "expert7", "expert8", "expert9",
             "expert10", "expert11", "expert12", "expert13", "expert14", "expert15", "expert16", "expert17", "expert18",
             "expert19", "expert20", "expert21"]
names_e = ["expert1", "expert2", "expert3", "expert4", "expert5", "expert6", "expert7", "expert8", "expert9",
             "expert10", "expert11", "expert12", "expert13", "expert14", "expert15", "expert16", "expert17", "expert18",
             "expert19", "expert20", "expert21"]
df = pd.read_excel(file_path, sheet_name=0, header=0)
valid_score = [0, 1]
data = {names_e[i]: [] for i in range(len(names))}


def statistics_time_cost(people_name):
    average_time_cost = df[people_name + "用时"][:100].mean()
    for index, row in df.iterrows():
        pattern = re.compile(r'^\d*', re.S)
        species_score = row[people_name + "判定种名得分"]
        if species_score not in valid_score:
            species_score = 0.0
        if species_score in valid_score:
            time_cost = re.search(pattern, str(row[people_name + "用时"]).strip()).group()
            if time_cost == "":
                time_cost = average_time_cost
            data[names_e[names.index(people_name)]].append(int(time_cost))


if __name__ == '__main__':
    # for name in names:
    #     statistics_time_cost(name)
    # for i in data.values():
    #     print(len(i))
    # df_time = pd.DataFrame(data)
    #
    # plt.figure()
    # plt.rcParams['savefig.dpi'] = 600  # 图片像素
    # plt.rcParams['figure.dpi'] = 600  # 分辨率
    # plt.boxplot(df_time, patch_artist=True,
    #             capprops={"color": "#E18D65"},
    #             flierprops={"markeredgecolor": "#E18D65", "markerfacecolor": "#E18D65"},
    #             boxprops={"color": "#88C0A4", "facecolor": "#88C0A4"},
    #             medianprops={"linestyle": "-", "color": "#E18D65"},
    #             whiskerprops={"color": "#E18D65"})
    # plt.title("Time cost of each expert", fontdict=font)
    # plt.grid(linestyle='--')
    # plt.xlabel('Experts', fontdict=font)
    # plt.ylabel('Time cost', fontdict=font)
    # plt.xticks(ticks=[i+1 for i in range(len(names))], labels=names, rotation=45)
    # plt.savefig("./test100_timecost.jpg")
    # plt.show()

    y_points = [0.48, 0.51, 0.59, 0.83, 0.72, 0.47, 0.47, 0.62, 0.42, 0.4, 0.64, 0.34, 0.46, 0.35, 0.45, 0.33, 0.25, 0.23, 0.19, 0.22, 0.2]
    y_points_academic, y_points_industry = [], []
    expert_type_list = ["s", "s", "s", "s", "s", "s", "j", "s", "j", "j", "s", "j", "j", "j", "o", "o", "o", "o", "o",
                        "o", "o"]
    for i in range(len(y_points)):
        if expert_type_list[i] in ["j", "s"]:
            y_points_academic.append(y_points[i])
        else:
            y_points_industry.append(y_points[i])
    data_academic = {
        'sensitivities': y_points_academic
    }
    data_industry = {
        'sensitivities': y_points_industry
    }
    df_academic = pd.DataFrame(data_academic)
    df_industry = pd.DataFrame(data_industry)
    plt.figure()
    plt.rcParams['savefig.dpi'] = 600  # 图片像素
    plt.rcParams['figure.dpi'] = 600  # 分辨率
    plt.boxplot(df_academic, patch_artist=True,
                capprops={"color": "#E18D65"},
                flierprops={"markeredgecolor": "#E18D65", "markerfacecolor": "#E18D65"},
                boxprops={"color": "#88C0A4", "facecolor": "#88C0A4"},
                medianprops={"linestyle": "-", "color": "#E18D65"},
                whiskerprops={"color": "#E18D65"})

    plt.boxplot(df_industry, patch_artist=True,
                capprops={"color": "#E18D65"},
                flierprops={"markeredgecolor": "#E18D65", "markerfacecolor": "#E18D65"},
                boxprops={"color": "#88C0A4", "facecolor": "#88C0A4"},
                medianprops={"linestyle": "-", "color": "#E18D65"},
                whiskerprops={"color": "#E18D65"})
    # plt.title("Time cost of each expert", fontdict=font)
    plt.grid(linestyle='--')
    plt.xlabel('Experts', fontdict=font)
    plt.ylabel('Sensitivities', fontdict=font)
    # plt.xticks(ticks=[i + 1 for i in range(len(names))], labels=names, rotation=45)
    plt.savefig("./test100_timecost.jpg")
    plt.show()
