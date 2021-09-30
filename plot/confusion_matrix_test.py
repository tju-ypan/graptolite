import re
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

nclasses = 113
test_results = \
[0, 0, 0, 4, 5, 5, 5, 6, 6, 7, 7, 8, 8, 8, 11, 11, 11, 11, 14, 14, 14, 34, 16, 16, 17, 17, 79, 19, 73, 19, 21, 21, 21, 22, 22, 22, 22, 23, 30, 30, 30, 53, 38, 39, 109, 45, 45, 48, 48, 48, 53, 54, 111, 45, 70, 70, 73, 73, 73, 73, 75, 75, 69, 75, 93, 77, 92, 78, 109, 109, 81, 46, 81, 101, 101, 101, 84, 102, 102, 102, 104, 104, 104, 104, 105, 105, 105, 105, 108, 108, 108, 110, 61, 84, 111, 79, 111, 112, 112, 112]
test_labels = \
[0, 0, 0, 4, 5, 5, 5, 6, 6, 7, 7, 8, 8, 8, 11, 11, 11, 11, 14, 14, 14, 16, 16, 16, 17, 17, 17, 19, 19, 19, 21, 21, 21, 22, 22, 22, 22, 23, 30, 30, 30, 39, 39, 39, 45, 45, 45, 48, 48, 48, 54, 54, 54, 70, 70, 70, 73, 73, 73, 73, 75, 75, 75, 75, 77, 77, 77, 78, 78, 78, 81, 81, 81, 101, 101, 101, 101, 102, 102, 102, 104, 104, 104, 104, 105, 105, 105, 105, 108, 108, 108, 110, 110, 110, 111, 111, 111, 112, 112, 112]
print('种平均准确率为: {:.4f}'.format(np.equal(test_results, test_labels).astype(np.float32).mean()))
acc_list = []
name_list = ['10Demirastrites triangulatus', '10Dicellograptus tumidus', '10Dicellograptus turgidus',
             '10Paraorthograptus pacificus', '10Spirograptus turriculatus', '11Appendispinograptus venustus',
             '11Nicholsonograptus fasciculatus', '11Nicholsonograptus praelongus', '11Paraorthograptus longispinus',
             '12Cryptograptus tricornis (Juvenile)', '12Phyllograptus anna', '12Rastrites guizhouensis',
             '12Tangyagraptus typicus', '12Yinograptus grandis', '13Coronograptus cyphus', '13Cystograptus vesiculosus',
             '13Normalograptus extraordinarius', '13Normalograptus persculptus', '13Parakidograptus acuminatus',
             '14Diceratograptus mirus', '14Lituigraptus convolutus', '14Paraplegmatograptus connectus',
             '14Pararetiograptus regularis', '1Dicellograptus bispiralis', '1Dicellograptus caduceus',
             '1Dicellograptus divaricatus salopiensis', '1Dicellograptus smithi', '1Dicellograptus undatus',
             '1Dicranograptus irregularis', '1Dicranograptus sinensis', '1Didymograptus jiangxiensis',
             '1Didymograptus latus tholiformis', '1Didymograptus miserabilis', '2Amplexograptus acusiformis',
             '2Amplexograptus fusiformis', '2Cryptograptus arcticus sinensis', '2Cryptograptus gracilicornis',
             '2Dicellograptus divaricatus', '2Dicranograptus nicholsoni parvangulus', '2Dicranograptus ramosus',
             '2Didymograptus euodus', '2Didymograptus linearis longus', '2Didymograptus saerganensis',
             '3Climacograptus pauperatus', '3Cryptograptus arcticus', '3Cryptograptus marcidus',
             '3Cryptograptus tricornis', '3Glossograptus briaros', '3Glossograptus robustus',
             '3Glyptograptus plurithecatus wuningensis', '3Glyptograptus teretiusculus siccatus',
             '3Pseudoclimacograptus parvus jiangxiensis', '3Pseudoclimacograptus wannanensis',
             '4Diplograptus proelongatus', '4Glyptograptus teretiusculus', '4Jiangxigraptus inculus',
             '4Jishougraptus mui', '4Leptograptus flaccidus trentonensis', '4Monoclimacis neimengolensis',
             '4Pseudoclimacograptus angulatus', '4Pseudoclimacograptus longus', '4Pseudoclimacograptus modestus',
             '4Pseudoclimacograptus parvus', '5Amplexograptus disjunctus yangtzensis', '5Amplexograptus suni',
             '5Climacograptus miserabilis', '5Climacograptus supernus', '5Dicellograptus ornatus',
             '5Diplograptus modestus', '5Glyptograptus incertus', '5Petalolithus elongatus', '5Petalolithus folium',
             '5Streptograptus runcinatus', '6Dicellograptus szechuanensis', '6Diplograptus bohemicus',
             '6Glyptograptus austrodentatus', '6Glyptograptus gracilis', '6Glyptograptus lungmaensis',
             '6Glyptograptus tamariscus', '6Glyptograptus tamariscus linealis', '6Glyptograptus tamariscus magnus',
             '6Reteograptus uniformis', '6Retiolites geinitzianus', '7Amplexograptus orientalis',
             '7Climacograptus angustatus', '7Climacograptus leptothecalis', '7Climacograptus minutus',
             '7Climacograptus normalis', '7Climacograptus tianbaensis', '7Colonograptus praedeubeli',
             '7Diplograptus angustidens', '7Diplograptus diminutus', '7Rectograptus pauperatus',
             '8Amplexograptus confertus', '8Climacograptus angustus', '8Climacograptus textilis yichangensis',
             '8Colonograptus deubeli', '8Dicellograptus cf. complanatus', '8Diplograptus concinnus',
             '8Pristiograptus variabilis', '8Pseudoclimacograptus demittolabiosus', '8Pseudoclimacograptus formosus',
             '8Rectograptus abbreviatus', '9Akidograptus ascensus', '9Amplexograptus cf. maxwelli',
             '9Cardiograptus amplus', '9Climacograptus bellulus', '9Climacograptus hastatus', '9Glyptograptus dentatus',
             '9Glyptograptus elegans', '9Glyptograptus elegantulus', '9Orthograptus calcaratus',
             '9Trigonograptus ensiformis']
# 去掉开头的批次号
for index, name in enumerate(name_list):
    if re.match(r'^\d{2}', name):
        name_list[index] = name[2:]
    elif re.match(r'^\d', name):
        name_list[index] = name[1:]
# 解决编码问题
name_list = [name[:].replace(u'\xa0', u' ') for name in name_list]

def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Greens):
    global name_list, nclasses
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix'

    # Compute confusion matrix

    # Only use the labels that appear in the data
    #     classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if np.isnan(cm[i][j]):
                    cm[i][j] = 0.
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    # 根据每个类别的准确率排序混淆矩阵
    id_name_acc_list = [(i, line_acc[i], name_list[i]) for i, line_acc in enumerate(cm)]
    id_name_acc_list = sorted(id_name_acc_list, key=lambda x: x[1], reverse=True)
    sort_id = [i[0] for i in id_name_acc_list]
    sort_name = [i[2] for i in id_name_acc_list]
    cm_sort = np.empty((cm.shape[0], cm.shape[1]))
    # 排序每一行
    for i, line in enumerate(cm):
        cm_sort[i] = cm[sort_id[i]]
    cm = cm_sort
    # 排序每一列
    for i, line in enumerate(cm):
        sort_line = [0 for _ in range(nclasses)]
        for j, value in enumerate(line):
            sort_line[j] = line[sort_id[j]]
        cm[i] = sort_line
    # 开始绘制
    valid_index = []
    for i in range(cm.shape[0]):
        if not (cm[:, i].sum() == 0. and cm[i, :].sum() == 0.):
            valid_index.append(i)
    print(valid_index)
    cm = cm[valid_index, :][:, valid_index]
    print(cm.shape)

    fig, ax = plt.subplots(figsize=(60, 60))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    cb = plt.colorbar(im, ax=ax, fraction=0.05, pad=0.05)
    cb.ax.tick_params(labelsize=50)
    # We want to show all ticks...
    plt.rcParams['font.size'] = 50
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        # xticklabels=classes, yticklabels=classes,
        xticklabels=sort_name,
        yticklabels=sort_name,
        title=title)
    ax.set_xlabel('Predicted label', fontsize=50)   # 30
    ax.set_ylabel('Ground truth label', fontsize=50)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",     # 20
             rotation_mode="anchor", fontsize=40)
    plt.setp(ax.get_yticklabels(), fontsize=40)

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt if cm[i, j] > 0 else '.1f'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=35)   # 12

    fig.tight_layout()
    plt.savefig('set113_confusion_matrix.jpg')
    # plt.show()
    return ax


def plot_matrix(test_results, test_labels):
    global nclasses
    sess = tf.InteractiveSession()
    confusion_matrix = tf.confusion_matrix(labels=test_labels, predictions=test_results, num_classes=nclasses).eval()

    plot_confusion_matrix(confusion_matrix, range(nclasses))


if __name__ == '__main__':
    plot_matrix(test_results, test_labels)