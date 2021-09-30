"""
在经过像素级裁剪的图像中进一步裁剪出包含笔石区域的最小矩形框，尽量保证笔石的长宽比不变，并且在四周随机留白
"""
import random
import cv2 as cv
import numpy as np
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor

category_list1 = [  # set100-10
    '1Dicellograptus bispiralis',         # re1
    '1Dicellograptus caduceus',           # re1
    '1Dicellograptus divaricatus salopiensis',
    '1Dicellograptus smithi',             # re1, re2
    '1Dicellograptus undatus',
    '1Dicranograptus irregularis',        # re1
    '1Dicranograptus sinensis',           # re1
    '1Didymograptus jiangxiensis',        # re1
    '1Didymograptus latus tholiformis',   # re1
    '1Didymograptus miserabilis'
]
category_list2 = [  # set100-20
    '2Amplexograptus acusiformis',          # re1
    '2Amplexograptus fusiformis',
    '2Cryptograptus arcticus sinensis',     # re1
    '2Cryptograptus gracilicornis',         # re1
    '2Dicellograptus divaricatus',
    '2Dicranograptus nicholsoni parvangulus',
    '2Dicranograptus ramosus',
    '2Didymograptus euodus',
    '2Didymograptus linearis longus',
    '2Didymograptus saerganensis'
]
category_list3 = [  # set100-30
    '3Climacograptus pauperatus',                   # re1, re2
    '3Cryptograptus arcticus',                      # re1
    '3Cryptograptus marcidus',
    '3Cryptograptus tricornis',                     # re1, re2
    '3Glossograptus briaros',
    '3Glossograptus robustus',
    '3Glyptograptus plurithecatus wuningensis',     # re1
    '3Glyptograptus teretiusculus siccatus',        # re1
    '3Pseudoclimacograptus parvus jiangxiensis',    # re1
    '3Pseudoclimacograptus wannanensis'             # re1, re2, re3
]
category_list4 = [  # set100-40
    '4Diplograptus proelongatus',       # re1
    '4Glyptograptus teretiusculus',     # re1
    '4Jiangxigraptus inculus',
    '4Jishougraptus mui',
    '4Leptograptus flaccidus trentonensis',
    '4Monoclimacis neimengolensis',
    '4Pseudoclimacograptus angulatus',
    '4Pseudoclimacograptus longus',
    '4Pseudoclimacograptus modestus',
    '4Pseudoclimacograptus parvus'
]
category_list5 = [  # set100-50
    '5Amplexograptus disjunctus yangtzensis',
    '5Amplexograptus suni',             # re1
    '5Climacograptus miserabilis',      # re1
    '5Climacograptus supernus',         # re1
    '5Dicellograptus ornatus',          # re1, re2
    '5Diplograptus modestus',
    '5Glyptograptus incertus',
    '5Petalolithus elongatus',
    '5Petalolithus folium',
    '5Streptograptus runcinatus'
]
category_list6 = [  # set100-60
    '6Dicellograptus szechuanensis',
    '6Diplograptus bohemicus',
    '6Glyptograptus austrodentatus',    # re1, re2
    '6Glyptograptus gracilis',          # re1
    '6Glyptograptus lungmaensis',
    '6Glyptograptus tamariscus',        # re1
    '6Glyptograptus tamariscus linealis',
    '6Glyptograptus tamariscus magnus',
    '6Reteograptus uniformis',
    '6Retiolites geinitzianus'
]
category_list7 = [  # set100-70
    '7Amplexograptus orientalis',
    '7Climacograptus angustatus',       # re1
    '7Climacograptus leptothecalis',    # re1
    '7Climacograptus minutus',          # re1
    '7Climacograptus normalis',
    '7Climacograptus tianbaensis',
    '7Colonograptus praedeubeli',
    '7Diplograptus angustidens',
    '7Diplograptus diminutus',
    '7Rectograptus pauperatus'
]
category_list8 = [  # set100-80
    '8Amplexograptus confertus',
    '8Climacograptus angustus',                 # re1
    '8Climacograptus textilis yichangensis',    # re1
    '8Colonograptus deubeli',
    '8Dicellograptus cf. complanatus',
    '8Diplograptus concinnus',
    '8Pristiograptus variabilis',
    '8Pseudoclimacograptus demittolabiosus',
    '8Pseudoclimacograptus formosus',
    '8Rectograptus abbreviatus'
]
category_list9 = [  # set100-90
    '9Akidograptus ascensus',
    '9Amplexograptus cf. maxwelli',
    '9Cardiograptus amplus',
    '9Climacograptus bellulus',
    '9Climacograptus hastatus',
    '9Glyptograptus dentatus',
    '9Glyptograptus elegans',
    '9Glyptograptus elegantulus',
    '9Orthograptus calcaratus',
    '9Trigonograptus ensiformis'
]
category_list10 = [  # set100-96
    '10Demirastrites triangulatus',
    '10Dicellograptus tumidus',
    '10Dicellograptus turgidus',
    '10Paraorthograptus pacificus',     # re1
    '10Paraorthograptus simplex',
    '10Spirograptus turriculatus'
]
category_list11 = [  # set100
    '11Appendispinograptus venustus',   # re1
    '11Nicholsonograptus fasciculatus',
    '11Nicholsonograptus praelongus',
    '11Paraorthograptus longispinus'
]
category_list12 = [  # set105
    '12Cryptograptus tricornis (Juvenile)',
    '12Phyllograptus anna',
    '12Rastrites guizhouensis',     # re1, re2
    '12Tangyagraptus typicus',      # re1
    '12Yinograptus grandis'
]
category_list13 = [  # set110
    '13Coronograptus cyphus',               # re1
    '13Cystograptus vesiculosus',           # re1
    '13Normalograptus extraordinarius',     # re1, re2
    '13Normalograptus persculptus',         # re1, re2
    '13Parakidograptus acuminatus'
]
category_list14 = [  # set114
    '14Diceratograptus mirus',
    '14Lituigraptus convolutus',
    '14Paraplegmatograptus connectus',    # re1
    '14Pararetiograptus regularis',
]
category_list = category_list1 + category_list2 + category_list3 + category_list4 + category_list5 + category_list6 + \
                category_list7 + category_list8 + category_list9 + category_list10 + category_list11 + category_list12 +\
                category_list13 + category_list14
category_paths = [Path(r'D:\set113_ori\annotated_images_550') / category for category in category_list]


# 在纯色背景图像中裁剪出包含前景信息的最小矩形框，保持长宽比并在四周留白
def find_smallest_rectangle(dir_path):
    output_size = 448
    for path in dir_path.iterdir():
        if path.suffix == '.jpg':
            img_path = str(path)
            img_name = path.name
            img = cv.imdecode(np.fromfile(img_path), -1)
            img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
            h, w = img_gray.shape
            if h == w == output_size:
                continue
            # bk_color = img_gray[0, 0]
            bk_color = 255  # 白色

            # 以背景色为阈值，遍历出图像中笔石区域的边界
            step = 5
            leftmost = w
            for i in range(0, h, step):
                for j in range(0, w, step):
                    if img_gray[i, j] != bk_color:
                        if j < leftmost:
                            leftmost = j
                            break
            rightmost = 0
            for i in range(0, h, step):
                for j in range(w - 1, -1, -step):
                    if img_gray[i, j] != bk_color:
                        if j > rightmost:
                            rightmost = j
                            break
            highest = h
            for i in range(0, w, step):
                for j in range(0, h, step):
                    if img_gray[j, i] != bk_color:
                        if j < highest:
                            highest = j
                            break
            lowest = 0
            for i in range(0, w, step):
                for j in range(h - 1, -1, -step):
                    if img_gray[j, i] != bk_color:
                        if j > lowest:
                            lowest = j
                            break
            # 笔石区域的高和宽
            new_width = abs(rightmost - leftmost)
            new_high = abs(lowest - highest)
            margin = int(abs(new_width - new_high))

            # 笔石区域的高>宽
            if new_high > new_width:
                mid = abs(rightmost - new_width / 2)
                # 左右边距均足，则将margin平分
                if mid - (new_width / 2 + margin / 2) >= 0 and mid + (new_width / 2 + margin / 2) / 2 <= w:
                    new_img = img[highest:lowest,
                              int(mid - (new_width / 2 + margin / 2)):int(mid + (new_width / 2 + margin / 2))]
                # 左边距不足，则从最左端开始切片，并尽量将剩余margin全部分配给右半部分（若右也不足则取最右）
                elif mid - (new_width / 2 + margin / 2) < 0 and mid + (new_width / 2 + margin / 2) / 2 <= w:
                    left_margin = int(mid - new_width / 2)
                    right_margin = margin - left_margin
                    new_img = img[
                              highest:lowest,
                              0:int(mid + (new_width / 2 + right_margin)) if int(
                                  mid + (new_width / 2 + right_margin)) <= w else w
                              ]
                # 右边距不足，则切片至图像最右端，并尽量将剩余margin全部分配给左半部分（若左也不足则取最左端）
                elif mid - (new_width / 2 + margin / 2) >= 0 and mid + (new_width / 2 + margin / 2) / 2 > w:
                    right_margin = int(w - rightmost)
                    left_margin = margin - right_margin
                    new_img = img[
                              highest:lowest,
                              int(mid - (new_width / 2 + left_margin)) if int(
                                  mid - (new_width / 2 + left_margin)) >= 0 else 0:w
                              ]
                # 左右边距均不足，则横向切片取全部
                else:
                    new_img = img[highest:lowest, :]

            # 笔石区域的高<宽
            elif new_high < new_width:
                mid = abs(lowest - new_high / 2)
                # 上下边距均足, 则将margin平分
                if mid - (new_high / 2 + margin / 2) >= 0 and mid + (new_high / 2 + margin / 2) <= h:
                    new_img = img[int(mid - (new_high / 2 + margin / 2)):int(mid + (new_high / 2 + margin / 2)),
                              leftmost:rightmost]
                # 上边距不足，则从最上端开始切片，并尽量将剩余margin全部分配给下半部分（若下也不足则取最下端）
                elif mid - (new_high / 2 + margin / 2) < 0 and mid + (new_high / 2 + margin / 2) <= h:
                    up_margin = int(mid - new_high / 2)
                    bottom_margin = margin - up_margin
                    new_img = img[
                              0:int(mid + (new_high / 2 + bottom_margin)) if int(
                                  mid + (new_high / 2 + bottom_margin)) <= h else h,
                              leftmost:rightmost
                              ]
                # 下边距不足，则切片至图像最下端，并尽量将剩余margin全部分配给上半部分（若上也不足则取最上端）
                elif mid - (new_high / 2 + margin / 2) >= 0 and mid + (new_high / 2 + margin / 2) > h:
                    bottom_margin = int(h - lowest)
                    up_margin = margin - bottom_margin
                    new_img = img[
                              int(mid - (new_high / 2 + up_margin)) if int(
                                  mid - (new_high / 2 + up_margin)) >= 0 else 0:h,
                              leftmost:rightmost
                              ]
                # 上下边距均不足，则纵向切片取全部
                else:
                    new_img = img[:, leftmost:rightmost]
            # 笔石区域的高宽刚好相等
            else:
                new_img = img[highest:lowest, leftmost:rightmost]

            # 为图像四周留白，大小为笔石区域的边长×0.2
            white_margin = 30
            half_white_margin = int(white_margin / 2)
            new_img = np.pad(new_img, ((half_white_margin, half_white_margin),
                                       (half_white_margin, half_white_margin),
                                       (0, 0)), 'constant', constant_values=255)

            # 随机缩放至size_list中的任一尺寸
            size_list = [300, 330, 370, 400]
            random_index = random.randint(0, len(size_list) - 1)
            random_size = size_list[random_index]
            new_img = cv.resize(new_img, (random_size, random_size))

            # 横纵随机填充至output_size
            padding = output_size - random_size
            w_padding = random.randint(5, padding-5)
            half_w_padding = padding - w_padding
            h_padding = random.randint(5, padding-5)
            half_h_padding = padding - h_padding
            new_img = np.pad(new_img, ((w_padding, half_w_padding),
                                       (h_padding, half_h_padding),
                                       (0, 0)), 'constant', constant_values=255)

            # new_img_mask = np.zeros((output_size, output_size, 3))
            # new_img = new_img + new_img_mask

            new_img = cv.resize(new_img, (output_size, output_size))
            is_success, im_buf_arr = cv.imencode('.jpg', new_img)
            im_buf_arr.tofile(img_path)
            print("reshape: ", img_path, new_img.shape)
            assert (new_img.shape[0] == new_img.shape[1] == output_size)
        else:
            continue
    print("类别 %s 裁剪完毕" % str(dir_path).split("\\")[-1])


if __name__ == '__main__':
    t1 = time.time()
    with ProcessPoolExecutor(2) as executor:
        results = executor.map(find_smallest_rectangle, category_paths)
    print(time.time() - t1)
