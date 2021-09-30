from sklearn.manifold import TSNE
import cv2
import numpy as np
from tqdm import tqdm
from tsne.img_tools import get_image


# 传入图像的embedding特征和对应的图像路径
def draw_tsne(features, images):
    """
    Args:
        :param features: [n_samples  embed_dim], full data embedding of test samples.
        :param images: list [n_samples], list of datapaths corresponding to <feature>
    """
    # 初始化TSNE模型
    print('features shape: ', features.shape)
    tsne = TSNE(n_components=2, init='pca', perplexity=30.)
    Y = tsne.fit_transform(features)

    Y -= Y.min(axis=0)
    Y /= Y.max(axis=0)
    print('normalized Y:', Y.shape)

    constructed_image = grid(Y, images)
    cv2.imwrite('tsne_embedding1.jpg', constructed_image)
    # scatter_image = scatter(Y, images)
    # cv2.imwrite('tsne_embedding2.jpg', scatter_image)


# scatter图  这个的处理是直接把跟之前的图片有遮挡部分的图片取消显示了 (。。。逃)
def scatter(projection_vectors, image_list):
    image_num = len(image_list)
    output_img_size = 2500  # 输出图片的大小
    each_img_size = 50
    tmp_vectors = projection_vectors * output_img_size
    image = np.ones((output_img_size + each_img_size, output_img_size + each_img_size, 3)) + 255
    for i in tqdm(range(image_num)):
        img_path = image_list[i]
        x0, y0 = map(int, tmp_vectors[i])
        small_img, x1, y1, dx, dy = get_image(img_path, each_img_size)
        if small_img is None:
            continue
        # test if there is an image there already
        if np.mean(image[y0 + dy:y0 + dy + y1, x0 + dx:x0 + dx + x1]) != 1:
            continue
        image[y0 + dy:y0 + dy + y1, x0 + dx:x0 + dx + x1] = small_img

    return image


def grid(projection_vectors, image_list):
    #print(projection_vectors)
    output_img_size = 4000
    half_output_img_size = int(output_img_size / 2)
    each_img_size = 50
    ratio = int(output_img_size / each_img_size)
    print("ratio:", ratio)
    tsne_norm = projection_vectors * output_img_size
    print("tsne_norm:", tsne_norm.shape)

    used_imgs = np.equal(projection_vectors[:, 0], None)    # (dataset_size,)

    image = np.ones((half_output_img_size + each_img_size, output_img_size + each_img_size, 3)) + 255
    print(image.shape[:2])
    for x in tqdm(range(ratio)):
        x0 = x * each_img_size  # (50, 2450, 50)
        x05 = (x + 0.5) * each_img_size     # (75, 2475, 50)
        y = 0
        while y < ratio:
            y0 = y * each_img_size
            y0 = int(y0 / 2)
            y05 = (y + 0.5) * each_img_size
            tmp_tsne = tsne_norm - [x05, y05]
            #tmp_tsne[used_imgs] = 99999  # don't use the same img twice
            tsne_dist = np.hypot(tmp_tsne[:, 0], tmp_tsne[:, 1])
            min_index = np.argmin(tsne_dist)
            y += 1
            if used_imgs[min_index] == True:
                continue
            used_imgs[min_index] = True
            img_path = image_list[min_index]
            small_img, x1, y1, dx, dy = get_image(img_path, each_img_size)
            if small_img is None:
                continue

            print(y0, y0+y1, "|", x0, x0+x1)
            image[y0 + dy:y0 + dy + y1, x0 + dx:x0 + dx + x1] = small_img

    return image


if __name__ == '__main__':
    x = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    y = [r'D:\ESSD2021_CAM\resize\1Dicellograptus bispiralis\103201_448.jpg' for _ in range(4)]

    draw_tsne(x, y)
