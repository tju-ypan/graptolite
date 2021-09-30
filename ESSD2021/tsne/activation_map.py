# encoding:utf-8
import torch
import torchvision.models as models
import torch.nn as nn
import cv2
import os
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import TSNE

from senet.se_resnet import FineTuneSEResnet50

# for resnet50
resume_path = r"D:\epoch_0146_top1_69.02655_checkpoint.pth.tar"
# for activation map visualization
single_img_path = r'C:\Users\admin\Desktop\temp_images\resized\Shot_201907121711130877.jpg'
therd_size = 256
exact_list = ['avgpool']
layer_list = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']

# transformer
transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


class FineTuneResnet50(nn.Module):
    def __init__(self, num_class=10):
        super(FineTuneResnet50, self).__init__()
        self.num_class = num_class
        resnet50_net = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet50_net.children())[:-1])
        self.classifier = nn.Linear(2048, self.num_class)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ActivationMapExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(ActivationMapExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = {}
        for i, (name, module) in enumerate(self.submodule.features._modules.items()):
            name = layer_list[i]
            x = module(x)
            # if self.extracted_layers is None or name in self.extracted_layers and 'fc' not in name:
            #     outputs[name] = x
        x = torch.flatten(x, 1)
        x = self.submodule.classifier(x)
        outputs[exact_list[0]] = x
        return outputs


# 传入图像的embedding特征和对应的图像的名字
def draw_tsne(features, imgs):
    print(f">>> t-SNE fitting")
    # 初始化一个TSNE模型，这里的参数设置可以查看SKlearn的官网
    tsne = TSNE(n_components=2, init='pca', perplexity=30)
    Y = tsne.fit_transform(features)
    print(f"<<< fitting over")

    fig, ax = plt.subplots()
    fig.set_size_inches(21.6, 14.4)
    plt.axis('off')
    print(f">>> plotting images")
    imscatter(Y[:, 0], Y[:, 1], imgs, zoom=0.1, ax=ax)
    print(f"<<< plot over")
    plt.savefig(fname='figure.jpg')
    # plt.show()


def imscatter(x, y, images, ax=None, zoom=1.):
    if ax is None:
        ax = plt.gca()
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0, image in zip(x, y, images):
        im = cv2.imread(image)
        im = cv2.resize(im, (224, 224))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_f = OffsetImage(im, zoom=zoom)
        ab = AnnotationBbox(im_f, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists


def vector_for_embedding(img_path, myexactor):
    # 读取图像
    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), -1)
    img = cv2.resize(img, (448, 448))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 归一化
    img = transform(img).cuda()
    img = img.unsqueeze(0)  # 增加一个维度
    # outs字典包含指定层name和输出的feature map的一阶向量形式
    outs = myexactor(img)[exact_list[0]]
    out = outs.view(outs.shape[0], -1)
    return out


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 加载训练好的resnet50模型参数
    model = FineTuneSEResnet50(num_class=113).to(device)
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint['state_dict'])

    # features exactor
    myexactor = ActivationMapExtractor(model, exact_list)

    dir = r'D:\ESSD2021_CAM\resize_crop'
    category_paths = [os.path.join(dir, category) for category in os.listdir(dir)]
    embedding_images_list = []
    ori_embedding_images_list = []
    for cid, category in enumerate(category_paths):
        for img_name in os.listdir(category):
            if img_name.find("layer") != -1:
                continue
            img_path = os.path.join(category, img_name)
            # img_name = str(cid) + '_' + img_name  # 图像名前加上类别索引
            save_path = os.path.join(r'D:\ESSD2021_CAM\for_tsne\crop', img_name)
            # shutil.copy(img_path, save_path)
            embedding_images_list.append(save_path)
            ori_path = save_path.replace(r"\crop", "\ori").replace("448_crop", "448")
            ori_embedding_images_list.append(ori_path)

    # 用裁剪图的特征向量做tsne，但显示原图
    vectors = [vector_for_embedding(img_path, myexactor).data.cpu().numpy()[0] for img_path in embedding_images_list]
    array_vectors = np.array(vectors)
    print(array_vectors.shape)
    '''
    绘制网格型image embedding
    '''
    draw_tsne(array_vectors, ori_embedding_images_list)
