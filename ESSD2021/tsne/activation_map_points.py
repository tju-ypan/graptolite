# encoding:utf-8
import shutil
import re
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
# resume_path = r"D:\epoch_0041_top1_81.000_'checkpoint.pth.tar'"
# for activation map visualization
exact_list = ['avgpool']
layer_list = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']

# transformer
transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

species_list = os.listdir(r"D:\ESSD2021_CAM\resize")
for index, name in enumerate(species_list):
    if re.match(r'^\d{2}', name):
        species_list[index] = name[2:]
    elif re.match(r'^\d', name):
        species_list[index] = name[1:]
relations_dict = {'Demirastrites triangulatus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Monograptidae'}, 'Dicellograptus tumidus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Dicranograptidae'}, 'Dicellograptus turgidus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Dicranograptidae'}, 'Paraorthograptus pacificus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Diplograptidae'}, 'Spirograptus turriculatus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Monograptidae'}, 'Appendispinograptus venustus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Climacograptidae'}, 'Nicholsonograptus fasciculatus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Sinograptidae'}, 'Nicholsonograptus praelongus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Sinograptidae'}, 'Paraorthograptus longispinus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Diplograptidae'}, 'Cryptograptus tricornis (Juvenile)': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Glossograptidae'}, 'Phyllograptus anna': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Phyllograptidae'}, 'Rastrites guizhouensis': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Monograptidae'}, 'Tangyagraptus typicus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Dicranograptidae'}, 'Yinograptus grandis': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Lasiograptidae'}, 'Coronograptus cyphus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Monograptidae'}, 'Cystograptus vesiculosus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Normalograptidae'}, 'Normalograptus extraordinarius': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Normalograptidae'}, 'Normalograptus persculptus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Normalograptidae'}, 'Parakidograptus acuminatus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Dimorphograptidae'}, 'Diceratograptus mirus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Dicranograptidae'}, 'Lituigraptus convolutus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Monograptidae'}, 'Paraplegmatograptus connectus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Lasiograptidae'}, 'Pararetiograptus regularis': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Diplograptidae'}, 'Dicellograptus bispiralis': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Dicranograptidae'}, 'Dicellograptus caduceus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Dicranograptidae'}, 'Dicellograptus divaricatus salopiensis': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Dicranograptidae'}, 'Dicellograptus smithi': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Dicranograptidae'}, 'Dicellograptus undatus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Dicranograptidae'}, 'Dicranograptus irregularis': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Dicranograptidae'}, 'Dicranograptus sinensis': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Dicranograptidae'}, 'Didymograptus jiangxiensis': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Didymograptidae'}, 'Didymograptus latus tholiformis': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Didymograptidae'}, 'Didymograptus miserabilis': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Didymograptidae'}, 'Amplexograptus acusiformis': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Diplograptidae'}, 'Amplexograptus fusiformis': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Diplograptidae'}, 'Cryptograptus arcticus sinensis': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Glossograptidae'}, 'Cryptograptus gracilicornis': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Glossograptidae'}, 'Dicellograptus divaricatus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Dicranograptidae'}, 'Dicranograptus nicholsoni parvangulus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Dicranograptidae'}, 'Dicranograptus ramosus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Dicranograptidae'}, 'Didymograptus euodus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Didymograptidae'}, 'Didymograptus linearis longus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Didymograptidae'}, 'Didymograptus saerganensis': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Didymograptidae'}, 'Climacograptus pauperatus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Climacograptidae'}, 'Cryptograptus arcticus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Glossograptidae'}, 'Cryptograptus marcidus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Glossograptidae'}, 'Cryptograptus tricornis': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Glossograptidae'}, 'Glossograptus briaros': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Glossograptidae'}, 'Glossograptus robustus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Glossograptidae'}, 'Glyptograptus plurithecatus wuningensis': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Diplograptidae'}, 'Glyptograptus teretiusculus siccatus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Diplograptidae'}, 'Pseudoclimacograptus parvus jiangxiensis': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Climacograptidae'}, 'Pseudoclimacograptus wannanensis': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Climacograptidae'}, 'Diplograptus proelongatus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Diplograptidae'}, 'Glyptograptus teretiusculus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Diplograptidae'}, 'Jiangxigraptus inculus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Dicranograptidae'}, 'Jishougraptus mui': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Sigmagraptidae'}, 'Leptograptus flaccidus trentonensis': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Dicranograptidae'}, 'Monoclimacis neimengolensis': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Monograptidae'}, 'Pseudoclimacograptus angulatus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Climacograptidae'}, 'Pseudoclimacograptus longus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Climacograptidae'}, 'Pseudoclimacograptus modestus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Climacograptidae'}, 'Pseudoclimacograptus parvus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Climacograptidae'}, 'Amplexograptus disjunctus yangtzensis': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Diplograptidae'}, 'Amplexograptus suni': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Diplograptidae'}, 'Climacograptus miserabilis': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Climacograptidae'}, 'Climacograptus supernus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Climacograptidae'}, 'Dicellograptus ornatus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Dicranograptidae'}, 'Diplograptus modestus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Diplograptidae'}, 'Glyptograptus incertus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Diplograptidae'}, 'Petalolithus elongatus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Neodiplograptidae'}, 'Petalolithus folium': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Neodiplograptidae'}, 'Streptograptus runcinatus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Monograptidae'}, 'Dicellograptus szechuanensis': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Dicranograptidae'}, 'Diplograptus bohemicus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Diplograptidae'}, 'Glyptograptus austrodentatus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Diplograptidae'}, 'Glyptograptus gracilis': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Diplograptidae'}, 'Glyptograptus lungmaensis': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Diplograptidae'}, 'Glyptograptus tamariscus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Diplograptidae'}, 'Glyptograptus tamariscus linealis': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Diplograptidae'}, 'Glyptograptus tamariscus magnus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Diplograptidae'}, 'Reteograptus uniformis': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Climacograptidae'}, 'Retiolites geinitzianus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Neodiplograptidae'}, 'Amplexograptus orientalis': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Diplograptidae'}, 'Climacograptus angustatus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Climacograptidae'}, 'Climacograptus leptothecalis': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Climacograptidae'}, 'Climacograptus minutus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Climacograptidae'}, 'Climacograptus normalis': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Climacograptidae'}, 'Climacograptus tianbaensis': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Climacograptidae'}, 'Colonograptus praedeubeli': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Monograptidae'}, 'Diplograptus angustidens': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Diplograptidae'}, 'Diplograptus diminutus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Diplograptidae'}, 'Rectograptus pauperatus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Diplograptidae'}, 'Amplexograptus confertus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Diplograptidae'}, 'Climacograptus angustus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Climacograptidae'}, 'Climacograptus textilis yichangensis': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Climacograptidae'}, 'Colonograptus deubeli': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Monograptidae'}, 'Dicellograptus cf. complanatus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Dicranograptidae'}, 'Diplograptus concinnus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Diplograptidae'}, 'Pristiograptus variabilis': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Monograptidae'}, 'Pseudoclimacograptus demittolabiosus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Climacograptidae'}, 'Pseudoclimacograptus formosus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Climacograptidae'}, 'Rectograptus abbreviatus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Diplograptidae'}, 'Akidograptus ascensus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Dimorphograptidae'}, 'Amplexograptus cf. maxwelli': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Diplograptidae'}, 'Cardiograptus amplus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Isograptidae'}, 'Climacograptus bellulus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Climacograptidae'}, 'Climacograptus hastatus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Climacograptidae'}, 'Glyptograptus dentatus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Diplograptidae'}, 'Glyptograptus elegans': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Diplograptidae'}, 'Glyptograptus elegantulus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Diplograptidae'}, 'Orthograptus calcaratus': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Diplograptidae'}, 'Trigonograptus ensiformis': {'phylum': 'Hemichordata', 'class': 'Pterobranchia', 'order': 'Graptoloidea', 'family': 'Tetragraptidae'}}

family_list = ["Dicranograptidae", "Didymograptidae", "Diplograptidae", "Glossograptidae", "Climacograptidae",
              "Sigmagraptidae", "Monograptidae", "Neodiplograptidae", "Dimorphograptidae", "Isograptidae",
              "Tetragraptidae", "Sinograptidae", "Lasiograptidae", "Normalograptidae", "Phyllograptidae"]
colors_list = ["#f1707d", "#ae716e", "#cb8e85", "#f1ccb8", "#b8d38f", "#ddff95", "#f1b8f1", "#b8f1ed",
               "#e26538", "#f3d751", "#c490a0", "#f28a63", "#ac5e74", "#e49f5e", "#cb7799"]
class2color = {family: color for family, color in zip(family_list, colors_list)}


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
    plt.savefig(fname='figure_points2.jpg',  dpi=600)


def imscatter(x, y, images, ax=None, zoom=1.):
    if ax is None:
        ax = plt.gca()
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0, image in zip(x, y, images):
        image_name = image.split("\\")[-1]
        if re.match(r'^\d{3}_', image_name):
            image_index = image_name[:3]
        elif re.match(r'^\d{2}_', image_name):
            image_index = image_name[:2]
        elif re.match(r'^\d_', image_name):
            image_index = image_name[:1]
        else:
            print("error", image_name)
            os._exit(0)
        color = class2color[relations_dict[species_list[int(image_index)]]["family"]]
        print(color)
        ax.plot(x0, y0, color=color, marker='o')

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


def copy_resize_crop_img():
    # 复制448裁剪图像
    resize_crop_dir = r'D:\ESSD2021_CAM\resize_crop'
    category_paths = [os.path.join(resize_crop_dir, category) for category in os.listdir(resize_crop_dir)]
    for cid, category in enumerate(category_paths):
        for img_name in os.listdir(category):
            if img_name.find("layer") != -1:
                continue
            img_path = os.path.join(category, img_name)
            img_name = str(cid) + '_' + img_name  # 图像名前加上类别索引
            save_path = os.path.join(r'D:\ESSD2021_CAM\for_tsne\crop', img_name)
            shutil.copy(img_path, save_path)


def copy_resize_img():
    # 复制448图像
    resize_dir = r'D:\ESSD2021_CAM\resize'
    category_paths = [os.path.join(resize_dir, category) for category in os.listdir(resize_dir)]
    for cid, category in enumerate(category_paths):
        for img_name in os.listdir(category):
            if img_name.find("layer") != -1:
                continue
            img_path = os.path.join(category, img_name)
            img_name = str(cid) + '_' + img_name  # 图像名前加上类别索引
            save_path = os.path.join(r'D:\ESSD2021_CAM\for_tsne\ori', img_name)
            shutil.copy(img_path, save_path)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 加载训练好的resnet50模型参数
    model = FineTuneSEResnet50(num_class=113).to(device)
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint['state_dict'])

    # features exactor
    myexactor = ActivationMapExtractor(model, exact_list)

    # 将所有类别的图像复制到一个文件夹用于embedding
    # copy_resize_crop_img()
    # copy_resize_img()

    # 生成图像路径列表
    embedding_dir = r'D:\ESSD2021_CAM\for_tsne\crop'
    embedding_images_list = []  # 保存裁剪图
    ori_embedding_images_list = []  # 保存原图
    for img_name in os.listdir(embedding_dir):
        img_path = os.path.join(embedding_dir, img_name)
        embedding_images_list.append(img_path)
        ori_path = img_path.replace(r"\crop", "\ori").replace("448_crop", "448")
        ori_embedding_images_list.append(ori_path)

    # 用裁剪图的特征向量做tsne，但显示原图
    vectors = [vector_for_embedding(img_path, myexactor).data.cpu().numpy()[0] for img_path in embedding_images_list]
    array_vectors = np.array(vectors)
    print(array_vectors.shape)
    # '''
    # 绘制网格型image embedding
    # '''
    draw_tsne(array_vectors, ori_embedding_images_list)
