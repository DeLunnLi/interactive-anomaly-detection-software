# This code is based on the PaDiM-Anomaly-Detection-Localization-master project
# which is licensed under the Apache License 2.0.
# 
# Copyright [2024] [xiahaifeng1995]
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modifications made by [Delun Li]
# See https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master

import random
from random import sample
import numpy as np
from collections import OrderedDict
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from .datasets import IADDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PaDiM():
    def __init__(self):
        self.train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
        
        self.model = resnet18(pretrained = True, progress = True)
        features_dim = 448
        random_selet_dim = 100
        self.model.to(device)
        self.model.eval()
        # 设置随机种子
        random.seed(1024)
        torch.manual_seed(1024)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1024)

        self.idx = torch.tensor(sample(range(0, features_dim), random_selet_dim))
        
    
    def train(self,file_paths):
        print("train")
        # 利用outputs存储模型的中间输出
        outputs = []
        train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

        def hook(module, input, output):
            outputs.append(output)

        self.model.layer1[-1].register_forward_hook(hook)
        self.model.layer2[-1].register_forward_hook(hook)
        self.model.layer3[-1].register_forward_hook(hook)

        train_dataset = IADDataset(file_paths)
        train_dataloader = DataLoader(train_dataset,batch_size=32,pin_memory=True)
        
        for x,_ in train_dataloader:
            # model prediction
            with torch.no_grad():
                _ = self.model(x.to(device))
            # get intermediate layer outputs
            for k, v in zip(train_outputs.keys(), outputs):
                train_outputs[k].append(v.cpu().detach())
            # initialize hook outputs
            outputs = []
        for k, v in train_outputs.items():
            train_outputs[k] = torch.cat(v, 0)

        # Embedding concat
        embedding_vectors = train_outputs['layer1']
        for layer_name in ['layer2', 'layer3']:
            embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name])

        # randomly select d dimension
        embedding_vectors = torch.index_select(embedding_vectors, 1, self.idx)
        # calculate multivariate Gaussian distribution
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W)
        mean = torch.mean(embedding_vectors, dim=0).numpy()
        cov = torch.zeros(C, C, H * W).numpy()
        I = np.identity(C)
        for i in range(H * W):
            # cov[:, :, i] = LedoitWolf().fit(embedding_vectors[:, :, i].numpy()).covariance_
            cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I
        # save learned distribution
        train_outputs = [mean, cov]
        self.train_outputs = train_outputs
        print("ready")
    
    def test(self,file_paths):
        outputs = []
        test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

        def hook(module, input, output):
            outputs.append(output)

        self.model.layer1[-1].register_forward_hook(hook)
        self.model.layer2[-1].register_forward_hook(hook)
        self.model.layer3[-1].register_forward_hook(hook)

        test_dataset = IADDataset(file_paths)
        test_dataloader = DataLoader(test_dataset,batch_size=32,pin_memory=True)

        for x,_ in test_dataloader:
            # model prediction
            with torch.no_grad():
                _ = self.model(x.to(device))
            # get intermediate layer outputs
            for k, v in zip(test_outputs.keys(), outputs):
                test_outputs[k].append(v.cpu().detach())
            # initialize hook outputs
            outputs = []
        for k, v in test_outputs.items():
            test_outputs[k] = torch.cat(v, 0)

        # Embedding concat
        embedding_vectors = test_outputs['layer1']
        for layer_name in ['layer2', 'layer3']:
            embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name])

        # randomly select d dimension
        embedding_vectors = torch.index_select(embedding_vectors, 1, self.idx)
        
        # calculate distance matrix
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()
        dist_list = []
        for i in range(H * W):
            mean = self.train_outputs[0][:, i]
            conv_inv = np.linalg.inv(self.train_outputs[1][:, :, i])
            dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
            dist_list.append(dist)

        dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

        # upsample
        dist_list = torch.tensor(dist_list)
        score_map = F.interpolate(dist_list.unsqueeze(1), size=x.size(2), mode='bilinear',
                                  align_corners=False).squeeze().numpy()
        
        # apply gaussian smoothing on the score map
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)
        
        # Normalization
        max_score = score_map.max()
        min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score)
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        caculate_heatmap(scores,file_paths)
        return img_scores,scores

def caculate_heatmap_path(file_path):
    normalized_path = os.path.normpath(file_path)
    parts = normalized_path.split(os.sep)
    heatmap_file_name = parts[-2] + parts[-1].replace(".png","_heatmap.png")
    target_dir = os.path.join("cache","heatmap")
    heatmap_path = os.path.join(target_dir,heatmap_file_name)
    return heatmap_path

def caculate_heatmap(scores,file_paths):
    num = len(file_paths)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    for i in range(num):
        heatmap_path = caculate_heatmap_path(file_paths[i])
        heat_map = scores[i] * 255
        plt.figure(figsize=(2.24, 2.24), dpi=100)
        plt.imshow(heat_map,cmap='jet',norm=norm,interpolation='none')
        plt.axis('off')
        plt.gca().set_position([0, 0, 1, 1])  # Remove margins
        plt.savefig(heatmap_path, dpi = 100,bbox_inches='tight', pad_inches=0)
        plt.close()

    
def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z