import os
# import tarfile
from PIL import Image
from tqdm import tqdm
# import urllib.request

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

class IADDataset(Dataset):
    def __init__(self,file_paths, is_train=True,resize=224, cropsize=224):
        self.resize = resize
        self.cropsize = cropsize
        # self.mvtec_folder_path = os.path.join(root_path, 'mvtec_anomaly_detection')

        # download dataset if not exist
        # self.download()

        # load dataset
        self.x = self.load_dataset(file_paths)

        # set transforms
        self.transform_x = T.Compose([T.Resize(resize, Image.LANCZOS),
                                      T.CenterCrop(cropsize),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])

    def __getitem__(self, idx):
        x = self.x[idx]
        x_path = x
        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        return x,x_path

    def __len__(self):
        return len(self.x)

    def load_dataset(self,file_paths):
        x = []
        x.extend(file_paths)

        return list(x)