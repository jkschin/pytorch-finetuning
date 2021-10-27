import cv2
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
import numpy as np

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BACKGROUND = 0
CITY = 1
PATH = 2
# ROOT_DIR="/Users/samuelchin/Desktop/MIT/Thesis/held-karp/"

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.total = len(os.listdir(os.path.join(root_dir, "inputs")))

    def __len__(self):
        return 2
        return self.total

    def transform_img(self, img, colors, classes):
        new_img = np.zeros(img.shape[0:2])
        zipped = zip(colors, classes)
        for color, classs in zipped:
            indexes = np.where((img == color).all(axis=2))
            new_img[indexes] = classs
        return new_img

    def transform_inp(self, img):
        colors = [BLACK, RED, WHITE]
        classes = [BACKGROUND, CITY, PATH]
        img = self.transform_img(img, colors, classes)
        return img

    def transform_out(self, img):
        colors = [GREEN]
        classes = [PATH]
        img = self.transform_img(img, colors, classes)
        return img

    def __getitem__(self, idx):
        inp_name = os.path.join(self.root_dir, "inputs", "input_%05d.png" %idx)
        inp = cv2.imread(inp_name)
        inp = self.transform_inp(inp)
        inp = torch.as_tensor(inp, dtype=torch.int64)
        inp = F.one_hot(inp)
        inp = torch.transpose(inp, 1, 2)
        inp = torch.transpose(inp, 0, 1)
        inp = torch.squeeze(inp)
        inp = inp.type(torch.FloatTensor)

        out_name = os.path.join(self.root_dir, "outputs", "output_%05d.png" %idx)
        out = cv2.imread(out_name)
        out = self.transform_out(out)
        out = torch.as_tensor(out, dtype=torch.int64)
        out = out.type(torch.LongTensor)
        return inp, out