from conv_autoencoder import MyNet
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch
import copy
import cv2
import os
import numpy as np
import torch.nn.functional as F

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BACKGROUND = 0
CITY = 1
PATH = 2

class CustomDataset(Dataset):
    def __init__(self, transform=None):
        if torch.cuda.is_available():
            self.root_dir="/home/gridsan/jchin/held-karp"
        else:
            self.root_dir = "/Users/samuelchin/Desktop/MIT/Thesis/held-karp/"
        self.transform = transform

    def __len__(self):
        return 20

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
        colors = [BLACK, RED, GREEN]
        classes = [BACKGROUND, CITY, PATH]
        img = self.transform_img(img, colors, classes)
        return img

    def __getitem__(self, idx):
        # inp_name = os.path.join(self.root_dir, "512x512/20/test/input", "input_%05d.png" %idx)
        inp_name = os.path.join(self.root_dir, "512x512/30/train/input", "input_%05d.png" %idx)
        inp = cv2.imread(inp_name)
        orig_inp = copy.deepcopy(inp)
        inp = self.transform_inp(inp)
        inp = torch.as_tensor(inp, dtype=torch.int64)
        inp = F.one_hot(inp)
        inp = torch.transpose(inp, 1, 2)
        inp = torch.transpose(inp, 0, 1)
        inp = torch.squeeze(inp)
        inp = inp.type(torch.FloatTensor)

        # out_name = os.path.join(self.root_dir, "512x512/20/test/output", "output_%05d.png" %idx)
        out_name = os.path.join(self.root_dir, "512x512/30/train/output", "output_%05d.png" %idx)
        out = cv2.imread(out_name)
        orig_out = copy.deepcopy(out)
        out = self.transform_out(out)
        out = torch.as_tensor(out, dtype=torch.int64)
        out = out.type(torch.LongTensor)
        return inp, out, orig_inp, orig_out

img_transform = transforms.Compose([
    transforms.ToTensor(),
])
batch_size = 10
dataset = CustomDataset(transform=img_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ss = models.densenet161(pretrained=True)
model = MyNet(my_pretrained_model=model_ss)
model.load_state_dict(torch.load("conv_autoencoder3.pth"))
model = model.to(device)
model.eval()

k = 0
for data in dataloader:
    print("Loaded Batch")
    inp, out, orig_inp, orig_out = data
    if torch.cuda.is_available():
        inp, out = inp.cuda(), out.cuda()
    # ===================forward=====================
    # print(inp.dtype)
    # print(inp.shape)
    output = model(inp)
    a = np.array(torch.argmax(output.cpu().data, dim=1))
    for l in range(batch_size):
        b = a[l, :, :]
        c = np.stack((b, ) * 3, -1) 
        for i in range(512):
            for j in range(512):
                if np.all(c[i, j, :] == [1,1,1]):
                  c[i,j,:] = RED
                elif np.all(c[i, j, :] == [2,2,2]):
                  c[i,j,:] = GREEN
        img1 = np.array(orig_inp[l])
        img2 = np.array(orig_out[l])
        vis = np.concatenate((img1, img2, c), axis=1)
        cv2.imwrite("eval512x512_30c_gen_m1/output_%05d.png" %k, vis)
        k += 1
        