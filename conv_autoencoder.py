__author__ = 'schin'

import cv2
import torch
from torch import nn
from torchvision import transforms, models
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
from collections import Counter
import numpy as np

output_dir = "20211222-Expt8-512-boosted-1150"
image_size = (512, 512)
if not os.path.exists('./%s' %output_dir):
    os.mkdir('./%s' %output_dir)

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BACKGROUND = 0
CITY = 1
PATH = 2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def to_img(x):
    x = torch.argmax(x, dim=1)
    print(torch.unique(x))
    print(Counter(torch.flatten(x).tolist()))
    x *= 126
    # x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.type(torch.FloatTensor)
    # print(x)
    # print(torch.max(x))
    # print(x.shape)
    # print(x.dtype)
    x = x.view(x.size(0), 1, image_size[0], image_size[1])
    # print(x.dtype)
    # print(torch.max(x))
    return x


num_epochs = 1000
if torch.cuda.is_available():
    batch_size = 16
else:
    batch_size = 1
learning_rate = 1e-3
print(batch_size)

img_transform = transforms.Compose([
    transforms.ToTensor(),
])

class CustomDataset(Dataset):
    def __init__(self, transform=None):
        if torch.cuda.is_available():
            self.root_dir="/home/gridsan/jchin/held-karp"
        else:
            self.root_dir = "/Users/samuelchin/Desktop/MIT/Thesis/held-karp/"
        self.transform = transform

    def __len__(self):
        if torch.cuda.is_available():
            return 10000
        else:
            return 100

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
        inp_name = os.path.join(self.root_dir, "512x512/20/train/input", "input_%05d.png" %idx)
        inp = cv2.imread(inp_name)
        inp = self.transform_inp(inp)
        inp = torch.as_tensor(inp, dtype=torch.int64)
        inp = F.one_hot(inp)
        inp = torch.transpose(inp, 1, 2)
        inp = torch.transpose(inp, 0, 1)
        inp = torch.squeeze(inp)
        inp = inp.type(torch.FloatTensor)

        out_name = os.path.join(self.root_dir, "512x512/20/train/output", "output_%05d.png" %idx)
        out = cv2.imread(out_name)
        out = self.transform_out(out)
        out = torch.as_tensor(out, dtype=torch.int64)
        out = out.type(torch.LongTensor)
        # test = torch.as_tensor(out, dtype=torch.int64)
        # test = F.one_hot(test)
        # test = torch.transpose(test, 1, 2)
        # test = torch.transpose(test, 0, 1)
        # test = torch.squeeze(test)
        # test = test.type(torch.FloatTensor)
        return inp, out

# dataset = MNIST('./data', transform=img_transform, download=True)
dataset = CustomDataset(transform=img_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.block4T = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        self.block3T = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.block2T = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.block1T = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )
        self.pool4 = nn.Sequential(
            nn.Conv2d(512, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(True),
            nn.Upsample(size=image_size, mode="bilinear"),
        )
        self.pool3 = nn.Sequential(
            nn.Conv2d(256, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(True),
            nn.Upsample(size=image_size, mode="bilinear"),
        )
        self.pool2 = nn.Sequential(
            nn.Conv2d(128, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(True),
            nn.Upsample(size=image_size, mode="bilinear"),
        )

        self.last = nn.Sequential(
            nn.Conv2d(36, 3, 1, stride=1),
        )

    def forward(self, x):
        x_orig = x
        x = self.block1(x)
        x = self.block2(x)
        x2 = self.pool2(x)
        x = self.block3(x)
        x3 = self.pool3(x)
        x = self.block4(x)
        x4 = self.pool4(x)
        x = self.block4T(x)
        x = self.block3T(x)
        x = self.block2T(x)
        x = self.block1T(x)
        print(x2.shape)
        x = torch.cat([x2, x3, x4, x, x_orig[:, 1:2, :, :]], 1)
        x = self.last(x)
        return x

# if torch.cuda.is_available():
#     model = autoencoder().cuda()
# else:
#     model = autoencoder()

num_classes = 3
class MyNet(nn.Module):
    def __init__(self, my_pretrained_model):
        super(MyNet, self).__init__()
        self.pretrained = my_pretrained_model
        self.u1 = nn.Sequential(
            nn.BatchNorm2d(2208),
            nn.ReLU(True),
            nn.Conv2d(2208, 32, 1),
            nn.Upsample(size=image_size, mode="bilinear"),
        )
        self.u2 = nn.Sequential(
            nn.BatchNorm2d(2112),
            nn.ReLU(True),
            nn.Conv2d(2112, 32, 1),
            nn.Upsample(size=image_size, mode="bilinear"),
        )
        self.u3 = nn.Sequential(
            nn.BatchNorm2d(768),
            nn.ReLU(True),
            nn.Conv2d(768, 32, 1),
            nn.Upsample(size=image_size, mode="bilinear"),
        )
        self.u4 = nn.Sequential(
            nn.BatchNorm2d(384),
            nn.ReLU(True),
            nn.Conv2d(384, 32, 1),
            nn.Upsample(size=image_size, mode="bilinear"),
        )
        self.last = nn.Sequential(
            nn.Conv2d(130, 3, 1, stride=1),
        )

    def forward(self, x):
        # for name, layer in self.pretrained.named_modules():
        #     print(name, layer)
        #     x = layer(x)
        outputs = []
        x_orig = x
        for ii, model in enumerate(list(self.pretrained.features)):
            if ii in [11]:
                continue
            x = model(x)
            if ii in [2, 4, 6, 8, 10]:
                outputs.append(x)
        u1 = self.u1(outputs[-1])
        u2 = self.u2(outputs[-2])
        u3 = self.u3(outputs[-3])
        u4 = self.u4(outputs[-4])
        x = torch.cat([u1, u2, u3, u4, x_orig[:, 1:, :, :]], 1)
        x = self.last(x)
        # a = self.u1(x)
        # b = self.u2(x)
        # x = self.upsample(self.pretrained.features.relu0)
        # x = self.last(x)
        return x

if __name__ == "__main__":
    model_ss = models.densenet161(pretrained=True)
    # for name, param in model_ss.named_parameters():
    # if not "classifier" in name:
    # param.requires_grad = False
    mynet = MyNet(my_pretrained_model=model_ss)
    model = mynet.to(device)
    from torchinfo import summary
    summary(model, input_size=(batch_size, 3, image_size[0], image_size[1]))
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_uniform
    model.apply(init_weights)
    weights = torch.FloatTensor([1, 1, 4])
    weights = weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    # criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                weight_decay=1e-5)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        print("Epoch: %d" %epoch)
        total_loss = 0
        for data in dataloader:
            print("Loaded Batch")
            inp, out = data
            if torch.cuda.is_available():
                inp, out = inp.cuda(), out.cuda()
            # ===================forward=====================
            # print(inp.dtype)
            # print(inp.shape)
            output = model(inp)
            loss = criterion(output, out)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        total_loss += loss.data
        print('epoch [{}/{}], loss:{:.4f}'
            .format(epoch+1, num_epochs, total_loss))
        # if epoch % 10 == 0:
        print("Image written")
        pic = to_img(output.cpu().data)
        save_image(pic, './{}/image_{}.png'.format(output_dir, epoch))
        torch.save(model.state_dict(), './conv_autoencoder_expt8.pth')

