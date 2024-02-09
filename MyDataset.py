import random
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, root=None, itemtype="Train"):
        super(MyDataset, self).__init__()
        self.root = root
        self.txt_path = self.root + '/' + "DataSets{}.txt".format(itemtype)
        self.item_dict = dict()

        f = open(self.txt_path, 'r')
        data = f.readlines()

        imgs = []
        temp = []
        labels = []
        labels_w = []
        for i, line in enumerate(data):
            word = line.strip().split()

            labels.append(word[1])
            labels_w.append(word[2])

            id = word[0][word[0].rfind("\\"):].replace("\\", "/")

            key = id[:id.rfind("_")]
            try:
                self.item_dict[key].append(i)
            except Exception as f:
                self.item_dict[key] = []
                self.item_dict[key].append(i)

            id = word[0].replace("\\", "/") + id

            l1 = ["_preWK0", "_preWK1", "_preWK2", "_preWK3", "_preWK4",
                  "_curWK0", "_curWK1", "_curWK2", "_curWK3", "_curWK4",
                  "_preMK0", "_preMK1", "_preMK2", "_preMK3", "_preMK4",
                  "_curMK0", "_curMK1", "_curMK2", "_curMK3", "_curMK4"]
            for each in l1:
                temp.append("{}{}".format(id, each))
            imgs.append(temp)
            temp = []

        self.img = imgs
        self.label = labels
        self.label_w = labels_w

        self.tran = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        self.book = dict()
        for i, each in enumerate(self.item_dict):
            self.book[i] = each
            if len(self.item_dict[each]) == 3:
                self.item_dict[each].append(self.item_dict[each][random.randint(0, 2)])

    def __len__(self):
        return len(self.item_dict)

    def getimg(self, index):
        mini_index = self.item_dict[self.book[index]][0]
        imgs = self.img[mini_index]
        tensor_imgs = None
        for each in imgs:
            img = Image.open(self.root + "/" + each + ".png").convert('RGB')
            np_img = np.array(img, dtype=np.uint8)
            np_img2 = np.rollaxis(np_img, 2)
            tensor_img = torch.from_numpy(np_img2).to(dtype=torch.float)
            if tensor_imgs is None:
                tensor_imgs = tensor_img
            else:
                tensor_imgs = torch.cat((tensor_imgs, tensor_img), dim=0)
        return tensor_imgs

    def getminiitem(self, item):
        imgs = self.img[item]
        label = self.label[item]
        label_w = self.label_w[item]

        tensor_imgs = None
        for each in imgs:
            img = Image.open(self.root + "/" + each + ".png").convert('RGB')

            # ToNumpy
            np_img = np.array(img, dtype=np.uint8)
            tensor_img = self.tran(np_img)

            if tensor_imgs is None:
                tensor_imgs = tensor_img
            else:
                tensor_imgs = torch.cat((tensor_imgs, tensor_img), dim=0)

        indexX = torch.tensor([])
        indexY = torch.tensor([])
        for each in imgs:
            with open(self.root + "/" + each + ".txt") as f:
                word = f.readlines()

                xs = np.array(word[0].strip().split(), dtype=np.float32)
                ys = np.array(word[1].strip().split(), dtype=np.float32)

                tensor_indexX = torch.from_numpy(xs)
                tensor_indexX.resize_(32, 1)

                tensor_indexY = torch.from_numpy(ys)
                tensor_indexY.resize_(1, 32)

                indexX = torch.cat((indexX, tensor_indexX), dim=1)
                indexY = torch.cat((indexY, tensor_indexY), dim=0)

        label = np.array(float(label))
        label = torch.from_numpy(label)
        label_w = np.array(float(label_w))
        label_w = torch.from_numpy(label_w)

        return tensor_imgs, indexX, indexY, label, label_w

    def __getitem__(self, item):
        mini_index = self.item_dict[self.book[item]]

        imgs = torch.tensor([])
        indexXs = torch.tensor([])
        indexYs = torch.tensor([])

        for index in mini_index:
            img, indexX, indexY, label, label_w = self.getminiitem(index)
            imgs = torch.cat((imgs, img), dim=0)
            indexXs = torch.cat((indexXs, indexX), dim=1)
            indexYs = torch.cat((indexYs, indexY), dim=0)

        return imgs, indexXs, indexYs, label.reshape((1)), label_w.reshape((1))
