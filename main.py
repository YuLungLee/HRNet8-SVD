import os
import cv2
from tensorboardX import SummaryWriter
from torch.backends import cudnn
from HRNetV2 import *
from HRNetWM import *
from MyDataset import *


def check_dir(root_dir=''):
    if os.path.exists("{}".format(root_dir)) is False and root_dir != "":
        os.mkdir("{}".format(root_dir))

    if os.path.exists('{}result_tensorboard'.format(root_dir)) is False:
        os.mkdir('{}result_tensorboard'.format(root_dir))
    if os.path.exists("{}heatmap".format(root_dir)) is False:
        os.mkdir("{}heatmap".format(root_dir))

    if os.path.exists("{}heatmap2".format(root_dir)) is False:
        os.mkdir("{}heatmap2".format(root_dir))

    if os.path.exists("{}heatmap/train".format(root_dir)) is False:
        os.mkdir("{}heatmap/train".format(root_dir))
    if os.path.exists("{}heatmap/valid".format(root_dir)) is False:
        os.mkdir("{}heatmap/valid".format(root_dir))

    if os.path.exists("{}heatmap2/train".format(root_dir)) is False:
        os.mkdir("{}heatmap2/train".format(root_dir))
    if os.path.exists("{}heatmap2/valid".format(root_dir)) is False:
        os.mkdir("{}heatmap2/valid".format(root_dir))


def train(end_epoch=150):
    gpuid = "0"
    root_dir = "HRNet8_SVD/"
    check_dir(root_dir)
    cudnn.benchmark = True
    cudnn.deterministic = True
    cudnn.enabled = True
    device = torch.device("cuda:{}".format(gpuid) if torch.cuda.is_available() else "cpu")
    torch.set_printoptions(precision=10)

    dataset_train = MyDataset(root="DataSet", itemtype="Train")
    dataset_valid = MyDataset(root="DataSet", itemtype="Valid")

    data_loader_train = DataLoader(dataset=dataset_train, batch_size=8, shuffle=True)
    data_loader_valid = DataLoader(dataset=dataset_valid, batch_size=64, shuffle=False)

    model = HRNetMW(hrnet8(pretrained=False))  # HRNetMW is the HRNet8_SVD

    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001, nesterov=False)

    RESUME = True
    try:
        assert (RESUME is True)
        checkpoint = torch.load("{}checkpoint.pth".format(root_dir))
        model.load_state_dict(checkpoint['model'])  # load checkpoint
        optimizer.load_state_dict(checkpoint['optimizer'])  # load optimizer

        start_epoch = checkpoint['epoch'] + 1  # set epoch num
        train_loss = checkpoint["train_loss"]
        train_wwloss = checkpoint["train_wwloss"]
        train_mmloss = checkpoint["train_mmloss"]
        train_wmloss = checkpoint["train_wmloss"]
        train_stdloss = checkpoint["train_stdloss"]
        valid_loss = checkpoint["valid_loss"]
        valid_wwloss = checkpoint["valid_wwloss"]
        valid_mmloss = checkpoint["valid_mmloss"]
        valid_wmloss = checkpoint["valid_wmloss"]
        valid_stdloss = checkpoint["valid_stdloss"]
        best_epoch = checkpoint["best_epoch"]
        best_model = checkpoint["best_model"]

        # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001, nesterov=False)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.003162277659685048, momentum=0.9, weight_decay=0.0001,
                                    nesterov=False)
        lrScheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                           [1, 2, 3, 4, 5],
                                                           gamma=0.7943282347,
                                                           last_epoch=-1)
        lrScheduler.step(5)
        print(
            "continue training\noptimizer param：{}\nstart epoch：{}\nbest epoch：{}\n".format
            (optimizer, start_epoch, best_epoch))

        print('################'.center(15), end="")
        for epoch_num in range(1, checkpoint['epoch'] + 1):
            print("##########".center(10), end="")
        print()

        print('epoch'.center(15), end="")
        print('|'.center(1), end="")
        for epoch_num in range(1, checkpoint['epoch'] + 1):
            print(str(epoch_num).center(10), end="")
        print()

        print('################'.center(15), end="")
        for epoch_num in range(1, checkpoint['epoch'] + 1):
            print("##########".center(10), end="")
        print()

        print('train loss'.center(15), end="")
        print('|'.center(1), end="")
        for epoch_num in range(1, checkpoint['epoch'] + 1):
            print("{:.6f}".format(train_loss[epoch_num - 1]).center(10), end="")
        print()

        print('train wwloss'.center(15), end="")
        print('|'.center(1), end="")
        for epoch_num in range(1, checkpoint['epoch'] + 1):
            print("{:.6f}".format(train_wwloss[epoch_num - 1]).center(10), end="")
        print()

        print('train mmloss'.center(15), end="")
        print('|'.center(1), end="")
        for epoch_num in range(1, checkpoint['epoch'] + 1):
            print("{:.6f}".format(train_mmloss[epoch_num - 1]).center(10), end="")
        print()

        print('train wmloss'.center(15), end="")
        print('|'.center(1), end="")
        for epoch_num in range(1, checkpoint['epoch'] + 1):
            print("{:.6f}".format(train_wmloss[epoch_num - 1]).center(10), end="")
        print()

        print('train stdloss'.center(15), end="")
        print('|'.center(1), end="")
        for epoch_num in range(1, checkpoint['epoch'] + 1):
            print("{:.6f}".format(train_stdloss[epoch_num - 1]).center(10), end="")
        print()

        print('---------------'.center(15), end="")
        for epoch_num in range(1, checkpoint['epoch'] + 1):
            print("----------".center(10), end="")
        print()

        print('valid loss'.center(15), end="")
        print('|'.center(1), end="")
        for epoch_num in range(1, checkpoint['epoch'] + 1):
            print("{:.6f}".format(valid_loss[epoch_num - 1]).center(10), end="")
        print()

        print('valid wwloss'.center(15), end="")
        print('|'.center(1), end="")
        for epoch_num in range(1, checkpoint['epoch'] + 1):
            print("{:.6f}".format(valid_wwloss[epoch_num - 1]).center(10), end="")
        print()

        print('valid mmloss'.center(15), end="")
        print('|'.center(1), end="")
        for epoch_num in range(1, checkpoint['epoch'] + 1):
            print("{:.6f}".format(valid_mmloss[epoch_num - 1]).center(10), end="")
        print()

        print('valid wmloss'.center(15), end="")
        print('|'.center(1), end="")
        for epoch_num in range(1, checkpoint['epoch'] + 1):
            print("{:.6f}".format(valid_wmloss[epoch_num - 1]).center(10), end="")
        print()

        print('valid stdloss'.center(15), end="")
        print('|'.center(1), end="")
        for epoch_num in range(1, checkpoint['epoch'] + 1):
            print("{:.6f}".format(valid_stdloss[epoch_num - 1]).center(10), end="")
        print()

        print('################'.center(15), end="")
        for epoch_num in range(1, checkpoint['epoch'] + 1):
            print("##########".center(10), end="")
        print()

    except Exception:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001, nesterov=False)
        lrScheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                           [1, 2, 3, 4, 5],
                                                           gamma=1.5848931925,
                                                           last_epoch=-1)  # 0.7943282347

        start_epoch = 1
        train_loss = []
        train_wwloss = []
        train_mmloss = []
        train_wmloss = []
        train_stdloss = []
        valid_loss = []
        valid_wwloss = []
        valid_mmloss = []
        valid_wmloss = []
        valid_stdloss = []
        best_epoch = 0
        best_model = model.state_dict()
        print(
            "traing a new model\noptimizer paraml：{}\nstart epoch：{}\ntrain loss：{}\nvalid loss:{}\nbest epoch：{}\n".format
            (optimizer, start_epoch, train_loss, valid_loss, best_epoch))

    assert (end_epoch >= start_epoch)
    assert (len(train_loss) == len(valid_loss))
    assert (len(train_loss) + 1 == start_epoch)

    writer = SummaryWriter('{}result_tensorboard'.format(root_dir))

    for epoch_index in range(start_epoch, end_epoch + 1):
        # training ############################################################
        model.train()
        model.model.__setattr__("Istrain", True)
        torch.set_grad_enabled(True)
        this_train_loss = 0
        this_train_wwloss = 0
        this_train_mmloss = 0
        this_train_wmloss = 0
        this_train_stdloss = 0
        print("epoch:{}".format(epoch_index))
        for i, data in enumerate(data_loader_train):
            images, indexX, indexY, labels, labels_w = data

            images = images.to(device)
            indexX = indexX.to(device)
            indexY = indexY.to(device)
            labels = labels.to(device)
            labels_w = labels_w.to(device)

            deg_pre_ww, deg_pre_mm, deg_pre_wm, std_dev = model(images, indexX, indexY)

            loss_ww = torch.mean(torch.abs(deg_pre_ww - labels_w))
            loss_mm = torch.mean(torch.abs(deg_pre_mm - (labels_w - labels)))
            loss_wm = torch.mean(torch.abs(deg_pre_wm - labels))

            loss = loss_mm + loss_ww + loss_wm + std_dev

            print("loss:{:.6f}, wwloss:{:.6f}, mmloss:{:.6f}, wmloss:{:.6f}, stdloss:{:.6f}".
                  format(loss.data,
                         loss_ww.data,
                         loss_mm.data,
                         loss_wm.data,
                         std_dev.data
                         ))
            this_train_loss = this_train_loss + float(loss.data * labels.shape[0])
            this_train_wwloss = this_train_wwloss + float(loss_ww.data * labels.shape[0])
            this_train_mmloss = this_train_mmloss + float(loss_mm.data * labels.shape[0])
            this_train_wmloss = this_train_wmloss + float(loss_wm.data * labels.shape[0])
            this_train_stdloss = this_train_stdloss + float(std_dev.data * labels.shape[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        lrScheduler.step()

        this_train_loss = this_train_loss / dataset_train.__len__()
        this_train_wwloss = this_train_wwloss / dataset_train.__len__()
        this_train_mmloss = this_train_mmloss / dataset_train.__len__()
        this_train_wmloss = this_train_wmloss / dataset_train.__len__()
        this_train_stdloss = this_train_stdloss / dataset_train.__len__()
        train_loss.append(this_train_loss)
        train_wwloss.append(this_train_wwloss)
        train_mmloss.append(this_train_mmloss)
        train_wmloss.append(this_train_wmloss)
        train_stdloss.append(this_train_stdloss)

        # validiation #########################################################
        model.eval()
        model.model.__setattr__("Istrain", False)
        torch.set_grad_enabled(False)
        this_valid_loss = 0
        this_valid_wwloss = 0
        this_valid_mmloss = 0
        this_valid_wmloss = 0
        this_valid_stdloss = 0
        for i, data in enumerate(data_loader_valid):
            images, indexX, indexY, labels, labels_w = data
            images = images.to(device)
            indexX = indexX.to(device)
            indexY = indexY.to(device)
            labels = labels.to(device)
            labels_w = labels_w.to(device)

            deg_pre_ww, deg_pre_mm, deg_pre_wm, std_dev = model(images, indexX, indexY)

            loss_ww = torch.mean(torch.abs(deg_pre_ww - labels_w))
            loss_mm = torch.mean(torch.abs(deg_pre_mm - (labels_w - labels)))
            loss_wm = torch.mean(torch.abs(deg_pre_wm - labels))

            loss = loss_mm + loss_ww + loss_wm + std_dev

            this_valid_loss = this_valid_loss + float(loss.data * labels.shape[0])
            this_valid_wwloss = this_valid_wwloss + float(loss_ww.data * labels.shape[0])
            this_valid_mmloss = this_valid_mmloss + float(loss_mm.data * labels.shape[0])
            this_valid_wmloss = this_valid_wmloss + float(loss_wm.data * labels.shape[0])
            this_valid_stdloss = this_valid_stdloss + float(std_dev.data * labels.shape[0])

        this_valid_loss = this_valid_loss / dataset_valid.__len__()
        this_valid_wwloss = this_valid_wwloss / dataset_valid.__len__()
        this_valid_mmloss = this_valid_mmloss / dataset_valid.__len__()
        this_valid_wmloss = this_valid_wmloss / dataset_valid.__len__()
        this_valid_stdloss = this_valid_stdloss / dataset_valid.__len__()
        if len(valid_loss) > 0 and this_valid_loss < min(valid_loss):
            best_epoch = epoch_index
            best_model = model.state_dict()
        valid_loss.append(this_valid_loss)
        valid_wwloss.append(this_valid_wwloss)
        valid_mmloss.append(this_valid_mmloss)
        valid_wmloss.append(this_valid_wmloss)
        valid_stdloss.append(this_valid_stdloss)

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch_index,
            "train_loss": train_loss,
            "train_wwloss": train_wwloss,
            "train_mmloss": train_mmloss,
            "train_wmloss": train_wmloss,
            "train_stdloss": train_stdloss,
            "valid_loss": valid_loss,
            "valid_wwloss": valid_wwloss,
            "valid_mmloss": valid_mmloss,
            "valid_wmloss": valid_wmloss,
            "valid_stdloss": valid_stdloss,
            "best_epoch": best_epoch,
            "best_model": best_model
        }
        torch.save(checkpoint, "{}checkpoint.pth".format(root_dir))

        exampleImageindex = [12, 99]
        dirtype = ['valid', 'train']
        for imgtype in dirtype:
            for index in exampleImageindex:
                if imgtype == 'train':
                    images, indexX, indexY, _, _ = dataset_train.__getitem__(index)
                    img255 = dataset_train.getimg(index)
                else:
                    images, indexX, indexY, _, _ = dataset_valid.__getitem__(index)
                    img255 = dataset_valid.getimg(index)
                maskimg = img255[30:33].cpu().numpy().astype(np.uint8)
                maskimg = np.transpose(maskimg, [1, 2, 0])[:, :, [2, 1, 0]]
                waferimg = img255[3:6].cpu().numpy().astype(np.uint8)
                waferimg = np.transpose(waferimg, [1, 2, 0])[:, :, [2, 1, 0]]
                images = torch.unsqueeze(images, dim=0).to(device)
                indexX = torch.unsqueeze(indexX, dim=0).to(device)
                indexY = torch.unsqueeze(indexY, dim=0).to(device)
                waferheatmap, wx, wy, maskheatmap, mx, my = model.forward(images, indexX, indexY, heatmap_flag=True)
                wx = wx.cpu().numpy()[0][0]
                wy = wy.cpu().numpy()[0][0]
                mx = mx.cpu().numpy()[0][0]
                my = my.cpu().numpy()[0][0]
                waferheatmap = waferheatmap[:32]
                maskheatmap = maskheatmap[:32]

                maskheatmap = maskheatmap - torch.min(maskheatmap)
                maskheatmap = (maskheatmap / torch.max(maskheatmap) * 255).cpu().numpy().astype(np.uint8)
                maskheatmap = cv2.applyColorMap(maskheatmap, cv2.COLORMAP_JET)
                maskheatmap = cv2.addWeighted(maskheatmap, 0.15, maskimg, 1, 0)
                cv2.imwrite("{}heatmap2/{}/mask{}_epoch{}_heatmap.png".format(root_dir, imgtype, index, epoch_index),
                            maskheatmap)
                cv2.circle(maskheatmap, (int(mx + 0.5), int(my + 0.5)), 1, (0, 255, 0))
                cv2.imwrite("{}heatmap/{}/mask{}_epoch{}_heatmap.png".format(root_dir, imgtype, index, epoch_index),
                            maskheatmap)
                cv2.imwrite("{}heatmap/{}/mask{}_epoch{}_heatmap.png".format(root_dir, imgtype, index, 0), maskimg)

                waferheatmap = waferheatmap - torch.min(waferheatmap)
                waferheatmap = (waferheatmap / torch.max(waferheatmap) * 255).cpu().numpy().astype(np.uint8)
                waferheatmap = cv2.applyColorMap(waferheatmap, cv2.COLORMAP_JET)
                waferheatmap = cv2.addWeighted(waferheatmap, 0.15, waferimg, 1, 0)
                cv2.imwrite("{}heatmap2/{}/wafer{}_epoch{}_heatmap.png".format(root_dir, imgtype, index, epoch_index),
                            waferheatmap)
                cv2.circle(waferheatmap, (int(wx + 0.5), int(wy + 0.5)), 1, (0, 255, 0))
                cv2.imwrite("{}heatmap/{}/wafer{}_epoch{}_heatmap.png".format(root_dir, imgtype, index, epoch_index),
                            waferheatmap)
                cv2.imwrite("{}heatmap/{}/wafer{}_epoch{}_heatmap.png".format(root_dir, imgtype, index, 0), waferimg)

        print(
            "train loss:{:.6f}, train loss_ww:{:.6f}, train loss_mm:{:.6f}, train loss_wm:{:.6f}, train dev_std:{:.6f}"
            .format(this_train_loss, this_train_wwloss, this_train_mmloss, this_train_wmloss, this_train_stdloss))
        print(
            "valid loss:{:.6f}, valid loss_ww:{:.6f}, valid loss_mm:{:.6f}, valid loss_wm:{:.6f}, valid dev_std:{:.6f}"
            .format(this_valid_loss, this_valid_wwloss, this_valid_mmloss, this_valid_wmloss, this_valid_stdloss))

        writer.add_scalars("loss", {
            "train": this_train_loss,
            "valid": this_valid_loss
        }, epoch_index)
        writer.add_scalars("wwloss", {
            "train": this_train_wwloss,
            "valid": this_valid_wwloss
        }, epoch_index)
        writer.add_scalars("mmloss", {
            "train": this_train_mmloss,
            "valid": this_valid_mmloss
        }, epoch_index)

        writer.add_scalars("wmloss", {
            "train": this_train_wmloss,
            "valid": this_valid_wmloss
        }, epoch_index)

        writer.add_scalars("stdloss", {
            "train": this_train_stdloss,
            "valid": this_valid_stdloss
        }, epoch_index)

        writer.close()


if __name__ == "__main__":
    pass
    train()
