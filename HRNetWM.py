# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by lyl258@gs.zzu.edu.cn)
# ------------------------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
from dropblock import DropBlock2D

# HRNetMW is the HRNet8_SVD
class HRNetMW(nn.Module):
    def __init__(self, model):
        super(HRNetMW, self).__init__()
        self.model = model
        self.model.drop_block = DropBlock2D(block_size=7, drop_prob=0.1)

        self.last_inp_channels = self.model.last_inp_channels
        self.norm_layer = nn.BatchNorm2d

        self.gen_heatmap = nn.Sequential(
            nn.Conv2d(
                in_channels=self.last_inp_channels,
                out_channels=self.last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            self.norm_layer(self.last_inp_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=self.last_inp_channels,
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=0),
        )

    def inference(self, images, indexXs, indexYs):
        torch.set_printoptions(threshold=np.inf)

        miniimages = images
        indexX = indexXs
        indexY = indexYs

        images = miniimages.reshape((-1, 3, 32, 32))
        Feature = self.model(images)

        Feature = Feature.reshape((-1, self.last_inp_channels, 32, 32))
        heatmap = self.gen_heatmap(Feature)

        heatmap = heatmap.reshape((-1, 6, 32, 32))

        WK1 = heatmap[:, 0:1]
        WK2 = heatmap[:, 1:2]
        WK3 = heatmap[:, 2:3]
        MK1 = heatmap[:, 3:4]
        MK2 = heatmap[:, 4:5]
        MK3 = heatmap[:, 5:6]

        WK1 = nn.Softmax(dim=1)(WK1.reshape(WK1.shape[0], -1)).reshape(WK1.shape)
        WK1 = torch.squeeze(WK1, dim=1)

        WK2 = nn.Softmax(dim=1)(WK2.reshape(WK2.shape[0], -1)).reshape(WK2.shape)
        WK2 = torch.squeeze(WK2, dim=1)

        WK3 = nn.Softmax(dim=1)(WK3.reshape(WK3.shape[0], -1)).reshape(WK3.shape)
        WK3 = torch.squeeze(WK3, dim=1)

        MK1 = nn.Softmax(dim=1)(MK1.reshape(MK1.shape[0], -1)).reshape(MK1.shape)
        MK1 = torch.squeeze(MK1, dim=1)

        MK2 = nn.Softmax(dim=1)(MK2.reshape(MK2.shape[0], -1)).reshape(MK2.shape)
        MK2 = torch.squeeze(MK2, dim=1)

        MK3 = nn.Softmax(dim=1)(MK3.reshape(MK3.shape[0], -1)).reshape(MK3.shape)
        MK3 = torch.squeeze(MK3, dim=1)

        ax = (torch.squeeze(torch.matmul(WK1, indexX[:, :, 0:1]), dim=2).sum(dim=1)).unsqueeze(dim=1).unsqueeze(1)
        ay = (torch.squeeze(torch.matmul(indexY[:, 0:1, :], WK1), dim=1).sum(dim=1)).unsqueeze(dim=1).unsqueeze(1)

        bx = (torch.squeeze(torch.matmul(WK2, indexX[:, :, 1:2]), dim=2).sum(dim=1)).unsqueeze(dim=1).unsqueeze(1)
        by = (torch.squeeze(torch.matmul(indexY[:, 1:2, :], WK2), dim=1).sum(dim=1)).unsqueeze(dim=1).unsqueeze(1)

        cx = (torch.squeeze(torch.matmul(WK3, indexX[:, :, 2:3]), dim=2).sum(dim=1)).unsqueeze(dim=1).unsqueeze(1)
        cy = (torch.squeeze(torch.matmul(indexY[:, 2:3, :], WK3), dim=1).sum(dim=1)).unsqueeze(dim=1).unsqueeze(1)

        dx = (torch.squeeze(torch.matmul(MK1, indexX[:, :, 3:4]), dim=2).sum(dim=1)).unsqueeze(dim=1).unsqueeze(1)
        dy = (torch.squeeze(torch.matmul(indexY[:, 3:4, :], MK1), dim=1).sum(dim=1)).unsqueeze(dim=1).unsqueeze(1)

        ex = (torch.squeeze(torch.matmul(MK2, indexX[:, :, 4:5]), dim=2).sum(dim=1)).unsqueeze(dim=1).unsqueeze(1)
        ey = (torch.squeeze(torch.matmul(indexY[:, 4:5, :], MK2), dim=1).sum(dim=1)).unsqueeze(dim=1).unsqueeze(1)

        fx = (torch.squeeze(torch.matmul(MK3, indexX[:, :, 5:6]), dim=2).sum(dim=1)).unsqueeze(dim=1).unsqueeze(1)
        fy = (torch.squeeze(torch.matmul(indexY[:, 5:6, :], MK3), dim=1).sum(dim=1)).unsqueeze(dim=1).unsqueeze(1)

        waferX = torch.cat((ax, bx, cx), dim=1)
        waferY = torch.cat((ay, by, cy), dim=1)

        maskX = torch.cat((dx, ex, fx), dim=1)
        maskY = torch.cat((dy, ey, fy), dim=1)

        WaferCoor = torch.cat((waferX, waferY), dim=2)
        MaskCoor = torch.cat((maskX, maskY), dim=2)

        WaferCoor = WaferCoor - torch.unsqueeze(torch.mean(WaferCoor, dim=1), dim=1)
        MaskCoor = MaskCoor - torch.unsqueeze(torch.mean(MaskCoor, dim=1), dim=1)

        covariance_matrix = torch.matmul(WaferCoor.transpose(1, 2), MaskCoor)
        U, _, Vt = torch.linalg.svd(covariance_matrix)
        R = torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2))
        deg_prePre = torch.squeeze(torch.rad2deg(torch.arctan(R[:, 1:2, 0:1] / R[:, 0:1, 0:1])), dim=1)

        return deg_prePre, ax, ay, bx, by, cx, cy, dx, dy, ex, ey, fx, fy

    def forward(self, images, indexXs, indexYs, heatmap_flag=False):
        torch.set_printoptions(threshold=np.inf)

        # l1 = ["_preWK0", "_preWK1", "_preWK2", "_preWK3", "_preWK4",
        #       "_curWK0", "_curWK1", "_curWK2", "_curWK3", "_curWK4",
        #       "_preMK0", "_preMK1", "_preMK2", "_preMK3", "_preMK4",
        #       "_curMK0", "_curMK1", "_curMK2", "_curMK3", "_curMK4"]

        standard_deviationList1 = torch.tensor([], device=images.device)
        standard_deviationList2 = torch.tensor([], device=images.device)
        wwList = torch.tensor([], device=images.device)
        mmList = torch.tensor([], device=images.device)
        wmList = torch.tensor([], device=images.device)

        for mini_index in range(4):
            miniimages = images[:, 60 * mini_index:60 * (mini_index + 1)]
            indexX = indexXs[:, :, 20 * mini_index:20 * (mini_index + 1)]
            indexY = indexYs[:, 20 * mini_index:20 * (mini_index + 1), :]

            imagesW = miniimages[:, :30].reshape((-1, 3, 32, 32))
            imagesM = miniimages[:, 30:].reshape((-1, 3, 32, 32))

            imagesWM = torch.cat((imagesW, imagesM), dim=0)

            Feature = self.model(imagesWM)
            n = Feature.shape[0]

            waferFeature = Feature[:int(n / 2)].reshape((-1, self.last_inp_channels * 10, 32, 32))  # (bs,12*n,32,32)
            maskFeature = Feature[int(n / 2):].reshape((-1, self.last_inp_channels * 10, 32, 32))  # (bs,12*n,32,32)

            # wafer  k0 and  k2
            waferFeature1 = torch.cat((waferFeature[:, self.last_inp_channels * 0:self.last_inp_channels * 1],
                                       waferFeature[:, self.last_inp_channels * 2:self.last_inp_channels * 3],
                                       waferFeature[:, self.last_inp_channels * 5:self.last_inp_channels * 6],
                                       waferFeature[:, self.last_inp_channels * 7:self.last_inp_channels * 8]
                                       ), dim=1)
            waferFeature1 = waferFeature1.reshape((-1, self.last_inp_channels, 32, 32))
            heatmapW1 = self.gen_heatmap(waferFeature1)
            heatmapW1 = heatmapW1.reshape((-1, 4, 32, 32))

            # wafer  k1  k3  k4
            waferFeature2 = torch.cat((waferFeature[:, self.last_inp_channels * 1:self.last_inp_channels * 2],
                                       waferFeature[:, self.last_inp_channels * 3:self.last_inp_channels * 5],
                                       waferFeature[:, self.last_inp_channels * 6:self.last_inp_channels * 7],
                                       waferFeature[:, self.last_inp_channels * 8:self.last_inp_channels * 10]
                                       ), dim=1)
            waferFeature2 = waferFeature2.reshape((-1, self.last_inp_channels, 32, 32))
            heatmapW2 = self.gen_heatmap(waferFeature2)
            heatmapW2 = heatmapW2.reshape((-1, 6, 32, 32))

            # mask  k0 and  k2
            maskFeature1 = torch.cat((maskFeature[:, self.last_inp_channels * 0:self.last_inp_channels * 1],
                                      maskFeature[:, self.last_inp_channels * 2:self.last_inp_channels * 3],
                                      maskFeature[:, self.last_inp_channels * 5:self.last_inp_channels * 6],
                                      maskFeature[:, self.last_inp_channels * 7:self.last_inp_channels * 8]
                                      ), dim=1)
            maskFeature1 = maskFeature1.reshape((-1, self.last_inp_channels, 32, 32))
            heatmapM1 = self.gen_heatmap(maskFeature1)
            heatmapM1 = heatmapM1.reshape((-1, 4, 32, 32))

            # mask  k1  k3  k4
            maskFeature2 = torch.cat((maskFeature[:, self.last_inp_channels * 1:self.last_inp_channels * 2],
                                      maskFeature[:, self.last_inp_channels * 3:self.last_inp_channels * 5],
                                      maskFeature[:, self.last_inp_channels * 6:self.last_inp_channels * 7],
                                      maskFeature[:, self.last_inp_channels * 8:self.last_inp_channels * 10]
                                      ), dim=1)
            maskFeature2 = maskFeature2.reshape((-1, self.last_inp_channels, 32, 32))
            heatmapM2 = self.gen_heatmap(maskFeature2)
            heatmapM2 = heatmapM2.reshape((-1, 6, 32, 32))

            _preWK0 = heatmapW1[:, 0:1]
            _preWK2 = heatmapW1[:, 1:2]
            _curWK0 = heatmapW1[:, 2:3]
            _curWK2 = heatmapW1[:, 3:4]

            _preWK1 = heatmapW2[:, 0:1]
            _preWK3 = heatmapW2[:, 1:2]
            _preWK4 = heatmapW2[:, 2:3]
            _curWK1 = heatmapW2[:, 3:4]
            _curWK3 = heatmapW2[:, 4:5]
            _curWK4 = heatmapW2[:, 5:6]

            _preMK0 = heatmapM1[:, 0:1]
            _preMK2 = heatmapM1[:, 1:2]
            _curMK0 = heatmapM1[:, 2:3]
            _curMK2 = heatmapM1[:, 3:4]

            _preMK1 = heatmapM2[:, 0:1]
            _preMK3 = heatmapM2[:, 1:2]
            _preMK4 = heatmapM2[:, 2:3]
            _curMK1 = heatmapM2[:, 3:4]
            _curMK3 = heatmapM2[:, 4:5]
            _curMK4 = heatmapM2[:, 5:6]

            _preWK0 = nn.Softmax(dim=1)(_preWK0.reshape(_preWK0.shape[0], -1)).reshape(_preWK0.shape)
            _preWK0 = torch.squeeze(_preWK0, dim=1)

            _preWK1 = nn.Softmax(dim=1)(_preWK1.reshape(_preWK1.shape[0], -1)).reshape(_preWK1.shape)
            _preWK1 = torch.squeeze(_preWK1, dim=1)

            _preWK2 = nn.Softmax(dim=1)(_preWK2.reshape(_preWK2.shape[0], -1)).reshape(_preWK2.shape)
            _preWK2 = torch.squeeze(_preWK2, dim=1)

            _preWK3 = nn.Softmax(dim=1)(_preWK3.reshape(_preWK3.shape[0], -1)).reshape(_preWK3.shape)
            _preWK3 = torch.squeeze(_preWK3, dim=1)

            _preWK4 = nn.Softmax(dim=1)(_preWK4.reshape(_preWK4.shape[0], -1)).reshape(_preWK4.shape)
            _preWK4 = torch.squeeze(_preWK4, dim=1)

            _curWK0 = nn.Softmax(dim=1)(_curWK0.reshape(_curWK0.shape[0], -1)).reshape(_curWK0.shape)
            _curWK0 = torch.squeeze(_curWK0, dim=1)

            _curWK1 = nn.Softmax(dim=1)(_curWK1.reshape(_curWK1.shape[0], -1)).reshape(_curWK1.shape)
            _curWK1 = torch.squeeze(_curWK1, dim=1)

            _curWK2 = nn.Softmax(dim=1)(_curWK2.reshape(_curWK2.shape[0], -1)).reshape(_curWK2.shape)
            _curWK2 = torch.squeeze(_curWK2, dim=1)

            _curWK3 = nn.Softmax(dim=1)(_curWK3.reshape(_curWK3.shape[0], -1)).reshape(_curWK3.shape)
            _curWK3 = torch.squeeze(_curWK3, dim=1)

            _curWK4 = nn.Softmax(dim=1)(_curWK4.reshape(_curWK4.shape[0], -1)).reshape(_curWK4.shape)
            _curWK4 = torch.squeeze(_curWK4, dim=1)

            _preMK0 = nn.Softmax(dim=1)(_preMK0.reshape(_preMK0.shape[0], -1)).reshape(_preMK0.shape)
            _preMK0 = torch.squeeze(_preMK0, dim=1)

            _preMK1 = nn.Softmax(dim=1)(_preMK1.reshape(_preMK1.shape[0], -1)).reshape(_preMK1.shape)
            _preMK1 = torch.squeeze(_preMK1, dim=1)

            _preMK2 = nn.Softmax(dim=1)(_preMK2.reshape(_preMK2.shape[0], -1)).reshape(_preMK2.shape)
            _preMK2 = torch.squeeze(_preMK2, dim=1)

            _preMK3 = nn.Softmax(dim=1)(_preMK3.reshape(_preMK3.shape[0], -1)).reshape(_preMK3.shape)
            _preMK3 = torch.squeeze(_preMK3, dim=1)

            _preMK4 = nn.Softmax(dim=1)(_preMK4.reshape(_preMK4.shape[0], -1)).reshape(_preMK4.shape)
            _preMK4 = torch.squeeze(_preMK4, dim=1)

            _curMK0 = nn.Softmax(dim=1)(_curMK0.reshape(_curMK0.shape[0], -1)).reshape(_curMK0.shape)
            _curMK0 = torch.squeeze(_curMK0, dim=1)

            _curMK1 = nn.Softmax(dim=1)(_curMK1.reshape(_curMK1.shape[0], -1)).reshape(_curMK1.shape)
            _curMK1 = torch.squeeze(_curMK1, dim=1)

            _curMK2 = nn.Softmax(dim=1)(_curMK2.reshape(_curMK2.shape[0], -1)).reshape(_curMK2.shape)
            _curMK2 = torch.squeeze(_curMK2, dim=1)

            _curMK3 = nn.Softmax(dim=1)(_curMK3.reshape(_curMK3.shape[0], -1)).reshape(_curMK3.shape)
            _curMK3 = torch.squeeze(_curMK3, dim=1)

            _curMK4 = nn.Softmax(dim=1)(_curMK4.reshape(_curMK4.shape[0], -1)).reshape(_curMK4.shape)
            _curMK4 = torch.squeeze(_curMK4, dim=1)

            # coordinate compute unit work ############################################################

            ax = (torch.squeeze(torch.matmul(_preWK0, indexX[:, :, 0:1]), dim=2).sum(dim=1)).unsqueeze(dim=1).unsqueeze(
                1)
            ay = (torch.squeeze(torch.matmul(indexY[:, 0:1, :], _preWK0), dim=1).sum(dim=1)).unsqueeze(dim=1).unsqueeze(
                1)

            bx = (torch.squeeze(torch.matmul(_preWK1, indexX[:, :, 1:2]), dim=2).sum(dim=1)).unsqueeze(dim=1).unsqueeze(
                1)
            by = (torch.squeeze(torch.matmul(indexY[:, 1:2, :], _preWK1), dim=1).sum(dim=1)).unsqueeze(dim=1).unsqueeze(
                1)

            cx = (torch.squeeze(torch.matmul(_preWK2, indexX[:, :, 2:3]), dim=2).sum(dim=1)).unsqueeze(dim=1).unsqueeze(
                1)
            cy = (torch.squeeze(torch.matmul(indexY[:, 2:3, :], _preWK2), dim=1).sum(dim=1)).unsqueeze(dim=1).unsqueeze(
                1)

            b4x = (torch.squeeze(torch.matmul(_preWK3, indexX[:, :, 3:4]), dim=2).sum(dim=1)).unsqueeze(
                dim=1).unsqueeze(1)
            b4y = (torch.squeeze(torch.matmul(indexY[:, 3:4, :], _preWK3), dim=1).sum(dim=1)).unsqueeze(
                dim=1).unsqueeze(1)

            b5x = (torch.squeeze(torch.matmul(_preWK4, indexX[:, :, 4:5]), dim=2).sum(dim=1)).unsqueeze(
                dim=1).unsqueeze(1)
            b5y = (torch.squeeze(torch.matmul(indexY[:, 4:5, :], _preWK4), dim=1).sum(dim=1)).unsqueeze(
                dim=1).unsqueeze(1)

            dx = (torch.squeeze(torch.matmul(_curWK0, indexX[:, :, 5:6]), dim=2).sum(dim=1)).unsqueeze(dim=1).unsqueeze(
                1)
            dy = (torch.squeeze(torch.matmul(indexY[:, 5:6, :], _curWK0), dim=1).sum(dim=1)).unsqueeze(dim=1).unsqueeze(
                1)

            ex = (torch.squeeze(torch.matmul(_curWK1, indexX[:, :, 6:7]), dim=2).sum(dim=1)).unsqueeze(dim=1).unsqueeze(
                1)
            ey = (torch.squeeze(torch.matmul(indexY[:, 6:7, :], _curWK1), dim=1).sum(dim=1)).unsqueeze(dim=1).unsqueeze(
                1)

            fx = (torch.squeeze(torch.matmul(_curWK2, indexX[:, :, 7:8]), dim=2).sum(dim=1)).unsqueeze(dim=1).unsqueeze(
                1)
            fy = (torch.squeeze(torch.matmul(indexY[:, 7:8, :], _curWK2), dim=1).sum(dim=1)).unsqueeze(dim=1).unsqueeze(
                1)

            e4x = (torch.squeeze(torch.matmul(_curWK3, indexX[:, :, 8:9]), dim=2).sum(dim=1)).unsqueeze(
                dim=1).unsqueeze(1)
            e4y = (torch.squeeze(torch.matmul(indexY[:, 8:9, :], _curWK3), dim=1).sum(dim=1)).unsqueeze(
                dim=1).unsqueeze(1)

            e5x = (torch.squeeze(torch.matmul(_curWK4, indexX[:, :, 9:10]), dim=2).sum(dim=1)).unsqueeze(
                dim=1).unsqueeze(1)
            e5y = (torch.squeeze(torch.matmul(indexY[:, 9:10, :], _curWK4), dim=1).sum(dim=1)).unsqueeze(
                dim=1).unsqueeze(1)

            gx = (torch.squeeze(torch.matmul(_preMK0, indexX[:, :, 10:11]), dim=2).sum(dim=1)).unsqueeze(
                dim=1).unsqueeze(1)
            gy = (torch.squeeze(torch.matmul(indexY[:, 10:11, :], _preMK0), dim=1).sum(dim=1)).unsqueeze(
                dim=1).unsqueeze(1)

            hx = (torch.squeeze(torch.matmul(_preMK1, indexX[:, :, 11:12]), dim=2).sum(dim=1)).unsqueeze(
                dim=1).unsqueeze(1)
            hy = (torch.squeeze(torch.matmul(indexY[:, 11:12, :], _preMK1), dim=1).sum(dim=1)).unsqueeze(
                dim=1).unsqueeze(1)

            ix = (torch.squeeze(torch.matmul(_preMK2, indexX[:, :, 12:13]), dim=2).sum(dim=1)).unsqueeze(
                dim=1).unsqueeze(1)
            iy = (torch.squeeze(torch.matmul(indexY[:, 12:13, :], _preMK2), dim=1).sum(dim=1)).unsqueeze(
                dim=1).unsqueeze(1)

            h4x = (torch.squeeze(torch.matmul(_preMK3, indexX[:, :, 13:14]), dim=2).sum(dim=1)).unsqueeze(
                dim=1).unsqueeze(
                1)
            h4y = (torch.squeeze(torch.matmul(indexY[:, 13:14, :], _preMK3), dim=1).sum(dim=1)).unsqueeze(
                dim=1).unsqueeze(
                1)
            h5x = (torch.squeeze(torch.matmul(_preMK4, indexX[:, :, 14:15]), dim=2).sum(dim=1)).unsqueeze(
                dim=1).unsqueeze(
                1)
            h5y = (torch.squeeze(torch.matmul(indexY[:, 14:15, :], _preMK4), dim=1).sum(dim=1)).unsqueeze(
                dim=1).unsqueeze(
                1)

            jx = (torch.squeeze(torch.matmul(_curMK0, indexX[:, :, 15:16]), dim=2).sum(dim=1)).unsqueeze(
                dim=1).unsqueeze(1)
            jy = (torch.squeeze(torch.matmul(indexY[:, 15:16, :], _curMK0), dim=1).sum(dim=1)).unsqueeze(
                dim=1).unsqueeze(1)

            kx = (torch.squeeze(torch.matmul(_curMK1, indexX[:, :, 16:17]), dim=2).sum(dim=1)).unsqueeze(
                dim=1).unsqueeze(1)
            ky = (torch.squeeze(torch.matmul(indexY[:, 16:17, :], _curMK1), dim=1).sum(dim=1)).unsqueeze(
                dim=1).unsqueeze(1)

            lx = (torch.squeeze(torch.matmul(_curMK2, indexX[:, :, 17:18]), dim=2).sum(dim=1)).unsqueeze(
                dim=1).unsqueeze(1)
            ly = (torch.squeeze(torch.matmul(indexY[:, 17:18, :], _curMK2), dim=1).sum(dim=1)).unsqueeze(
                dim=1).unsqueeze(1)

            k4x = (torch.squeeze(torch.matmul(_curMK3, indexX[:, :, 18:19]), dim=2).sum(dim=1)).unsqueeze(
                dim=1).unsqueeze(
                1)
            k4y = (torch.squeeze(torch.matmul(indexY[:, 18:19, :], _curMK3), dim=1).sum(dim=1)).unsqueeze(
                dim=1).unsqueeze(
                1)
            k5x = (torch.squeeze(torch.matmul(_curMK4, indexX[:, :, 19:20]), dim=2).sum(dim=1)).unsqueeze(
                dim=1).unsqueeze(
                1)
            k5y = (torch.squeeze(torch.matmul(indexY[:, 19:20, :], _curMK4), dim=1).sum(dim=1)).unsqueeze(
                dim=1).unsqueeze(1)

            if heatmap_flag is True:
                return _preWK1[0], bx[0] - indexX[0][:, 1:2][0], by[0] - indexY[0][1:2, :][0], \
                    _preMK0[0], gx[0] - indexX[0][:, 10:11][0], gy[0] - indexY[0][10:11, :][0]

            preWx = torch.cat((ax, bx, cx, b4x, b5x), dim=1)
            preWy = torch.cat((ay, by, cy, b4y, b5y), dim=1)
            curWx = torch.cat((dx, ex, fx, e4x, e5x), dim=1)
            curWy = torch.cat((dy, ey, fy, e4y, e5y), dim=1)

            preMx = torch.cat((gx, hx, ix, h4x, h5x), dim=1)
            preMy = torch.cat((gy, hy, iy, h4y, h5y), dim=1)
            curMx = torch.cat((jx, kx, lx, k4x, k5x), dim=1)
            curMy = torch.cat((jy, ky, ly, k4y, k5y), dim=1)

            preW = torch.cat((preWx, preWy), dim=2)
            curW = torch.cat((curWx, curWy), dim=2)
            preM = torch.cat((preMx, preMy), dim=2)
            curM = torch.cat((curMx, curMy), dim=2)

            # angle prediction unit work#################################################################
            # Centralize each point set
            preW = preW - torch.unsqueeze(torch.mean(preW, dim=1), dim=1)
            curW = curW - torch.unsqueeze(torch.mean(curW, dim=1), dim=1)
            preM = preM - torch.unsqueeze(torch.mean(preM, dim=1), dim=1)
            curM = curM - torch.unsqueeze(torch.mean(curM, dim=1), dim=1)

            # wafer to wafer
            covariance_matrix = torch.matmul(curW.transpose(1, 2), preW)
            U, _, Vt = torch.linalg.svd(covariance_matrix)
            R = torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2))
            deg_preWafer = torch.squeeze(torch.rad2deg(torch.arctan(R[:, 1:2, 0:1] / R[:, 0:1, 0:1])), dim=1)

            # mask to mask
            covariance_matrix = torch.matmul(curM.transpose(1, 2), preM)
            U, _, Vt = torch.linalg.svd(covariance_matrix)
            R = torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2))
            deg_preMask = torch.squeeze(torch.rad2deg(torch.arctan(R[:, 1:2, 0:1] / R[:, 0:1, 0:1])), dim=1)

            # wafer to mask A
            covariance_matrix = torch.matmul(preW.transpose(1, 2), preM)
            U, _, Vt = torch.linalg.svd(covariance_matrix)
            R = torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2))
            deg_prePre = torch.squeeze(torch.rad2deg(torch.arctan(R[:, 1:2, 0:1] / R[:, 0:1, 0:1])), dim=1)

            # wafer to mask B
            covariance_matrix = torch.matmul(curW.transpose(1, 2), curM)
            U, _, Vt = torch.linalg.svd(covariance_matrix)
            R = torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2))
            deg_preCur = torch.squeeze(torch.rad2deg(torch.arctan(R[:, 1:2, 0:1] / R[:, 0:1, 0:1])), dim=1)

            # wwlist append new item
            wwList = torch.cat((wwList, deg_preWafer), dim=1)

            # mmlist append new item
            mmList = torch.cat((mmList, deg_preMask), dim=1)

            # wmlist append new item
            wmList = torch.cat((wmList, deg_preCur - deg_prePre), dim=1)

            # stdlist1 append new item
            standard_deviationList1 = torch.cat((standard_deviationList1, deg_preCur), dim=1)

            # stdlist2 append new item
            standard_deviationList2 = torch.cat((standard_deviationList2, deg_prePre), dim=1)

        # stdloss compute
        standard_deviationList1 = torch.mean(torch.std(standard_deviationList1, dim=1))
        standard_deviationList2 = torch.mean(torch.std(standard_deviationList2, dim=1))

        return wwList, mmList, wmList, \
            (standard_deviationList1 + standard_deviationList2) / 2



