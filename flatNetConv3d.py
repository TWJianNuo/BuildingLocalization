from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models.vgg import VGG
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.optim as optim
import os
import time
import pickle
import glob
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class singleBuildingComp:
    def __init__(self, bdComp, sequenceName, bdDict_cct, bdDict_M, bdDict_rgb, bdDict_depth, bdDict_imgSize,
                 bdDict_imgPath, bdDict_new2oldAffs, bdDict_xDiff, bdDict_yDiff, bdDict_dDiff, bdDict_toEstM,
                 bdDict_gtAff, bdDict_selector, bdDict_mskRecMap, bdDict_planeFlatIndRec, bdDict_xidRecMap, bdDict_yidRecMap):
        self.bdComp = bdComp
        self.sequenceName = sequenceName
        self.bdDict_cct = bdDict_cct
        self.bdDict_M = bdDict_M
        self.bdDict_rgb = bdDict_rgb
        self.bdDict_depth = bdDict_depth
        self.bdDict_imgSize = bdDict_imgSize
        self.bdDict_imgPath = bdDict_imgPath
        self.bdDict_new2oldAffs = bdDict_new2oldAffs
        self.bdDict_xDiff = bdDict_xDiff
        self.bdDict_yDiff = bdDict_yDiff
        self.bdDict_dDiff = bdDict_dDiff
        self.bdDict_toEstM = bdDict_toEstM
        self.bdDict_gtAff = bdDict_gtAff
        self.bdDict_selector = bdDict_selector
        self.bdDict_mskRecMap = bdDict_mskRecMap
        self.bdDict_planeFlatIndRec = bdDict_planeFlatIndRec
        self.bdDict_xidRecMap = bdDict_xidRecMap
        self.bdDict_yidRecMap = bdDict_yidRecMap


class pickleReader():
    def __init__(self, seqNameSet, generalPrefix):
        self.seqNameSet = seqNameSet
        self.generalPrefix = generalPrefix
        self.dataSetLoc = 'trainData_flat_test'
        self.rootPath = os.path.join(self.generalPrefix, self.dataSetLoc)
        self.bdEntry = dict()
        self.totPathRec = list()
        for seq in seqNameSet:
            tmpPath = os.path.join(self.rootPath, seq)
            bdNames = glob.glob(os.path.join(tmpPath, "*.p"))
            self.bdEntry[seq] = bdNames
            for name in bdNames:
                self.totPathRec.append(name)
        # Split to 80% as training and 20% as test
        random.shuffle(self.totPathRec)
        self.tranPortion = self.totPathRec[0:int(len(self.totPathRec) * 0.8)]
        self.testPortion = self.totPathRec[int(len(self.totPathRec) * 0.8):]


class VGGNet(VGG):
    def __init__(self, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True, show_params=False):
        super().__init__(make_layers(cfg[model]))
        self.ranges = ranges[model]

        if pretrained:
            self.init_cus()
            # exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:  # delete redundant fully-connected layer params, can save memory
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}

        # get the output of each maxpooling layer (5 maxpool in VGG net)
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output["x%d"%(idx+1)] = x

        return output
    def init_cus(self):
        pre_vgg = models.vgg16(pretrained=True)
        count = 0
        for idx, l in enumerate(list(self.features)):
            if isinstance(l, nn.Conv3d):
                for i in range(3):
                    l.weight.data[:,:,i,:,:] = pre_vgg.features[count].weight.clone()
            count = count + 1

ranges = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

# cropped version from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    MCount = 0
    for v in cfg:
        if v == 'M':
            if MCount == 1 or MCount == 0:
                layers += [nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))]
            else:
                layers += [nn.MaxPool3d(kernel_size=2, stride=2)]
            MCount = MCount + 1
        else:
            conv3d = nn.Conv3d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv3d, nn.BatchNorm3d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv3d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)



class FCNs(nn.Module):

    def __init__(self):
        super().__init__()
        self.n_class = 4
        self.maxDepth = 150


        self.pretrained_net = VGGNet()
        """
        pre_vgg = models.vgg16(pretrained=True)
        self.vggPretrain = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)),

            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        """

        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose3d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm3d(512)
        self.deconv2 = nn.ConvTranspose3d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm3d(256)
        self.deconv3 = nn.ConvTranspose3d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm3d(128)
        self.deconv4 = nn.ConvTranspose3d(128, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), output_padding=(0, 1, 1))
        self.bn4     = nn.BatchNorm3d(64)
        self.deconv5 = nn.ConvTranspose3d(64, 32, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), output_padding=(0, 1, 1))
        self.bn5     = nn.BatchNorm3d(32)
        self.classifier = nn.Conv3d(32, self.n_class, kernel_size=1)

        self.imageTransforms = transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406, 0], [0.229, 0.224, 0.225, 1])
        ])
        self.opt = optim.SGD(list(self.parameters()), lr=0.00001)
        self.sfax = torch.nn.Softmax(dim = 1)
        self.selectedFeatureNum = 1024
        self.cuda()
    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)
        x2 = output['x2']  # size=(N, 128, x.H/4,  x.W/4)
        x1 = output['x1']  # size=(N, 64, x.H/2,  x.W/2)

        score = self.bn1(self.relu(self.deconv1(x5)))     # size=(N, 512, x.H/16, x.W/16)
        score = score + x4                                # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = score + x3                                # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = score + x2                                # element-wise add, size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = score + x1                                # element-wise add, size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)  # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)
    def train_cus(self, bdList):
        loss = torch.Tensor([0])[0].cuda()
        for buildingEntity in bdList:
            imSizeNp = buildingEntity.bdDict_imgSize
            imSize = [imSizeNp[0].item(), imSizeNp[1].item()]
            imSizeInterpNp = np.power(2, np.ceil(np.log2(imSizeNp))).astype(np.intc)
            imSizeInterp = [imSizeInterpNp[0].item(), imSizeInterpNp[1].item()]
            if imSizeInterp[0] < 32:
                imSizeInterp[0] = 32
            if imSizeInterp[1] < 32:
                imSizeInterp[1] = 32
            imgs = list(buildingEntity.bdDict_rgb.values())
            gtValxs = list(buildingEntity.bdDict_xDiff.values())
            gtValys = list(buildingEntity.bdDict_yDiff.values())
            gtValds = list(buildingEntity.bdDict_dDiff.values())
            msks = list(buildingEntity.bdDict_selector.values())

            gtdatasNp = [np.stack(gtValxs, axis=0), np.stack(gtValys, axis=0), np.stack(gtValds, axis=0)]
            gtdataNp = np.stack(gtdatasNp, axis = 1)
            # msksS = np.repeat(np.expand_dims(np.stack(msks, axis=0), axis=1), 3, axis = 1).astype(np.float32)
            msksS = np.expand_dims(np.stack(msks, axis=0), axis=1).astype(np.float32)
            imgsCon = np.concatenate(imgs, axis=2)
            imgsCon_chanRearr = np.stack([imgsCon[:,:,0::3], imgsCon[:,:,1::3], imgsCon[:,:,2::3]], axis=3)
            # Check for image Content
            # imgSample = Image.fromarray((imgs[0] * 255).astype(dtype = np.uint8))
            # imgSample.show()
            # imgSample = Image.fromarray((imgsCon[:,:,0:3] * 255).astype(dtype = np.uint8))
            # imgSample.show()


            imgsS = torch.from_numpy(imgsCon_chanRearr).permute(2, 3, 0, 1).cuda()
            # Check for image content
            # imgSample = Image.fromarray((imgsS[0,:,:,:].permute(1,2,0).numpy() * 255).astype(dtype = np.uint8))
            # imgSample.show()
            gtdata = torch.from_numpy(gtdataNp).cuda()
            msksST = torch.from_numpy(msksS).cuda()

            imgsS_rz = torch.nn.functional.interpolate(imgsS, size=imSizeInterp, mode='bilinear', align_corners=False).unsqueeze(0).permute(0,2,1,3,4)
            # a = nn.Conv3d(3, 64, kernel_size=3, padding=1)
            # a = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))
            # b = a(imgsS_rz)
            # b = a(imgsS_rz)
            # a = self.vggPretrain(imgsS_rz)
            # a = self.testNet(imgsS_rz)
            imputImgOut = self.forward(imgsS_rz).squeeze(0).permute(1,0,2,3)
            imputImgOut_rz = torch.nn.functional.interpolate(imputImgOut, size=imSize, mode='bilinear',
                                                             align_corners=False)

            # Start to compute loss
            maxRe = self.sfax(imputImgOut_rz[:,3,:,:].reshape(1,-1)) * msksST.reshape(1,-1)
            reVal, idx = torch.topk(maxRe, self.selectedFeatureNum)
            for k in range(3):
                loss = loss + torch.sum((imputImgOut_rz[:,k,:,:].reshape(1,-1)[0,idx[0,:]] - gtdata[:,k,:,:].reshape(1,-1)[0,idx[0,:]]).pow(2))

        loss = loss / self.selectedFeatureNum / len(bdList)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss

    def test_cus(self, bdList):
        loss = torch.Tensor([0])[0].cuda()
        for buildingEntity in bdList:
            imSizeNp = buildingEntity.bdDict_imgSize
            imSize = [imSizeNp[0].item(), imSizeNp[1].item()]
            imSizeInterpNp = np.power(2, np.ceil(np.log2(imSizeNp))).astype(np.intc)
            imSizeInterp = [imSizeInterpNp[0].item(), imSizeInterpNp[1].item()]
            if imSizeInterp[0] < 32:
                imSizeInterp[0] = 32
            if imSizeInterp[1] < 32:
                imSizeInterp[1] = 32
            imgs = list(buildingEntity.bdDict_rgb.values())
            gtValxs = list(buildingEntity.bdDict_xDiff.values())
            gtValys = list(buildingEntity.bdDict_yDiff.values())
            gtValds = list(buildingEntity.bdDict_dDiff.values())
            msks = list(buildingEntity.bdDict_selector.values())

            gtdatasNp = [np.stack(gtValxs, axis=0), np.stack(gtValys, axis=0), np.stack(gtValds, axis=0)]
            gtdataNp = np.stack(gtdatasNp, axis = 1)
            # msksS = np.repeat(np.expand_dims(np.stack(msks, axis=0), axis=1), 3, axis = 1).astype(np.float32)
            msksS = np.expand_dims(np.stack(msks, axis=0), axis=1).astype(np.float32)
            imgsCon = np.concatenate(imgs, axis=2)
            imgsCon_chanRearr = np.stack([imgsCon[:,:,0::3], imgsCon[:,:,1::3], imgsCon[:,:,2::3]], axis=3)
            # Check for image Content
            # imgSample = Image.fromarray((imgs[0] * 255).astype(dtype = np.uint8))
            # imgSample.show()
            # imgSample = Image.fromarray((imgsCon[:,:,0:3] * 255).astype(dtype = np.uint8))
            # imgSample.show()


            imgsS = torch.from_numpy(imgsCon_chanRearr).permute(2, 3, 0, 1).cuda()
            # Check for image content
            # imgSample = Image.fromarray((imgsS[0,:,:,:].permute(1,2,0).numpy() * 255).astype(dtype = np.uint8))
            # imgSample.show()
            gtdata = torch.from_numpy(gtdataNp).cuda()
            msksST = torch.from_numpy(msksS).cuda()

            imgsS_rz = torch.nn.functional.interpolate(imgsS, size=imSizeInterp, mode='bilinear', align_corners=False).unsqueeze(0).permute(0,2,1,3,4)
            # a = nn.Conv3d(3, 64, kernel_size=3, padding=1)
            # a = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))
            # b = a(imgsS_rz)
            # b = a(imgsS_rz)
            # a = self.vggPretrain(imgsS_rz)
            # a = self.testNet(imgsS_rz)
            imputImgOut = self.forward(imgsS_rz).squeeze(0).permute(1,0,2,3)
            imputImgOut_rz = torch.nn.functional.interpolate(imputImgOut, size=imSize, mode='bilinear',
                                                             align_corners=False)

            # Start to compute loss
            maxRe = self.sfax(imputImgOut_rz[:,3,:,:].reshape(1,-1)) * msksST.reshape(1,-1)
            reVal, idx = torch.topk(maxRe, self.selectedFeatureNum)
            for k in range(3):
                loss = loss + torch.sum((imputImgOut_rz[:,k,:,:].reshape(1,-1)[0,idx[0,:]] - gtdata[:,k,:,:].reshape(1,-1)[0,idx[0,:]]).pow(2))

        loss = loss / self.selectedFeatureNum / len(bdList)
        return loss


allSeq = [
    '2011_09_30_drive_0018_sync',
    '2011_09_26_drive_0096_sync',
    '2011_09_26_drive_0104_sync',
    '2011_09_26_drive_0117_sync',
    '2011_09_30_drive_0033_sync',
    '2011_10_03_drive_0034_sync',
    '2011_10_03_drive_0027_sync',
    '2011_09_30_drive_0028_sync',
    '2011_09_26_drive_0019_sync',
    '2011_09_26_drive_0020_sync',
    '2011_09_26_drive_0022_sync',
    '2011_09_26_drive_0023_sync',
    '2011_09_26_drive_0035_sync',
    '2011_09_26_drive_0036_sync',
    '2011_09_26_drive_0039_sync',
    '2011_09_26_drive_0046_sync',
    '2011_09_26_drive_0061_sync',
    '2011_09_26_drive_0064_sync',
    '2011_09_26_drive_0079_sync',
    '2011_09_26_drive_0086_sync',
]
generalPrefix = '/media/shengjie/other/KITTI_scene_understanding/python_code/BuildingLocalization'
batchSize = 8
pkr = pickleReader(allSeq, generalPrefix)
fcn = FCNs()

testComp = pkr.testPortion
trainComp = pkr.tranPortion

writer = SummaryWriter(os.path.join(generalPrefix, 'runs/flatNetConv3d1'))
for i in range(100000):
    blist = list()
    for k in range(batchSize):
        ranint = random.randint(0, len(trainComp) - 1)
        curPath = trainComp[ranint]
        with open(curPath, "rb") as input:
            bdcomp = pickle.load(input)
            blist.append(bdcomp)
    lossVal = fcn.train_cus(blist)
    if lossVal is not None:
        print("%dth iteration, loss is %f" % (i, lossVal))
        writer.add_scalar('TrainLoss', lossVal, i)
    if i % 200 == 0:
        tLossl = list()
        for add in testComp:
            with open(add, "rb") as input:
                bdcomp = pickle.load(input)
                testVal = fcn.test_cus([bdcomp])
                if testVal is not None:
                    tLossl.append(testVal.cpu().detach().numpy())
        print("TestLoss is %f" % np.mean(tLossl))
        if np.mean(tLossl) > 0:
            writer.add_scalar('TestLoss', np.mean(tLossl), i)
