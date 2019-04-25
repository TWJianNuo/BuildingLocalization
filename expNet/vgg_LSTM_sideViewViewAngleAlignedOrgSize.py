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
import json

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
    def __init__(self, seqNameSet, generalPrefix, datasetName):
        self.seqNameSet = seqNameSet
        self.generalPrefix = generalPrefix
        self.dataSetLoc = datasetName
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
        random.Random(30).shuffle(self.totPathRec)
        self.tranPortion = self.totPathRec[0:int(len(self.totPathRec) * 0.8)]
        self.testPortion = self.totPathRec[int(len(self.totPathRec) * 0.8):]


class VGGNet(VGG):
    def __init__(self, pretrained=True, model='vgg16', requires_grad=True, remove_fc=False, show_params=False):
        super().__init__(make_layers(cfg[model]))
        self.ranges = ranges[model]
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096)
        )
        if pretrained:
            # exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)
            self.init_cus()

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:  # delete redundant fully-connected layer params, can save memory
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def init_cus(self):
        pre_vgg = models.vgg16(pretrained=True)
        count = 0
        for idx, l in enumerate(list(self.features) + list(self.classifier)):
            if isinstance(l, nn.Conv2d) or isinstance(l, nn.Linear):
                if count == 0:
                    l.weight.data[:,0:3,:,:] = pre_vgg.features[count].weight.clone()
                else:
                    if count < 30:
                        l.weight.data = pre_vgg.features[count].weight.clone()
                    elif count >= 30:
                        l.weight.data = pre_vgg.classifier[count - 31].weight.clone()
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
    in_channels = 4
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    return nn.Sequential(*layers)



class vgg_LSTM(nn.Module):

    def __init__(self):
        super().__init__()
        self.n_class = 4
        self.maxDepth = 150
        self.lstmInput = 4096
        self.lstmHidden = 1024
        self.imageTransforms = transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406, 0], [0.229, 0.224, 0.225, 1])
        ])


        self.pretrained_net = VGGNet()
        self.lstm = nn.LSTM(self.lstmInput, self.lstmHidden)
        self.predictorNorm = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 3),
            nn.Sigmoid(),
        )
        # torch.Tensor([[8, 8, 6]])
        self.scale = nn.Parameter(torch.Tensor([[8, 8, 6]]))
        self.bias = nn.Parameter(torch.Tensor([[0, 0, 1]]))
        self.opt = optim.SGD(list(self.parameters()), lr=0.001)
        assert (len(list(self.parameters())) == len(list(self.pretrained_net.parameters())) + len(list(self.lstm.parameters())) + len(list(self.predictorNorm.parameters())) + len([self.scale]) + len([self.bias])), "Parameter not trained"
        self.cuda()
    def forward(self, x):
        lstmInputList = list()
        for i in range(x.shape[0]):
            output = self.pretrained_net(x[i,:,:,:,:])
            lstmInputList.append(output.unsqueeze(1))
        lstm_out, lstm_hState = self.lstm(torch.cat(lstmInputList, dim=1))
        predictNormed = self.predictorNorm(lstm_out[-1,:,:].unsqueeze(1))
        prediction = (predictNormed[:,0,:] - 0.5) * self.scale + self.bias
        return prediction
    def train_cus(self, bdList):
        imgDT_normalizedList = list()
        gtl = list()
        for buildingEntity in bdList:
            imgs = list(buildingEntity.bdDict_rgb.values())
            depth = list(buildingEntity.bdDict_depth.values())
            imgD = np.concatenate([np.stack(imgs, axis=0), np.expand_dims(np.stack(depth, axis=0), axis = 3)], axis=3)
            imgDT = torch.from_numpy(imgD).type(torch.float).permute(0,3,1,2) / 255
            imgDT_normalized = torch.zeros_like(imgDT).cuda()
            for k in range(imgDT.shape[0]):
                imgDT_normalized[k,:,:,:] = self.imageTransforms(imgDT[k,:,:,:])
            imgDT_normalizedList.append(imgDT_normalized.unsqueeze(0))
            gtl.append(torch.from_numpy(buildingEntity.bdComp.transition).unsqueeze(0))
            # Test for channel correctness
            # rgbImg = imgD[2, :, :, 0:3]
            # depthImg = imgD[2, :, :, 3]
            # Image.fromarray(rgbImg).show()
            # Image.fromarray(depthImg).show()
        imgDT_normalizedTot = torch.cat(imgDT_normalizedList, dim=0).cuda()
        gt = torch.cat(gtl, 0).cuda()
        imputImgOut = self.forward(imgDT_normalizedTot)
        loss = torch.sum((imputImgOut - gt).pow(2)) / len(bdList)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss

    def test_cus(self, bdList):
        imgDT_normalizedList = list()
        gtl = list()
        for buildingEntity in bdList:
            imgs = list(buildingEntity.bdDict_rgb.values())
            depth = list(buildingEntity.bdDict_depth.values())
            imgD = np.concatenate([np.stack(imgs, axis=0), np.expand_dims(np.stack(depth, axis=0), axis = 3)], axis=3)
            imgDT = torch.from_numpy(imgD).type(torch.float).permute(0,3,1,2) / 255
            imgDT_normalized = torch.zeros_like(imgDT).cuda()
            for k in range(imgDT.shape[0]):
                imgDT_normalized[k,:,:,:] = self.imageTransforms(imgDT[k,:,:,:])
            imgDT_normalizedList.append(imgDT_normalized.unsqueeze(0))
            gtl.append(torch.from_numpy(buildingEntity.bdComp.transition).unsqueeze(0))
            # Test for channel correctness
            # rgbImg = imgD[2, :, :, 0:3]
            # depthImg = imgD[2, :, :, 3]
            # Image.fromarray(rgbImg).show()
            # Image.fromarray(depthImg).show()
        imgDT_normalizedTot = torch.cat(imgDT_normalizedList, dim=0).cuda()
        gt = torch.cat(gtl, 0).cuda()
        imputImgOut = self.forward(imgDT_normalizedTot)
        loss = torch.sum(torch.abs(imputImgOut - gt)) / len(bdList) / 3
        return loss
    def save(self, path):
        torch.save(self.state_dict(), path)
    def load(self, path):
        self.load_state_dict(torch.load(path))
with open('jsonParam.json') as data_file:
    jsonParam = json.load(data_file)

# modelName = 'vgg_LSTM_sideViewViewAngleAligned'
fileName = os.path.basename(__file__)
fileNameComp = fileName.split('.')
modelName = fileNameComp[0]
print(modelName)
datasetName = jsonParam[modelName]
generalPrefix = jsonParam['prefixPath']
allSeq = jsonParam['allSeq']
batchSize = 8
pkr = pickleReader(allSeq, generalPrefix, datasetName)
fcn = vgg_LSTM()

testComp = pkr.testPortion
trainComp = pkr.tranPortion

writer = SummaryWriter(os.path.join(generalPrefix, 'runs/' + modelName + "2"))
for i in range(100000):
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
            a = 1
            writer.add_scalar('TestLoss', np.mean(tLossl), i)
    blist = list()
    for k in range(batchSize):
        ranint = random.randint(0, len(trainComp) - 1)
        curPath = trainComp[ranint]
        with open(curPath, "rb") as input:
            bdcomp = pickle.load(input)
            blist.append(bdcomp)
    lossVal = fcn.train_cus(blist)
    if lossVal is not None:
        print("%d th iteration, loss is %f" % (i, lossVal))
        writer.add_scalar('TrainLoss', lossVal, i)
    if i % 500 == 499:
        rootPath = os.path.join(generalPrefix, 'svModel')
        dirPath = os.path.join(generalPrefix, 'svModel', modelName)
        try:
            os.mkdir(rootPath)
        except OSError:
            a = 1
        try:
            os.mkdir(dirPath)
        except OSError:
            a = 1
        fcn.save(os.path.join(dirPath, str(i)))