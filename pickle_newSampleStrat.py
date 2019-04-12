import json
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import os
import videoSequenceProvider
from PIL import Image
import torch.optim as optim
import os
import time
import pickle
import glob
import random

class singleBuildingComp:
    def __init__(self, bdComp, seqName, rgbs_dict, depths_dict, imgSizeDict, tr_grid2oxtsDict, imgPathDict,
                 tr_oxts2velo, extrinsic, intrinsic, sampledPts):
        self.bdComp = bdComp
        self.seqName = seqName
        self.rgbs_dict = rgbs_dict
        self.depths_dict = depths_dict
        self.sampledPts = sampledPts
        self.imgSizeDict = imgSizeDict
        self.tr_grid2oxtsDict = tr_grid2oxtsDict
        self.imgPathDict = imgPathDict
        self.tr_oxts2velo = tr_oxts2velo
        self.extrinsic = extrinsic
        self.intrinsic = intrinsic
class pickleReader():
    def __init__(self, seqNameSet):
        self.seqNameSet = seqNameSet
        self.rootPath = '/media/shengjie/other/KITTI_scene_understanding/python_code/BuildingLocalization/trainingData'
        self.bdEntry = dict()
        self.validIndices = dict()
        self.invalidIndices= dict()
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

        savedDataSplitFilePath = '/media/shengjie/other/KITTI_scene_understanding/python_code/BuildingLocalization/dataSplit'
        if os.path.isdir(savedDataSplitFilePath) == False:
            try:
                os.mkdir(savedDataSplitFilePath)
            except OSError:
                print("Creation of the directory %s failed" % savedDataSplitFilePath)
            else:
                print("Successfully created the directory %s " % savedDataSplitFilePath)
        if os.path.isfile('/media/shengjie/other/KITTI_scene_understanding/python_code/BuildingLocalization/dataSplit/dataSplit.p') == False:
            tmpdict = dict()
            tmpdict['train'] = self.tranPortion
            tmpdict['test'] = self.testPortion
            pickle.dump(tmpdict, open("/media/shengjie/other/KITTI_scene_understanding/python_code/BuildingLocalization/dataSplit/dataSplit.p", "wb"))
        else:
            tmpdict = pickle.load(open("/media/shengjie/other/KITTI_scene_understanding/python_code/BuildingLocalization/dataSplit/dataSplit.p", "rb"))
            self.tranPortion = tmpdict['train']
            self.testPortion = tmpdict['test']
            for idx in self.testPortion:
                print(idx)
class baselineModel:
    def __init__(self) -> object:
        super(baselineModel, self).__init__()
        self.pre_imageNet = models.alexnet(pretrained=True)
        new_convLayer = nn.Conv2d(4,64,kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        nn.init.xavier_uniform(new_convLayer.weight)
        new_convLayer.weight[:,0:3,:,:] = self.pre_imageNet.features[0].weight
        new_convLayer.weight = nn.Parameter(new_convLayer.weight)
        feature_layerList = list(self.pre_imageNet.features.children())
        feature_layerList[0] = new_convLayer
        self.pre_imageNet.features = nn.Sequential(*feature_layerList)

        classifier_layerList = list(self.pre_imageNet.classifier.children())[:-1]
        self.pre_imageNet.classifier = nn.Sequential(*classifier_layerList)
        self.pre_imageNet.cuda()

        self.lstm_input = 4096
        self.lstm_hidden = int(4096 / 4)
        self.LSTM = nn.LSTM(self.lstm_input, self.lstm_hidden)
        self.LSTM.cuda()
        self.paramPredictor = nn.Sequential(nn.Linear(self.lstm_hidden, self.lstm_hidden), nn.ReLU(), nn.Linear(self.lstm_hidden, 3), nn.Sigmoid()).cuda()
        self.visibilityPredictor = nn.Sequential(nn.Linear(self.lstm_hidden, self.lstm_hidden), nn.ReLU(), nn.Linear(self.lstm_hidden, 1)).cuda()

        self.optimizerTrans = optim.SGD(list(self.pre_imageNet.parameters()) + list(self.LSTM.parameters()) + list(self.paramPredictor.parameters()), lr=0.001)
        self.optimizerVisibility = optim.SGD(self.visibilityPredictor.parameters(), lr=0.001)
        self.svPath = '/media/shengjie/other/KITTI_scene_understanding/python_code/BuildingLocalization/svModel'
        self.maxLen = 10
    def sv(self, idt):
        torch.save({
            'pre_imageNet_state_dict': self.pre_imageNet.state_dict(),
            'LSTM_state_dict': self.LSTM.state_dict(),
            'paramPredictor_state_dict': self.paramPredictor.state_dict(),
            'visibilityPredictor_state_dict': self.visibilityPredictor.state_dict(),
        }, os.path.join(self.svPath, str(idt)))
    def initSv(self):
        torch.save({
            'pre_imageNet_state_dict': self.pre_imageNet.state_dict(),
            'LSTM_state_dict': self.LSTM.state_dict(),
            'paramPredictor_state_dict': self.paramPredictor.state_dict(),
            'visibilityPredictor_state_dict': self.visibilityPredictor.state_dict(),
        }, os.path.join('/media/shengjie/other/KITTI_scene_understanding/python_code/BuildingLocalization/initedModel', str(0)))

    def initLoad(self):
        checkpoint = torch.load(os.path.join('/media/shengjie/other/KITTI_scene_understanding/python_code/BuildingLocalization/initedModel', str(0)))
        self.pre_imageNet.load_state_dict(checkpoint['pre_imageNet_state_dict'])
        self.LSTM.load_state_dict(checkpoint['LSTM_state_dict'])
        self.paramPredictor.load_state_dict(checkpoint['paramPredictor_state_dict'])
        self.visibilityPredictor.load_state_dict(checkpoint['visibilityPredictor_state_dict'])
    def load(self, idt):
        checkpoint = torch.load(os.path.join(self.svPath, str(idt)))
        self.pre_imageNet.load_state_dict(checkpoint['pre_imageNet_state_dict'])
        self.LSTM.load_state_dict(checkpoint['LSTM_state_dict'])
        self.paramPredictor.load_state_dict(checkpoint['paramPredictor_state_dict'])
        self.visibilityPredictor.load_state_dict(checkpoint['visibilityPredictor_state_dict'])

    def forward(self, videoSequence):
        imageTransforms = transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406, 0.406], [0.229, 0.224, 0.225, 0.225])
        ])
        lista = list(videoSequence.rgb.keys())
        listb = list(videoSequence.depth.keys())
        if videoSequence.imgnum > self.maxLen:
            toIndices = np.intc(np.round(np.linspace(0, videoSequence.imgnum-1, num=self.maxLen)))
        else:
            toIndices = np.intc(np.round(np.linspace(0, videoSequence.imgnum-1, num=videoSequence.imgnum)))
        lstmInput = torch.zeros(len(toIndices), 1, self.lstm_input).cuda()
        for idx, i in enumerate(toIndices):
            concatonatexImg = torch.from_numpy(np.dstack((videoSequence.rgb[lista[i]], videoSequence.depth[listb[i]]))).permute(2,0,1).type(torch.FloatTensor) / torch.FloatTensor([255])
            concatonatexImg_normed = imageTransforms(concatonatexImg).unsqueeze(0).cuda()
            feature = self.pre_imageNet.forward(concatonatexImg_normed)
            lstmInput[idx,0,:] = feature
        if torch.sum(torch.isnan(lstmInput)) == 0:
            lstmOutput, oth = self.LSTM.forward(lstmInput)
            paramPrediction = (self.paramPredictor(lstmOutput[-1,:,:]) - 0.5) * 16
            visiblityPrediction = self.visibilityPredictor(lstmOutput).squeeze(1)
            return paramPrediction, visiblityPrediction
        else:
            return None, None

    def train(self, videoSequence):
        paramPrediction, visiblityPrediction = self.forward(videoSequence)
        if paramPrediction is not None:
            tgtTrans = torch.from_numpy(videoSequence.gtTransition).cuda()
            # tgtVisibility = torch.from_numpy(videoSequence.gtVisibility).cuda()
            if videoSequence.isValid:
                self.optimizerTrans.zero_grad()
                loss1 = torch.mean(torch.sqrt((tgtTrans - paramPrediction) * (tgtTrans - paramPrediction)))
                loss1.backward(retain_graph=True)
                self.optimizerTrans.step()
            """
            self.optimizerVisibility.zero_grad()
            loss2 = torch.mean(torch.sqrt((tgtVisibility - visiblityPrediction) * (tgtVisibility - visiblityPrediction))) * torch.cuda.FloatTensor([100])
            loss2_np = loss2.data.item()
            loss2.backward()
            self.optimizerVisibility.step()
            """
            return loss1
        else:
            return -1
    def train_l2(self, videoSequence):
        paramPrediction, visiblityPrediction = self.forward(videoSequence)
        if paramPrediction is not None:
            tgtTrans = torch.from_numpy(videoSequence.gtTransition).cuda()
            # tgtVisibility = torch.from_numpy(videoSequence.gtVisibility).cuda()
            if videoSequence.isValid:
                self.optimizerTrans.zero_grad()
                loss1 = torch.mean((tgtTrans - paramPrediction).pow(2))
                loss1.backward(retain_graph=True)
                self.optimizerTrans.step()
            """
            self.optimizerVisibility.zero_grad()
            loss2 = torch.mean(torch.sqrt((tgtVisibility - visiblityPrediction) * (tgtVisibility - visiblityPrediction))) * torch.cuda.FloatTensor([100])
            loss2_np = loss2.data.item()
            loss2.backward()
            self.optimizerVisibility.step()
            """
            return loss1
        else:
            return -1

    def test(self, videoSequence):
        paramPrediction, visiblityPrediction = self.forward(videoSequence)
        if paramPrediction is not None:
            tgtTrans = torch.from_numpy(videoSequence.gtTransition).cuda()
            if videoSequence.isValid:
                loss1 = torch.mean(
                    torch.sqrt((tgtTrans - paramPrediction) * (tgtTrans - paramPrediction)))
            return loss1
        else:
            return -100000


class videoSequence:
    def __init__(self, renderedRgb, renderedDepth, gtTransition, gtVisibility, isValid):
        self.imgnum = len(renderedRgb)
        self.rgb = renderedRgb
        self.depth = renderedDepth
        self.gtTransition = gtTransition
        self.gtVisibility = gtVisibility
        self.isValid = isValid


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
pkr = pickleReader(allSeq)
bsm = baselineModel()
fixedFrameNum = 5
iterationTime = 100000000

testComp = pkr.testPortion
trainComp = pkr.tranPortion

if os.path.isfile('/media/shengjie/other/KITTI_scene_understanding/python_code/BuildingLocalization/initedModel/0') == False:
    bsm.initSv()
else:
    bsm.initLoad()
writer = SummaryWriter('/media/shengjie/other/KITTI_scene_understanding/python_code/BuildingLocalization/runs/baseLine_fixedInput_l2')
for i in range(iterationTime):
    randInt = random.randint(0, len(trainComp) - 1)
    curTrainFilePath = trainComp[randInt]
    with open(curTrainFilePath, "rb") as input:
        bdcomp = pickle.load(input)
        vds = videoSequence(bdcomp.rgbs_dict, bdcomp.depths_dict, bdcomp.bdComp.transition,
                            bdcomp.bdComp.visibility, np.sum(bdcomp.bdComp.visibility) > 0)
        lossVal = bsm.train_l2(vds)
        print("%dth iteration, loss is %f" % (i, lossVal))
        writer.add_scalar('TrainLoss', lossVal, i)
    if i % 200 == 0:
        testLossVals = list()
        for add in testComp:
            with open(add, "rb") as input:
                bdcomp = pickle.load(input)
                vds = videoSequence(bdcomp.rgbs_dict, bdcomp.depths_dict, bdcomp.bdComp.transition,
                                    bdcomp.bdComp.visibility, np.sum(bdcomp.bdComp.visibility) > 0)
                testLossVals.append(bsm.test(vds).cpu().detach().numpy())
        print("TestLoss is %f, valid entry is %d" %  (np.mean(testLossVals), len(testLossVals)))
        writer.add_scalar('TestLoss', np.mean(testLossVals), i)
    if i % 500 == 499:
        bsm.sv(i)