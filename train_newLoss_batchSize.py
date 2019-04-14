import json
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import os
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
        self.invalidIndices = dict()
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
        if os.path.isfile(
                '/media/shengjie/other/KITTI_scene_understanding/python_code/BuildingLocalization/dataSplit/dataSplit.p') == False:
            tmpdict = dict()
            tmpdict['train'] = self.tranPortion
            tmpdict['test'] = self.testPortion
            pickle.dump(tmpdict, open(
                "/media/shengjie/other/KITTI_scene_understanding/python_code/BuildingLocalization/dataSplit/dataSplit.p",
                "wb"))
        else:
            tmpdict = pickle.load(open(
                "/media/shengjie/other/KITTI_scene_understanding/python_code/BuildingLocalization/dataSplit/dataSplit.p",
                "rb"))
            self.tranPortion = tmpdict['train']
            self.testPortion = tmpdict['test']
            for idx in self.testPortion:
                print(idx)


class baselineModel:
    def __init__(self, batchSize) -> object:
        super(baselineModel, self).__init__()
        self.pre_imageNet = models.alexnet(pretrained=True)
        new_convLayer = nn.Conv2d(4, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        nn.init.xavier_uniform(new_convLayer.weight)
        new_convLayer.weight[:, 0:3, :, :] = self.pre_imageNet.features[0].weight
        new_convLayer.weight = nn.Parameter(new_convLayer.weight)
        feature_layerList = list(self.pre_imageNet.features.children())
        feature_layerList[0] = new_convLayer
        self.pre_imageNet.features = nn.Sequential(*feature_layerList)
        self.batchSize = batchSize

        classifier_layerList = list(self.pre_imageNet.classifier.children())[:-1]
        self.pre_imageNet.classifier = nn.Sequential(*classifier_layerList)
        self.pre_imageNet.cuda()

        self.lstm_input = 4096
        self.lstm_hidden = int(4096 / 4)
        self.LSTM = nn.LSTM(self.lstm_input, self.lstm_hidden)
        self.LSTM.cuda()
        self.paramPredictor = nn.Sequential(nn.Linear(self.lstm_hidden, self.lstm_hidden), nn.ReLU(),
                                            nn.Linear(self.lstm_hidden, 3), nn.Sigmoid()).cuda()
        self.visibilityPredictor = nn.Sequential(nn.Linear(self.lstm_hidden, self.lstm_hidden), nn.ReLU(),
                                                 nn.Linear(self.lstm_hidden, 1)).cuda()

        # self.optimizerTrans = optim.SGD(list(self.pre_imageNet.parameters()) + list(self.LSTM.parameters()) + list(self.paramPredictor.parameters()), lr=0.0000001)
        # self.optimizerTrans = optim.SGD(list(self.pre_imageNet.parameters()) + list(self.LSTM.parameters()) + list(
        #     self.paramPredictor.parameters()), lr=0.0000000001)
        self.optimizerTrans = optim.SGD(list(self.pre_imageNet.parameters()) + list(self.LSTM.parameters()) + list(
            self.paramPredictor.parameters()), lr=0.000001)
        self.optimizerVisibility = optim.SGD(self.visibilityPredictor.parameters(), lr=0.0000001)
        self.svPath = '/media/shengjie/other/KITTI_scene_understanding/python_code/BuildingLocalization/svModel'
        self.maxLen = 10
        self.imageTransforms = transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406, 0.406], [0.229, 0.224, 0.225, 0.225])
        ])

    def sv(self, idt):
        torch.save({
            'pre_imageNet_state_dict': self.pre_imageNet.state_dict(),
            'LSTM_state_dict': self.LSTM.state_dict(),
            'paramPredictor_state_dict': self.paramPredictor.state_dict(),
            'visibilityPredictor_state_dict': self.visibilityPredictor.state_dict(),
        }, os.path.join(self.svPath, str(idt)))

    def load(self, idt):
        checkpoint = torch.load(os.path.join(self.svPath, str(idt)))
        self.pre_imageNet.load_state_dict(checkpoint['pre_imageNet_state_dict'])
        self.LSTM.load_state_dict(checkpoint['LSTM_state_dict'])
        self.paramPredictor.load_state_dict(checkpoint['paramPredictor_state_dict'])
        self.visibilityPredictor.load_state_dict(checkpoint['visibilityPredictor_state_dict'])


    def forward(self, buildingEntityList):
        lstmOutputList = list()
        validList = list()
        for buildingEntity in buildingEntityList:
            imgNum = len(buildingEntity.rgbs_dict)
            lista = list(buildingEntity.rgbs_dict.keys())
            listb = list(buildingEntity.depths_dict.keys())
            if imgNum > self.maxLen:
                toIndices = np.intc(np.round(np.linspace(0, imgNum - 1, num=self.maxLen)))
            else:
                toIndices = np.intc(np.round(np.linspace(0, imgNum - 1, num=imgNum)))
            stackedInputImagesList = list()
            for idx, i in enumerate(toIndices):
                concatonatexImg = torch.from_numpy(
                    np.dstack((buildingEntity.rgbs_dict[lista[i]], buildingEntity.depths_dict[listb[i]]))).permute(2, 0,
                                                                                                                   1).type(
                    torch.FloatTensor) / torch.FloatTensor([255])
                stackedInputImagesList.append(self.imageTransforms(concatonatexImg).unsqueeze(0))
            stackedInputImages = torch.cat(stackedInputImagesList, dim=0).cuda()
            lstmInput = self.pre_imageNet(stackedInputImages)
            lstmInput_unqueezed = lstmInput.unsqueeze(dim=1)
            if torch.sum(torch.isnan(lstmInput_unqueezed)) == 0:
                validList.append(1)
                lstmOutput, oth = self.LSTM.forward(lstmInput_unqueezed)
                lstmOutput_val = lstmOutput[-1,:,:]
                lstmOutputList.append(lstmOutput_val.unsqueeze(0))
            else:
                validList.append(0)
        if len(lstmOutputList) > 0:
            lstmOutput = torch.cat(lstmOutputList, dim = 0)
            intemRe1 = self.paramPredictor(lstmOutput)
            paramPrediction = (intemRe1 - 0.5) * 16
            return paramPrediction, validList
        else:
            return None, None

    def initSv(self):
        torch.save({
            'pre_imageNet_state_dict': self.pre_imageNet.state_dict(),
            'LSTM_state_dict': self.LSTM.state_dict(),
            'paramPredictor_state_dict': self.paramPredictor.state_dict(),
            'visibilityPredictor_state_dict': self.visibilityPredictor.state_dict(),
        }, os.path.join('/media/shengjie/other/KITTI_scene_understanding/python_code/BuildingLocalization/initedModel',
                        str(0)))

    def initLoad(self):
        checkpoint = torch.load(
            os.path.join('/media/shengjie/other/KITTI_scene_understanding/python_code/BuildingLocalization/initedModel',
                         str(0)))
        self.pre_imageNet.load_state_dict(checkpoint['pre_imageNet_state_dict'])
        self.LSTM.load_state_dict(checkpoint['LSTM_state_dict'])
        self.paramPredictor.load_state_dict(checkpoint['paramPredictor_state_dict'])
        self.visibilityPredictor.load_state_dict(checkpoint['visibilityPredictor_state_dict'])

    def get3dPointLoss(self, sampledPts, transition, rotation, bdCenter):
        CenterMvArr1 = torch.eye(4)
        CenterMvArr1[0, 3] = -bdCenter[0]
        CenterMvArr1[1, 3] = -bdCenter[1]
        CenterMvArr1[2, 3] = -bdCenter[2]

        CenterMvArr2 = torch.eye(4)
        CenterMvArr2[0, 3] = bdCenter[0]
        CenterMvArr2[1, 3] = bdCenter[1]
        CenterMvArr2[2, 3] = bdCenter[2]

        rot_tranArr = torch.eye(4)
        rot_tranArr[0, 0] = torch.cos(rotation[2])
        rot_tranArr[0, 1] = -torch.sin(rotation[2])
        rot_tranArr[1, 0] = torch.sin(rotation[2])
        rot_tranArr[1, 1] = torch.cos(rotation[2])

        rot_tranArr[0, 3] = transition[0]
        rot_tranArr[1, 3] = transition[1]
        rot_tranArr[2, 3] = transition[2]

        # rotatedPts = torch.t(torch.matmul(CenterMvArr2, torch.matmul(rot_tranArr, torch.matmul(CenterMvArr1, torch.t(sampledPts)))))
        rotatedPts = torch.t(
            torch.matmul(torch.matmul(torch.matmul(CenterMvArr2, rot_tranArr), CenterMvArr1), torch.t(sampledPts)))
        return rotatedPts

    def trainLoss2d(self, buildingEntity):
        paramPrediction, visiblityPrediction = self.forward(buildingEntity)
        if paramPrediction is not None:
            torchSampledPts = torch.from_numpy(buildingEntity.sampledPts).type(torch.float32)
            transitionTorchGT = torch.from_numpy(buildingEntity.bdComp.transition).type(torch.float32)
            transitionTorchEst = paramPrediction[0]
            rotationTorch = torch.from_numpy(buildingEntity.bdComp.angles).type(torch.float32)
            bdCenterTorch = torch.from_numpy(np.mean(buildingEntity.bdComp.botPolygon, axis=0)).type(torch.float32)

            pts3dGt = self.get3dPointLoss(torchSampledPts, transitionTorchGT, rotationTorch, bdCenterTorch)
            pts3dEst = self.get3dPointLoss(torchSampledPts, transitionTorchEst, rotationTorch, bdCenterTorch)

            tr_oxts2velo = torch.from_numpy(buildingEntity.tr_oxts2velo)
            tr_grid2oxts = dict()

            """
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            pts3dGt_numpy = pts3dGt.detach().numpy()
            pts3dEst_numpy = pts3dEst.detach().numpy()
            ax.scatter(pts3dGt_numpy[:,0], pts3dGt_numpy[:,1], pts3dGt_numpy[:,2], c='r')
            ax.scatter(pts3dEst_numpy[:, 0], pts3dEst_numpy[:, 1], pts3dEst_numpy[:, 2], c='g')
            fig.show()
            """

            """
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            pts3dGt_numpy = a.detach().numpy()
            pts3dEst_numpy = b.detach().numpy()
            ax.scatter(pts3dGt_numpy[:,0], pts3dGt_numpy[:,1], pts3dGt_numpy[:,2], c='r')
            ax.scatter(pts3dEst_numpy[:, 0], pts3dEst_numpy[:, 1], pts3dEst_numpy[:, 2], c='g')
            fig.show()
            """
            lossPt2d = torch.zeros(1)[0]
            selectoraRec = dict()
            selectorbRec = dict()
            selectorRec = dict()
            imgSizeRec = dict()
            aRec = dict()
            bRec = dict()
            a_2d_pRec = dict()
            b_2d_pRec = dict()
            a_2dRec = dict()
            b_2dRec = dict()
            for idx in buildingEntity.rgbs_dict.keys():
                tr_grid2oxts[idx] = (torch.from_numpy(buildingEntity.tr_grid2oxtsDict[idx]).type(torch.float32))
                imgSize = torch.from_numpy(buildingEntity.imgSizeDict[idx])
                a = torch.t(torch.matmul(torch.matmul(tr_oxts2velo, tr_grid2oxts[idx]), torch.t(pts3dGt)))
                b = torch.t(torch.matmul(torch.matmul(tr_oxts2velo, tr_grid2oxts[idx]), torch.t(pts3dEst)))

                a_2d_p = torch.t(torch.matmul(torch.matmul(torch.from_numpy(buildingEntity.intrinsic),
                                                           torch.from_numpy(buildingEntity.extrinsic)), torch.t(a)))
                a_2d = torch.zeros((a_2d_p.shape[0], a_2d_p.shape[1] - 1))
                a_2d[:, 0] = a_2d_p[:, 0] / a_2d_p[:, 2]
                a_2d[:, 1] = a_2d_p[:, 1] / a_2d_p[:, 2]
                a_2d[:, 2] = a_2d_p[:, 2]
                selectora = (a_2d[:, 0] > 0) & (a_2d[:, 0] < imgSize[1].type(torch.float32)) & (a_2d[:, 1] > 0) & (
                            a_2d[:, 1] < imgSize[0].type(torch.float32)) & (a_2d[:, 2] > 0)

                b_2d_p = torch.t(torch.matmul(torch.matmul(torch.from_numpy(buildingEntity.intrinsic),
                                                           torch.from_numpy(buildingEntity.extrinsic)), torch.t(b)))
                b_2d = torch.zeros((b_2d_p.shape[0], b_2d_p.shape[1] - 1))
                b_2d[:, 0] = b_2d_p[:, 0] / b_2d_p[:, 2]
                b_2d[:, 1] = b_2d_p[:, 1] / b_2d_p[:, 2]
                b_2d[:, 2] = b_2d_p[:, 2]
                selectorb = (b_2d[:, 0] > 0) & (b_2d[:, 0] < imgSize[1].type(torch.float32)) & (b_2d[:, 1] > 0) & (
                            b_2d[:, 1] < imgSize[0].type(torch.float32)) & (b_2d[:, 2] > 0)
                selector = (selectora & selectorb).type(torch.float32).unsqueeze(1).repeat(1, 3)

                imgSizeRec[idx] = imgSize
                aRec[idx] = a
                bRec[idx] = b
                a_2d_pRec[idx] = a_2d_p
                b_2d_pRec[idx] = b_2d_p
                a_2dRec[idx] = a_2d
                b_2dRec[idx] = b_2d
                selectoraRec[idx] = selectora
                selectorbRec[idx] = selectorb
                selectorRec[idx] = selector
                lossPt2d = lossPt2d + torch.sum(((a_2d - b_2d) * selector).pow(2))
            lossPt2d = lossPt2d / len(buildingEntity.rgbs_dict.keys())

            """
            idx = list(buildingEntity.rgbs_dict.keys())[4]
            img = mpimg.imread(buildingEntity.imgPathDict[idx])
            imgplot = plt.imshow(img)
            plt.axis([0, 1226, 370, 0])

            plt.scatter(a_2d.detach().numpy()[:,0], a_2d.detach().numpy()[:,1], s=1, c='r')
            plt.scatter(b_2d.detach().numpy()[:, 0], b_2d.detach().numpy()[:, 1], s=1, c='b')
            """
            maxLoss = torch.Tensor([6e5])[0]

            if np.sum(buildingEntity.bdComp.visibility) > 0 and lossPt2d != 0 and lossPt2d < maxLoss:
                self.optimizerTrans.zero_grad()
                lossPt2d.backward()
                self.optimizerTrans.step()
            """
            self.optimizerVisibility.zero_grad()
            loss2 = torch.mean(torch.sqrt((tgtVisibility - visiblityPrediction) * (tgtVisibility - visiblityPrediction))) * torch.cuda.FloatTensor([100])
            loss2_np = loss2.data.item()
            loss2.backward()
            self.optimizerVisibility.step()
            """

            """
            sampleRgb = buildingEntity.rgbs_dict[258]
            plt.imshow(sampleRgb)
            """
            return lossPt2d
        else:
            return -1

    def trainLosslog2d(self, buildingEntityList):
        paramPredictionComb, validMark = self.forward(buildingEntityList)
        lossPt2dSum = torch.zeros(1)[0]
        count = 0
        for idx, buildingEntity in enumerate(buildingEntityList):
            if validMark[idx] != 0:
                paramPrediction = paramPredictionComb[count, :]
                count = count + 1
                torchSampledPts = torch.from_numpy(buildingEntity.sampledPts).type(torch.float32)
                transitionTorchGT = torch.from_numpy(buildingEntity.bdComp.transition).type(torch.float32)
                transitionTorchEst = paramPrediction[0]
                rotationTorch = torch.from_numpy(buildingEntity.bdComp.angles).type(torch.float32)
                bdCenterTorch = torch.from_numpy(np.mean(buildingEntity.bdComp.botPolygon, axis=0)).type(torch.float32)

                pts3dGt = self.get3dPointLoss(torchSampledPts, transitionTorchGT, rotationTorch, bdCenterTorch)
                pts3dEst = self.get3dPointLoss(torchSampledPts, transitionTorchEst, rotationTorch, bdCenterTorch)

                tr_oxts2velo = torch.from_numpy(buildingEntity.tr_oxts2velo)
                tr_grid2oxts = dict()
                """
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                pts3dGt_numpy = pts3dGt.detach().numpy()
                pts3dEst_numpy = pts3dEst.detach().numpy()
                ax.scatter(pts3dGt_numpy[:,0], pts3dGt_numpy[:,1], pts3dGt_numpy[:,2], c='r')
                ax.scatter(pts3dEst_numpy[:, 0], pts3dEst_numpy[:, 1], pts3dEst_numpy[:, 2], c='g')
                fig.show()
                """

                """
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                pts3dGt_numpy = a.detach().numpy()
                pts3dEst_numpy = b.detach().numpy()
                ax.scatter(pts3dGt_numpy[:,0], pts3dGt_numpy[:,1], pts3dGt_numpy[:,2], c='r')
                ax.scatter(pts3dEst_numpy[:, 0], pts3dEst_numpy[:, 1], pts3dEst_numpy[:, 2], c='g')
                fig.show()
                """
                lossPt2d = torch.zeros(1)[0]
                selectoraRec = dict()
                selectorbRec = dict()
                selectorRec = dict()
                imgSizeRec = dict()
                aRec = dict()
                bRec = dict()
                a_2d_pRec = dict()
                b_2d_pRec = dict()
                a_2dRec = dict()
                b_2dRec = dict()
                for idx in buildingEntity.rgbs_dict.keys():
                    tr_grid2oxts[idx] = (torch.from_numpy(buildingEntity.tr_grid2oxtsDict[idx]).type(torch.float32))
                    imgSize = torch.from_numpy(buildingEntity.imgSizeDict[idx])
                    a = torch.t(torch.matmul(torch.matmul(tr_oxts2velo, tr_grid2oxts[idx]), torch.t(pts3dGt)))
                    b = torch.t(torch.matmul(torch.matmul(tr_oxts2velo, tr_grid2oxts[idx]), torch.t(pts3dEst)))

                    a_2d_p = torch.t(torch.matmul(torch.matmul(torch.from_numpy(buildingEntity.intrinsic),
                                                               torch.from_numpy(buildingEntity.extrinsic)), torch.t(a)))
                    a_2d = torch.zeros((a_2d_p.shape[0], a_2d_p.shape[1] - 1))
                    a_2d[:, 0] = a_2d_p[:, 0] / a_2d_p[:, 2]
                    a_2d[:, 1] = a_2d_p[:, 1] / a_2d_p[:, 2]
                    a_2d[:, 2] = a_2d_p[:, 2]
                    selectora = (a_2d[:, 0] > 0) & (a_2d[:, 0] < imgSize[1].type(torch.float32)) & (a_2d[:, 1] > 0) & (
                                a_2d[:, 1] < imgSize[0].type(torch.float32)) & (a_2d[:, 2] > 0)

                    b_2d_p = torch.t(torch.matmul(torch.matmul(torch.from_numpy(buildingEntity.intrinsic),
                                                               torch.from_numpy(buildingEntity.extrinsic)), torch.t(b)))
                    b_2d = torch.zeros((b_2d_p.shape[0], b_2d_p.shape[1] - 1))
                    b_2d[:, 0] = b_2d_p[:, 0] / b_2d_p[:, 2]
                    b_2d[:, 1] = b_2d_p[:, 1] / b_2d_p[:, 2]
                    b_2d[:, 2] = b_2d_p[:, 2]
                    selectorb = (b_2d[:, 0] > 0) & (b_2d[:, 0] < imgSize[1].type(torch.float32)) & (b_2d[:, 1] > 0) & (
                                b_2d[:, 1] < imgSize[0].type(torch.float32)) & (b_2d[:, 2] > 0)
                    selector = (selectora & selectorb).type(torch.float32).unsqueeze(1).repeat(1, 3)

                    imgSizeRec[idx] = imgSize
                    aRec[idx] = a
                    bRec[idx] = b
                    a_2d_pRec[idx] = a_2d_p
                    b_2d_pRec[idx] = b_2d_p
                    a_2dRec[idx] = a_2d
                    b_2dRec[idx] = b_2d
                    selectoraRec[idx] = selectora
                    selectorbRec[idx] = selectorb
                    selectorRec[idx] = selector
                    lossPt2d = lossPt2d + torch.sum(((a_2d - b_2d) * selector).pow(2))
                lossPt2d = torch.log(1 + lossPt2d / len(buildingEntity.rgbs_dict.keys()))
                lossPt2dSum = lossPt2dSum + lossPt2d
                """
                img = mpimg.imread(buildingEntity.imgPathDict[idx])
                imgplot = plt.imshow(img)
                plt.axis([0, 1226, 370, 0])
    
                plt.scatter(a_2d.detach().numpy()[:,0], a_2d.detach().numpy()[:,1], c='r')
                plt.scatter(b_2d.detach().numpy()[:, 0], b_2d.detach().numpy()[:, 1], c='b')
    
                imgs = list(buildingEntity.rgbs_dict.values())
                imgplot = plt.imshow(imgs[0])
                """
        maxLoss = torch.Tensor([6e5])[0]
        lossPt2dSum = lossPt2dSum / np.sum(validMark)
        if np.sum(validMark) > 0 and lossPt2dSum != 0 and lossPt2dSum < maxLoss:
            self.optimizerTrans.zero_grad()
            lossPt2dSum.backward()
            self.optimizerTrans.step()
            """
            self.optimizerVisibility.zero_grad()
            loss2 = torch.mean(torch.sqrt((tgtVisibility - visiblityPrediction) * (tgtVisibility - visiblityPrediction))) * torch.cuda.FloatTensor([100])
            loss2_np = loss2.data.item()
            loss2.backward()
            self.optimizerVisibility.step()
            """
            return lossPt2dSum
        else:
            return -1


    def test(self, buildingEntityList):
        paramPrediction, validList = self.forward(buildingEntityList)
        loss1 = torch.Tensor([0])[0]
        if paramPrediction is not None:
            count = 0
            for idx, val in enumerate(validList):
                if val == 1:
                    tgtTrans = torch.from_numpy(buildingEntityList[idx].bdComp.transition).cuda()
                    loss1 = loss1 + torch.mean(torch.abs(tgtTrans - paramPrediction[count, :]))
                    count = count + 1
            loss1 = loss1 / paramPrediction.shape[0]
            return loss1
        else:
            return None


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
batchSize = 32
pkr = pickleReader(allSeq)
bsm = baselineModel(batchSize)
fixedFrameNum = 5
iterationTime = 100000

testComp = pkr.testPortion
trainComp = pkr.tranPortion

if os.path.isfile(
        '/media/shengjie/other/KITTI_scene_understanding/python_code/BuildingLocalization/initedModel/0') == False:
    bsm.initSv()
else:
    bsm.initLoad()

# writer = SummaryWriter('/media/shengjie/other/KITTI_scene_understanding/python_code/BuildingLocalization/runs/logLoss2d_ex_fixedInit_batchSize32')
for i in range(iterationTime):
    bdCompInputList = list()
    for k in range(batchSize):
        randInt = random.randint(0, len(trainComp) - 1)
        curTrainFilePath = trainComp[randInt]
        with open(curTrainFilePath, "rb") as input:
            bdcomp = pickle.load(input)
            bdCompInputList.append(bdcomp)
    lossVal = bsm.trainLosslog2d(bdCompInputList)
    if lossVal is not None:
        print("%dth iteration, loss is %f" % (i, lossVal))
        # writer.add_scalar('TrainLoss', lossVal, i)
    if i % 200 == 0:
        testLossVals = list()
        for add in testComp:
            with open(add, "rb") as input:
                bdcomp = pickle.load(input)
                testVal = bsm.test([bdcomp])
                if testVal is not None:
                    testLossVals.append(testVal.cpu().detach().numpy())
        print("TestLoss is %f" % np.mean(testLossVals))
        if np.mean(testLossVals) > 0:
            a = 1
            # writer.add_scalar('TestLoss', np.mean(testLossVals), i)
    if i % 500 == 499:
        bsm.sv(i)