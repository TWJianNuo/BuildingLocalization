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
class imageNetWrapper:
    def __init__(self) -> object:
        self.imageNet = models.alexnet(pretrained=True)
        firstConvWeight = self.imageNet.features[0].weight
        new_convLayer = nn.Conv2d(4,64,kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        nn.init.xavier_uniform(new_convLayer.weight)
        new_convLayer.weight[:,0:3,:,:] = self.imageNet.features[0].weight
        new_convLayer.weight = nn.Parameter(new_convLayer.weight)
        feature_layerList = list(self.imageNet.features.children())
        feature_layerList[0] = new_convLayer
        self.imageNet.features = nn.Sequential(*feature_layerList)

        classifier_layerList = list(self.imageNet.classifier.children())[:-1]
        self.imageNet.classifier = nn.Sequential(*classifier_layerList)
        self.imageNet.cuda()
    def getImageNet(self):
        return self.imageNet

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
        self.paramPredictor = nn.Sequential(nn.Linear(self.lstm_hidden, self.lstm_hidden), nn.ReLU(), nn.Linear(self.lstm_hidden, 3)).cuda()
        self.visibilityPredictor = nn.Sequential(nn.Linear(self.lstm_hidden, self.lstm_hidden), nn.ReLU(), nn.Linear(self.lstm_hidden, 1)).cuda()
        self.lossTrans = nn.MSELoss()
        self.lossVisibility = nn.MSELoss()
        self.optimizerTrans = optim.SGD(list(self.pre_imageNet.parameters()) + list(self.LSTM.parameters()) + list(self.paramPredictor.parameters()), lr=0.01, momentum=0.9)
        self.optimizerVisibility = optim.SGD(self.visibilityPredictor.parameters(), lr=0.01, momentum=0.9)
        self.svPath = '/media/shengjie/other/KITTI_scene_understanding/python_code/BuildingLocalization/svModel'
        a = 1

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

    def forward(self, videoSequence):
        max_depth = 300
        imageTransforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406, 0.406], [0.229, 0.224, 0.225, 0.225])
        ])
        lista = list(videoSequence.rgb.keys())
        listb = list(videoSequence.depth.keys())
        lstmInput = torch.zeros(len(lista),1,self.lstm_input).cuda()
        for i in range(videoSequence.imgnum):
            concatonatexImg = np.dstack((videoSequence.rgb[lista[i]], videoSequence.depth[listb[i]]/max_depth*255 )).astype(np.uint8)
            concatonatexImg_normed = imageTransforms(concatonatexImg).unsqueeze(0).cuda()
            feature = self.pre_imageNet.forward(concatonatexImg_normed)
            lstmInput[i,0,:] = feature
        lstmOutput, oth = self.LSTM.forward(lstmInput)
        paramPrediction = (self.paramPredictor(lstmOutput[-1,:,:]) - 0.5) * 16
        visiblityPrediction = self.visibilityPredictor(lstmOutput).squeeze(1)
        return paramPrediction, visiblityPrediction

    def train(self, videoSequence):
        paramPrediction, visiblityPrediction = self.forward(videoSequence)
        tgtTrans = torch.from_numpy(videoSequence.gtTransition).cuda()
        tgtVisibility = torch.from_numpy(videoSequence.gtVisibility).cuda()
        loss1_np = np.zeros(1)
        if videoSequence.isValid:
            self.optimizerTrans.zero_grad()
            loss1 = torch.mean(torch.sqrt((tgtTrans - paramPrediction) * (tgtTrans - paramPrediction))) * torch.cuda.FloatTensor([100])
            loss1_np = loss1.data.item()
            loss1.backward()
            self.optimizerTrans.step()

        paramPrediction, visiblityPrediction = self.forward(videoSequence)
        self.optimizerVisibility.zero_grad()
        loss2 = torch.mean(torch.sqrt((tgtVisibility - visiblityPrediction) * (tgtVisibility - visiblityPrediction))) * torch.cuda.FloatTensor([100])
        loss2_np = loss2.data.item()
        loss2.backward()
        self.optimizerVisibility.step()
        return loss1_np, loss2_np
    def test(self, videoSequence):
        paramPrediction, visiblityPrediction = self.forward(videoSequence)
        tgtTrans = torch.from_numpy(videoSequence.gtTransition).cuda()
        tgtVisibility = torch.from_numpy(videoSequence.gtVisibility).cuda()
        loss1_np = np.zeros(1)
        if videoSequence.isValid:
            loss1 = torch.mean(
                torch.sqrt((tgtTrans - paramPrediction) * (tgtTrans - paramPrediction))) * torch.cuda.FloatTensor([100])
            loss1_np = loss1.data.item()
        loss2 = torch.mean(torch.sqrt((tgtVisibility - visiblityPrediction) * (tgtVisibility - visiblityPrediction))) * torch.cuda.FloatTensor([100])
        loss2_np = loss2.data.item()
        return paramPrediction, visiblityPrediction.squeeze(1), loss1_np, loss2_np

"""
itNum = 5000
bsm = baselineModel()
tt =videoSequenceProvider.tt_struct()
rgbRenderTime = time.time() - time.time()
neuralNetTime = time.time() - time.time()
for i in range(itNum):
    timeRec = time.time()
    vs = tt.getRandomTrainSample()
    rgbRenderTime = rgbRenderTime + time.time() - timeRec
    if len(vs.rgb) > 0:
        timeRec = time.time()
        loss1, loss2 = bsm.train(vs)
        bsm.train(vs)
        neuralNetTime = neuralNetTime + time.time() - timeRec
        print("Loss1 %f, Loss2 %f" % (loss1, loss2))
    if i % 100 == 0:
        bsm.sv(i)
    print("Time ratio is %f" % (rgbRenderTime / (rgbRenderTime + neuralNetTime)))
    print("%dth iteration finished" % i)
"""


"""
# Below is for test
bsm = baselineModel()
bsm.load(4200)
tt =videoSequenceProvider.tt_struct()
seqNameTest = '2011_09_26_drive_0117_sync'
losstest1_rec = list()
losstest2_rec = list()
for i in range(tt.getSpecificSeqBdNum(seqNameTest)):
    paramPrediction, visiblityPrediction, loss1_np, loss2_np = bsm.test(tt.getSequentialTest(seqNameTest, i))
    visiblityPrediction_np = visiblityPrediction.cpu().detach().numpy()
    descretRizeRe = np.zeros_like(visiblityPrediction_np)
    descretRizeRe[visiblityPrediction_np > 1] = 1
    descretRizeRe[visiblityPrediction_np < 1] = 0
    descretRizeRe_bool = descretRizeRe == 1
    tt.changeDataReaderContent(seqNameTest, i, paramPrediction.cpu().detach().numpy(), descretRizeRe_bool)
    losstest1_rec.append(loss1_np)
    losstest2_rec.append(loss2_np)
    print(i)
tt.renderSepecificSequence(seqNameTest)

seqNameTrain = '2011_09_30_drive_0018_sync'
losstrain1_rec = list()
losstrain2_rec = list()
for i in range(tt.getSpecificSeqBdNum(seqNameTrain)):
    paramPrediction, visiblityPrediction, loss1_np, loss2_np = bsm.test(tt.getSequentialTest(seqNameTrain, i))
    visiblityPrediction_np = visiblityPrediction.cpu().detach().numpy()
    descretRizeRe = np.zeros_like(visiblityPrediction_np)
    descretRizeRe[visiblityPrediction_np > 1] = 1
    descretRizeRe[visiblityPrediction_np < 1] = 0
    descretRizeRe_bool = descretRizeRe == 1
    tt.changeDataReaderContent(seqNameTrain, i, paramPrediction.cpu().detach().numpy(), descretRizeRe_bool)
    losstrain1_rec.append(loss1_np)
    losstrain2_rec.append(loss2_np)
    print(i)
tt.renderSepecificSequence(seqNameTrain)


print(losstrain1_rec)
print(losstest2_rec)
print(losstrain1_rec)
print(losstrain2_rec)
"""