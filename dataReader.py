import os
import numpy as np
import numba
class buildingComp:
    def __init__(self, id, angles, transition, height, botPolygon, visibility, range_visibilityArray):
        self.id = id
        self.angles = angles
        self.transition = transition
        self.height = height
        self.botPolygon = botPolygon
        self.visibility = visibility
        self.range_visibilityArray = range_visibilityArray
class dataReader:
    def __init__(self, sequenceName):
        self.rootPath = '/media/shengjie/other/KITTI_scene_understanding/matlab_code/building_label_program/dataForRender'
        self.sequenceName = sequenceName
        self.readImageSize()
        self.readImagePath()
        self.readOtherParams()
        self.readTrGrid2Oxts()
        self.readBuildingBoundPolygon()
        self.valueList = list(self.buildingComp.keys())
    def readImageSize(self):
        txtFilePath = os.path.join(self.rootPath, self.sequenceName, 'imageSize.txt')
        content = open(txtFilePath).readlines()
        self.imageSize = np.zeros((len(content),2), dtype = np.intc)
        count = 0
        for line in content:
            lineComp = line.split(' ')
            self.imageSize[count, 0] = int(float(lineComp[2]))
            self.imageSize[count, 1] = int(float(lineComp[3]))
            count = count + 1
    def readImagePath(self):
        txtFilePath = os.path.join(self.rootPath, self.sequenceName, 'rgbPath.txt')
        self.rgbFilePaths = open(txtFilePath).readlines()
        for i in range(len(self.rgbFilePaths)):
            tmpPath = self.rgbFilePaths[i]
            self.rgbFilePaths[i] = tmpPath[0:-1]
    def readOtherParams(self):
        txtFilePath = os.path.join(self.rootPath, self.sequenceName, 'otherParams.txt')
        content = open(txtFilePath).readlines()
        oxts2veloComp = content[0].split(' ')
        self.tr_oxts2velo = np.zeros([4,4], dtype = np.float32)
        count = 0
        for i in range(4):
            for j in range(4):
                self.tr_oxts2velo[j][i] = float(oxts2veloComp[count + 1])
                count = count + 1
        self.tr_oxts2velo = np.transpose(self.tr_oxts2velo)

        extrinsicComp = content[1].split(' ')
        self.extrinsic = np.zeros([4, 4], dtype = np.float32)
        count = 0
        for i in range(4):
            for j in range(4):
                self.extrinsic[j][i] = float(extrinsicComp[count + 1])
                count = count + 1
        self.extrinsic = np.transpose(self.extrinsic)

        intrinsicComp = content[2].split(' ')
        self.intrinsic = np.zeros([4, 4], dtype = np.float32)
        count = 0
        for i in range(4):
            for j in range(4):
                self.intrinsic[j][i] = float(intrinsicComp[count + 1])
                count = count + 1
        self.intrinsic = np.transpose(self.intrinsic)

        imagePlaneContent = content[3].split(' ')
        self.imagePlane = np.zeros(4, dtype=np.float32)
        count = 0
        for i in range(4):
            self.imagePlane[i] = float(imagePlaneContent[count + 1])
            count = count + 1
    def readTrGrid2Oxts(self):
        txtFilePath = os.path.join(self.rootPath, self.sequenceName, 'Trs_grid2oxts.txt')
        content = open(txtFilePath).readlines()
        self.trs_grid2oxts = list()
        for line in content:
            tmpTr = np.zeros([4, 4], dtype=np.float32)
            lineComp = line.split(' ')
            count = 0
            for i in range(4):
                for j in range(4):
                    tmpTr[j][i] = float(lineComp[count + 2])
                    count = count + 1
            tmpTr_transpoed = np.transpose(tmpTr)
            self.trs_grid2oxts.append(tmpTr_transpoed)
    def readBuildingBoundPolygon(self):
        txtFilePath = os.path.join(self.rootPath, self.sequenceName, 'buildingBoundingPolygon.txt')
        content = open(txtFilePath).readlines()
        self.buildingComp = dict()
        count = 0
        roundNum = 8
        for line in content:
            if count % roundNum == 0:
                tmpComp = line.split(' ')
                buildingInd = np.int32(int(tmpComp[1]))
                count = count + 1
            elif count % roundNum == 1:
                tmpComp = line.split(' ')
                angles = np.zeros(3, dtype=np.float32)
                for i in range(3):
                    angles[i] = float(tmpComp[i+1])
                count = count + 1
            elif count % roundNum == 2:
                tmpComp = line.split(' ')
                transitions = np.zeros(3, dtype=np.float32)
                for i in range(3):
                    transitions[i] = float(tmpComp[i+1])
                count = count + 1
            elif count % roundNum == 3:
                tmpComp = line.split(' ')
                height = np.float32(float(tmpComp[1]))
                count = count + 1
            elif count % roundNum == 4:
                if line == 'BotPolygon:\n':
                    npPtList = list()
                    continue
                elif line == 'Visibility Record:\n':
                    count = count + 1
                    continue
                else:
                    tmpComp = line.split(' ')
                    tmpPoint = np.zeros([3], dtype=np.float32)
                    for i in range(3):
                        tmpPoint[i] = float(tmpComp[i])
                    npPtList.append(tmpPoint)
            elif count % roundNum == 5:
                botPolygon = np.array(npPtList)
                tmpComp = line.split(' ')
                visibilityArray = np.zeros(len(tmpComp)-1, dtype=np.float32)
                for i in range(len(tmpComp)-1):
                    visibilityArray[i] = np.int32(tmpComp[i])
                count = count + 1
            elif count % roundNum == 6:
                count = count + 1
            elif count % roundNum == 7:
                tmpComp = line.split(' ')
                range_visibilityArray = np.zeros(len(tmpComp)-1, dtype=np.float32)
                for i in range(len(tmpComp)-1):
                    range_visibilityArray[i] = np.int32(tmpComp[i])
                self.buildingComp[buildingInd] = buildingComp(buildingInd, angles, transitions, height, botPolygon,
                                                              visibilityArray, range_visibilityArray)
                count = count + 1
