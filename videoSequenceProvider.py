import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
from PIL import Image
import dataReader
from numba import jit
import matplotlib.pyplot as plt
import gpuWrappedFunc
import matplotlib.image as mpimg
import os
import sys
from mpl_toolkits.mplot3d import Axes3D
import time
import random
import pickle

def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype="float32")
    return data


def cashImageIntoGPU(data_reader, imgIndex):
    imgRef = list()
    for i in range(len(imgIndex)):
        tmpRgb = load_image(data_reader.rgbFilePaths[imgIndex[i]])
        tmpRgb_gpu = cuda.mem_alloc(tmpRgb.nbytes)
        imgRef.append(tmpRgb_gpu)
    return imgRef


def wrapPolygonGeneration(data_reader):
    mod = SourceModule("""
      #include <stdio.h>
      #include <string.h>
      __global__ void doublify(float *a)
      {
        int idx = threadIdx.x + threadIdx.y*4;
        a[idx] *= 2;
        const char *colors[4] = {"1 "};
        printf(colors[0]);
        std::to_string(3.1415926);
      }
      """)
    func = mod.get_function("doublify")
    return func


def drawBuilding_PointForm(triPlane):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(np.size(triPlane, 0)):
        cx = triPlane[i * 5: i * 5 + 5, 0]
        cy = triPlane[i * 5: i * 5 + 5, 1]
        cz = triPlane[i * 5: i * 5 + 5, 2]
        plt.plot(cx, cy, cz)

    plt.axis('equal')
    plt.show()


def getBdCompPlanePts(botPolygon, angles, height, transition, valCount):
    bt_center = np.mean(botPolygon, 0)
    num_botPts = np.size(botPolygon, 0)
    bt_rotM = np.eye(3)

    bt_rotM[0, 0] = np.cos(angles[2])
    bt_rotM[0, 1] = -np.sin(angles[2])
    bt_rotM[1, 0] = np.sin(angles[2])
    bt_rotM[1, 1] = np.cos(angles[2])
    botPolygon = np.transpose(
        np.matmul(bt_rotM, np.transpose(botPolygon - np.tile(bt_center, [num_botPts, 1])))) + np.tile(transition,
                                                                                                      [num_botPts,
                                                                                                       1]) + np.tile(
        bt_center, [num_botPts, 1])
    topPolygon = botPolygon + np.tile(np.array([0, 0, height]), [num_botPts, 1])
    botPolygon_exp = np.insert(botPolygon, 3, 1, axis=1)
    topPolygon_exp = np.insert(topPolygon, 3, 1, axis=1)
    triPlane = np.zeros([(num_botPts - 1) * 5, 4])
    planeBdInsRec = np.zeros([num_botPts - 1, 1], dtype=np.intc)
    planeBdInsRec.fill(valCount)
    for i in range(num_botPts - 1):
        curInd = i * 5
        triPlane[curInd] = botPolygon_exp[i]
        triPlane[curInd + 1] = botPolygon_exp[i + 1]
        triPlane[curInd + 2] = topPolygon_exp[i + 1]
        triPlane[curInd + 3] = topPolygon_exp[i]
        triPlane[curInd + 4] = botPolygon_exp[i]
    # drawBuilding_PointForm(triPlane)
    return triPlane, planeBdInsRec


def wrapIdxTest():
    mod = SourceModule("""
        #include <stdio.h>
        __global__ void doublify(int* testArr)
        {
            // int blockId = blockIdx.x + blockIdx.y * gridDim.x;
            // int idx = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
            int idx = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
            testArr[idx] = idx;
        }
    """)
    func = mod.get_function("doublify")
    return func


@jit(nopython=True)
def tileBoundingBox(bbBox):
    bbBox_shape = bbBox.shape
    totPtsNum = 0
    for i in range(bbBox_shape[0]):
        if np.abs(bbBox[i, 1] - bbBox[i, 0]) != 0 and np.abs(bbBox[i, 3] - bbBox[i, 2]) != 0:
            totPtsNum = totPtsNum + (bbBox[i, 1] - bbBox[i, 0] + 1) * (bbBox[i, 3] - bbBox[i, 2] + 1)
    tilePts = np.zeros((totPtsNum, 2))
    count = 0
    for i in range(bbBox_shape[0]):
        if np.abs(bbBox[i, 1] - bbBox[i, 0]) != 0 and np.abs(bbBox[i, 3] - bbBox[i, 2]) != 0:
            for m in range(bbBox[i, 1] - bbBox[i, 0] + 1):
                for n in range(bbBox[i, 3] - bbBox[i, 2] + 1):
                    tilePts[count, 0] = bbBox[i, 1] + m
                    tilePts[count, 1] = bbBox[i, 3] + n
                    count = count + 1
    return tilePts


def meminfo(kernel):
    shared = kernel.shared_size_bytes
    regs = kernel.num_regs
    local = kernel.local_size_bytes
    const = kernel.const_size_bytes
    mbpt = kernel.max_threads_per_block
    print(
        "=MEM=\nLocal:%d,\nShared:%d,\nRegisters:%d,\nConst:%d,\nMax Threads/B:%d" % (local, shared, regs, const, mbpt))


def get_randPtOnPlaneAndNormalDir(planeParam):
    normalDir = planeParam[0:3] / np.linalg.norm(planeParam[0:3])
    onePtOnPlane = np.zeros(4)
    onePtOnPlane[1:3] = np.array([-0.3, -0.4])
    onePtOnPlane[0] = -(planeParam[3] + planeParam[1] * onePtOnPlane[1] + planeParam[2] * onePtOnPlane[2]) / planeParam[
        0]
    onePtOnPlane[3] = 1
    if np.sum(planeParam * onePtOnPlane) > 0.01:
        print("Problematic")
    return (normalDir.astype(np.float32), onePtOnPlane.astype(np.float32))


def singleFrameRender(data_reader, renderIndex, triPlane_list, buildingIndsRecList, func_integration, funcLinParamCal, func_IntersectLine,
                             funcLineAff, func_computeNewBbox, func_depthRender, func_lineRender, func_pointCheck,
                             func_acceleratedFormPolygons):
    triPlaneVelo_tot_mulFrameList = list()
    buildingIndsRec_tot_mulFrameList = list()


    triPlane_tot = np.concatenate(triPlane_list, axis=0)
    buildingIndsRec = np.concatenate(buildingIndsRecList, axis=0)
    triPlaneVelo_tot_mulFrameList.append(np.transpose(
        np.matmul(np.matmul(data_reader.tr_oxts2velo, data_reader.trs_grid2oxts[renderIndex]),
                  np.transpose(triPlane_tot))))
    buildingIndsRec_tot_mulFrameList.append(buildingIndsRec)

    triPlaneVelo_tot_mulFrame = np.concatenate(triPlaneVelo_tot_mulFrameList, axis=0).astype(np.float32)
    normDir, onePtOnPlane = get_randPtOnPlaneAndNormalDir(data_reader.imagePlane)

    # Form the line representation form of the 3d shapes
    lines_start = np.zeros((int(np.size(triPlaneVelo_tot_mulFrame, 0) / 5 * 4), 4)).astype(np.float32)
    lines_end = np.zeros((int(np.size(triPlaneVelo_tot_mulFrame, 0) / 5 * 4), 4)).astype(np.float32)
    count = 0
    for i in range(np.size(triPlaneVelo_tot_mulFrame, 0)):
        if i % 5 == 4:
            continue
        else:
            lines_start[count] = triPlaneVelo_tot_mulFrame[i]
            lines_end[count] = triPlaneVelo_tot_mulFrame[i + 1]
            count = count + 1

    num_of_lines = np.size(lines_start, 0)
    num_of_lines = np.int32(num_of_lines)

    intersectionCondition = np.zeros(num_of_lines).astype(np.intc)
    intersectionCondition_gpu = cuda.mem_alloc(intersectionCondition.nbytes)
    lines_start_gpu = cuda.mem_alloc(lines_start.nbytes)
    lines_end_gpu = cuda.mem_alloc(lines_end.nbytes)
    normDir_gpu = cuda.mem_alloc(normDir.nbytes)
    onePtOnPlane_gpu = cuda.mem_alloc(onePtOnPlane.nbytes)
    numOfPoints_gpu = cuda.mem_alloc(num_of_lines.nbytes)


    cuda.memcpy_htod(lines_start_gpu, lines_start.flatten())
    cuda.memcpy_htod(lines_end_gpu, lines_end.flatten())
    cuda.memcpy_htod(normDir_gpu, normDir)
    cuda.memcpy_htod(onePtOnPlane_gpu, onePtOnPlane)
    cuda.memcpy_htod(numOfPoints_gpu, num_of_lines)

    func_pointCheck(lines_start_gpu, lines_end_gpu, normDir_gpu, onePtOnPlane_gpu, numOfPoints_gpu,
                    intersectionCondition_gpu,
                    block=(32, 32, 1), grid=(8, 1))
    processedLinesS = np.zeros_like(lines_start)
    processedLinesE = np.zeros_like(lines_end)
    cuda.memcpy_dtoh(intersectionCondition, intersectionCondition_gpu)
    cuda.memcpy_dtoh(processedLinesS, lines_start_gpu)
    cuda.memcpy_dtoh(processedLinesE, lines_end_gpu)

    num_of_planes = np.intc((num_of_lines / 4))
    M_inex = np.matmul(data_reader.intrinsic, data_reader.extrinsic)
    startPts_ex = np.zeros((int(np.size(lines_start, 0) / 4 * 5), int(np.size(lines_start, 1)))).astype(np.float32)
    endPts_ex = np.zeros((int(np.size(lines_end, 0) / 4 * 5), int(np.size(lines_end, 1)))).astype(np.float32)
    startPts_exMapped = np.zeros_like(startPts_ex)
    endPts_exMapped = np.zeros_like(endPts_ex)
    lineIndRec = np.zeros(int(np.size(intersectionCondition) / 4 * 5)).astype(np.intc)
    porposedImgArea = np.zeros((np.size(lineIndRec), 4)).astype(np.intc)
    bbBox = np.zeros((num_of_planes, 4)).astype(np.intc)
    planesVaalidity = np.zeros(num_of_planes, dtype=np.intc)
    planeParamRec = np.zeros((num_of_planes, 4), dtype=np.float32)

    startPts_ex_gpu = cuda.mem_alloc(startPts_ex.nbytes)
    endPts_ex_gpu = cuda.mem_alloc(endPts_ex.nbytes)
    lineIndRec_gpu = cuda.mem_alloc(lineIndRec.nbytes)
    num_of_planes_gpu = cuda.mem_alloc(num_of_planes.nbytes)
    # extrinsic_gpu = cuda.mem_alloc(data_reader.extrinsic.nbytes)
    # intrinsic_gpu = cuda.mem_alloc(data_reader.intrinsic.nbytes)
    M_inex_gpu = cuda.mem_alloc(M_inex.nbytes)
    imageSize_gpu = cuda.mem_alloc(data_reader.imageSize[renderIndex].nbytes)
    porposedImgArea_gpu = cuda.mem_alloc(porposedImgArea.nbytes)
    startPts_exMapped_gpu = cuda.mem_alloc(startPts_exMapped.nbytes)
    endPts_exMapped_gpu = cuda.mem_alloc(endPts_exMapped.nbytes)
    bbBox_gpu = cuda.mem_alloc(bbBox.nbytes)
    planesVaalidity_gpu = cuda.mem_alloc(planesVaalidity.nbytes)
    planeParamRec_gpu = cuda.mem_alloc(planeParamRec.nbytes)
    lines_startOrg_gpu = cuda.mem_alloc(lines_start.nbytes)
    lines_endOrg_gpu = cuda.mem_alloc(lines_end.nbytes)
    # cuda.memcpy_htod(extrinsic_gpu, data_reader.extrinsic.flatten())
    # cuda.memcpy_htod(intrinsic_gpu, data_reader.intrinsic.flatten())



    cuda.memcpy_htod(M_inex_gpu, M_inex.flatten())
    cuda.memcpy_htod(num_of_planes_gpu, num_of_planes)
    cuda.memcpy_htod(imageSize_gpu, data_reader.imageSize[renderIndex])
    cuda.memcpy_htod(lines_startOrg_gpu, lines_start.flatten())
    cuda.memcpy_htod(lines_endOrg_gpu, lines_end.flatten())


    func_acceleratedFormPolygons(lines_start_gpu, lines_end_gpu, num_of_planes_gpu, intersectionCondition_gpu,
                                 startPts_ex_gpu, endPts_ex_gpu,
                                 lineIndRec_gpu, M_inex_gpu, imageSize_gpu, startPts_exMapped_gpu, endPts_exMapped_gpu,
                                 bbBox_gpu,
                                 planesVaalidity_gpu, planeParamRec_gpu, lines_startOrg_gpu, lines_endOrg_gpu,
                                 block=(32, 32, 1), grid=(8, 1))
    cuda.memcpy_dtoh(lineIndRec, lineIndRec_gpu)
    cuda.memcpy_dtoh(startPts_ex, startPts_ex_gpu)
    cuda.memcpy_dtoh(endPts_ex, endPts_ex_gpu)
    cuda.memcpy_dtoh(startPts_exMapped, startPts_exMapped_gpu)
    cuda.memcpy_dtoh(endPts_exMapped, endPts_exMapped_gpu)
    cuda.memcpy_dtoh(bbBox, bbBox_gpu)
    cuda.memcpy_dtoh(planesVaalidity, planesVaalidity_gpu)
    cuda.memcpy_dtoh(planeParamRec, planeParamRec_gpu)


    calculatedValue1 = np.transpose(np.matmul(M_inex, np.transpose(startPts_ex)))
    calculatedValue2 = np.transpose(np.matmul(M_inex, np.transpose(endPts_ex)))
    selector = np.abs(startPts_ex[:, 3]) > 1e-7
    calculatedValue1_normed = np.copy(calculatedValue1)
    calculatedValue2_normed = np.copy(calculatedValue2)
    calculatedValue1_normed[selector, 0] = np.divide(calculatedValue1[selector, 0], calculatedValue1[selector, 2])
    calculatedValue1_normed[selector, 1] = np.divide(calculatedValue1[selector, 1], calculatedValue1[selector, 2])
    calculatedValue2_normed[selector, 0] = np.divide(calculatedValue2[selector, 0], calculatedValue2[selector, 2])
    calculatedValue2_normed[selector, 1] = np.divide(calculatedValue2[selector, 1], calculatedValue2[selector, 2])
    # print(np.sum(calculatedValue1_normed - startPts_exMapped))
    # print(np.sum(calculatedValue2_normed - endPts_exMapped))
    # print(planesVaalidity)
    # print(bbBox)
    imgMask = np.zeros((data_reader.imageSize[renderIndex][0], data_reader.imageSize[renderIndex][1]),
                       dtype=np.float32)
    imgMask.fill(1e20)
    imgMask_gpu = cuda.mem_alloc(imgMask.nbytes)
    cuda.memcpy_htod(imgMask_gpu, imgMask.flatten())
    selector_val = (planesVaalidity == 1)
    # print("Diff is %f" % np.sum(palneParam[planesVaalidity, :] - planeParamRec[planesVaalidity,:]))
    # print(bbBox)
    # print(planesVaalidity)
    # print(planeParamRec)
    # print("ImageSize: %d, %d" % (data_reader.imageSize[renderIndex][0], data_reader.imageSize[renderIndex][1]))
    # print("startPts_exMapped")
    # print(startPts_exMapped)
    # print("endPts_exMapped")
    # print(endPts_exMapped)


    num_of_lines_expand = np.intc(num_of_lines / 4 * 5)
    num_of_lines_expand_gpu = cuda.mem_alloc(num_of_lines_expand.nbytes)
    cuda.memcpy_htod(num_of_lines_expand_gpu, num_of_lines_expand)
    line2dParam = np.zeros((num_of_lines_expand, 3), dtype=np.float32)
    line2dParam_gpu = cuda.mem_alloc(line2dParam.nbytes)

    funcLinParamCal(startPts_exMapped_gpu, endPts_exMapped_gpu, lineIndRec_gpu, num_of_lines_expand_gpu,
                    line2dParam_gpu, block=(32, 32, 1), grid=(8, 1))
    cuda.memcpy_dtoh(line2dParam, line2dParam_gpu)



    startPts_exMappedNew = np.zeros_like(startPts_exMapped)
    endPts_exMappedNew = np.zeros_like(endPts_exMapped)
    lineIndRecNew = np.zeros_like(lineIndRec)
    startPts_exMappedNew_gpu = cuda.mem_alloc(startPts_exMappedNew.nbytes)
    endPts_exMappedNew_gpu = cuda.mem_alloc(endPts_exMappedNew.nbytes)
    lineIndRecNew_gpu = cuda.mem_alloc(lineIndRecNew.nbytes)

    func_IntersectLine(startPts_exMapped_gpu, endPts_exMapped_gpu, lineIndRec_gpu, num_of_lines_expand_gpu,
                       imageSize_gpu, startPts_exMappedNew_gpu, endPts_exMappedNew_gpu, lineIndRecNew_gpu,
                       block=(32, 32, 1), grid=(8, 1))
    cuda.memcpy_dtoh(lineIndRecNew, lineIndRecNew_gpu)
    cuda.memcpy_dtoh(startPts_exMappedNew, startPts_exMappedNew_gpu)
    cuda.memcpy_dtoh(endPts_exMappedNew, endPts_exMappedNew_gpu)
    # print(startPts_exMappedNew)

    # selector = (lineIndRec != -1)
    # line_paramVal = np.zeros((num_of_lines_expand, 3), dtype = np.float32)
    # line_paramVal[selector,0] = np.divide((startPts_exMapped[selector,1] - endPts_exMapped[selector,1]), (startPts_exMapped[selector,0] - endPts_exMapped[selector,0]))
    # line_paramVal[selector,1] = -1
    # line_paramVal[selector,2] = startPts_exMapped[selector,1] - line_paramVal[selector,0] * startPts_exMapped[selector,0]
    # print(np.sum(np.abs(line_paramVal[selector,2] - line2dParam[selector,2])))
    # a = 1
    # selector = np.zeros(np.size(lineIndRec), dtype = bool)
    # selector[21] = True
    # k = 26
    # print(startPts_exMapped[26,:])
    # print(line2dParam[26,:])
    """
    for k in range(num_of_lines_expand):
        if selector[k]:
            a = np.abs(
                startPts_exMapped[k, 0] * line_paramVal[k, 0] + startPts_exMapped[k, 1] * line_paramVal[k, 1] + line_paramVal[
                    k, 2])
            b = np.abs(
                startPts_exMapped[k, 0] * line2dParam[k, 0] + startPts_exMapped[k, 1] * line2dParam[k, 1] + line2dParam[
                    k, 2])
            print(k, a, b)
    valSum1 = np.sum(np.abs(startPts_exMapped[selector,0] * line2dParam[selector,0] + startPts_exMapped[selector,1] * line2dParam[selector,1] + line2dParam[selector,2]))
    valSum2 = np.sum(np.abs(
        endPts_exMapped[selector, 0] * line2dParam[selector, 0] + endPts_exMapped[selector, 1] * line2dParam[
            selector, 1] + line2dParam[selector, 2]))
    """


    invM_inex_gpu = cuda.mem_alloc(M_inex.nbytes)
    invM = gpuWrappedFunc.naiveMatrixInverse(M_inex)
    cuda.memcpy_htod(invM_inex_gpu, invM.flatten())

    aMinv = np.zeros_like(planeParamRec)
    aMinv_gpu = cuda.mem_alloc(aMinv.nbytes)


    funcLineAff(num_of_planes_gpu, invM_inex_gpu, planeParamRec_gpu, planesVaalidity_gpu, aMinv_gpu, block=(32, 32, 1),
                grid=(1, 1))
    cuda.memcpy_dtoh(aMinv, aMinv_gpu)



    newBbox = np.zeros_like(bbBox)
    planesVaalidity_new = np.zeros_like(planesVaalidity)
    newBbox_gpu = cuda.mem_alloc(newBbox.nbytes)
    planesVaalidity_new_gpu = cuda.mem_alloc(planesVaalidity_new.nbytes)


    func_computeNewBbox(startPts_exMappedNew_gpu, endPts_exMappedNew_gpu, lineIndRecNew_gpu, num_of_planes_gpu,
                        imageSize_gpu, newBbox_gpu, planesVaalidity_new_gpu, block=(32, 32, 1), grid=(1, 1))
    cuda.memcpy_dtoh(newBbox, newBbox_gpu)
    cuda.memcpy_dtoh(planesVaalidity_new, planesVaalidity_new_gpu)
    cuda.memcpy_dtoh(lineIndRecNew, lineIndRecNew_gpu)
    """
    image = mpimg.imread(data_reader.rgbFilePaths[renderIndex])
    # plt.axis([0, 1226, 0, 370])
    plt.axis([0, 1226, 370, 0])
    plt.imshow(image)
    for i in range(num_of_planes):
        for j in range(5):
            if (lineIndRecNew[i * 5 + j] != -1):
                x = np.array((startPts_exMappedNew[i * 5 + j, 0], endPts_exMappedNew[i * 5 + j, 0]))
                y = np.array((startPts_exMappedNew[i * 5 + j, 1], endPts_exMappedNew[i * 5 + j, 1]))
                plt.plot(x, y, 'b', linewidth=1)
            tmpBox = newBbox[i, :]
            pts1 = [[tmpBox[0], tmpBox[2]]]
            pts2 = [[tmpBox[0], tmpBox[3]]]
            pts3 = [[tmpBox[1], tmpBox[3]]]
            pts4 = [[tmpBox[1], tmpBox[2]]]
            pts = np.concatenate((pts1, pts2, pts3, pts4, pts1), axis=0)
            plt.plot(pts[:, 0], pts[:, 1], 'r', linewidth=1.0)
    plt.show()
    for i in range(num_of_planes):
        for j in range(5):
            if(lineIndRecNew[i * 5 + j] != -1):
                x = np.array((startPts_exMappedNew[i * 5 + j, 0], endPts_exMappedNew[i * 5 + j, 0]))
                y = np.array((startPts_exMappedNew[i * 5 + j, 1], endPts_exMappedNew[i * 5 + j, 1]))
                plt.plot(x, y,'b')
            tmpBox = newBbox[i,:]
            pts1 = [[tmpBox[0], tmpBox[2]]]
            pts2 = [[tmpBox[0], tmpBox[3]]]
            pts3 = [[tmpBox[1], tmpBox[3]]]
            pts4 = [[tmpBox[1], tmpBox[2]]]
            pts = np.concatenate((pts1, pts2, pts3, pts4, pts1), axis = 0)
            plt.plot(pts[:,0], pts[:,1], 'r')
    plt.show()
    """

    func_depthRender(num_of_planes_gpu, aMinv_gpu, imageSize_gpu, startPts_exMapped_gpu, endPts_exMapped_gpu, bbBox_gpu,
                     planesVaalidity_gpu, line2dParam_gpu, lineIndRec_gpu, imgMask_gpu, block=(32, 32, 1), grid=(10, 1))
    # func_depthRender(num_of_planes_gpu, aMinv_gpu, imageSize_gpu, startPts_exMapped_gpu, endPts_exMapped_gpu, newBbox_gpu,
    #       planesVaalidity_new_gpu, line2dParam_gpu, lineIndRec_gpu, imgMask_gpu, block=(32, 32, 1), grid=(10, 1))
    cuda.memcpy_dtoh(imgMask, imgMask_gpu)
    selector = imgMask > 1e10
    imgMask[selector] = 0


    """
    fig = plt.figure()
    plt.imshow(imgMask, cmap='Greys')
    plt.axis([0, 1226, 370, 0])
    fig.show()
    """

    imgLineMask = np.zeros_like(imgMask)
    imgLineMask_gpu = cuda.mem_alloc(imgLineMask.nbytes)

    func_lineRender(num_of_lines_expand_gpu, aMinv_gpu, imageSize_gpu, startPts_exMappedNew_gpu, endPts_exMappedNew_gpu,
                    planesVaalidity_new_gpu, line2dParam_gpu, lineIndRecNew_gpu, imgMask_gpu, imgLineMask_gpu,
                    block=(32, 32, 1), grid=(10, 1))
    cuda.memcpy_dtoh(imgLineMask, imgLineMask_gpu)

    """
    fig = plt.figure()
    imgLineMask_norm = imgLineMask / np.max(imgLineMask)
    plt.imshow(imgLineMask_norm, cmap='Greys')
    plt.axis([0, 1226, 370, 0])
    fig.show()
    for idx in range(num_of_lines_expand):
        if lineIndRecNew[idx] != -1:
            ptsx = [startPts_exMappedNew[idx, 0], endPts_exMappedNew[idx, 0]]
            ptsy = [startPts_exMappedNew[idx, 1], endPts_exMappedNew[idx, 1]]
            plt.plot(ptsx, ptsy, 'r')
    fig.show()
    """

    image = mpimg.imread(data_reader.rgbFilePaths[renderIndex])
    r = np.copy(image[:, :, 0])
    g = np.copy(image[:, :, 1])
    b = np.copy(image[:, :, 2])
    r_gpu = cuda.mem_alloc(r.nbytes)
    g_gpu = cuda.mem_alloc(g.nbytes)
    b_gpu = cuda.mem_alloc(b.nbytes)
    depthMap = np.zeros_like(imgMask)


    cuda.memcpy_htod(r_gpu, r.flatten())
    cuda.memcpy_htod(g_gpu, g.flatten())
    cuda.memcpy_htod(b_gpu, b.flatten())

    func_integration(imageSize_gpu, imgMask_gpu, imgLineMask_gpu, r_gpu, g_gpu, b_gpu, block=(32, 32, 1), grid=(10, 1))
    cuda.memcpy_dtoh(r, r_gpu)
    cuda.memcpy_dtoh(g, g_gpu)
    cuda.memcpy_dtoh(b, b_gpu)
    cuda.memcpy_dtoh(depthMap, imgMask_gpu)
    integratedImage = np.stack((r, g, b), axis=2)

    # fig = plt.figure()
    # plt.imshow(integratedImage)
    # fig.show()
    return integratedImage, depthMap


def renderAndSaveImage(data_reader, renderIndex, svPath, func_integration, funcLinParamCal, func_IntersectLine, funcLineAff, func_computeNewBbox, func_depthRender, func_lineRender, func_pointCheck, func_acceleratedFormPolygons):
    triPlane_list = list()
    planeBdInsRecList = list()
    valCount = 0
    for idx, bdCompInd in enumerate(data_reader.buildingComp):
        # if data_reader.buildingComp[bdCompInd].visibility[renderIndex]:
        if data_reader.buildingComp[bdCompInd].visibility[renderIndex]:
            curBotPolygon = data_reader.buildingComp[bdCompInd].botPolygon
            curAngle = data_reader.buildingComp[bdCompInd].angles
            curHeight = data_reader.buildingComp[bdCompInd].height
            curTransition = data_reader.buildingComp[bdCompInd].transition
            re, planeBdInsRectmp = getBdCompPlanePts(curBotPolygon, curAngle, curHeight, curTransition, valCount)
            triPlane_list.append(re)
            planeBdInsRecList.append(planeBdInsRectmp)
            valCount = valCount + 1
    if len(triPlane_list) == 0:
        return
    triPlane_tot = np.concatenate(triPlane_list, axis=0)
    planeBdInsRec = np.concatenate(planeBdInsRecList, axis=0)
    triPlaneVelo_tot_mulFrame = np.transpose(
        np.matmul(np.matmul(data_reader.tr_oxts2velo, data_reader.trs_grid2oxts[renderIndex]),
                  np.transpose(triPlane_tot)))


    normDir, onePtOnPlane = get_randPtOnPlaneAndNormalDir(data_reader.imagePlane)


    # Form the line representation form of the 3d shapes
    lines_start = np.zeros((int(np.size(triPlaneVelo_tot_mulFrame, 0) / 5 * 4), 4)).astype(np.float32)
    lines_end = np.zeros((int(np.size(triPlaneVelo_tot_mulFrame, 0) / 5 * 4), 4)).astype(np.float32)
    count = 0
    for i in range(np.size(triPlaneVelo_tot_mulFrame, 0)):
        if i % 5 == 4:
            continue
        else:
            lines_start[count] = triPlaneVelo_tot_mulFrame[i]
            lines_end[count] = triPlaneVelo_tot_mulFrame[i + 1]
            count = count + 1

    num_of_lines = np.size(lines_start, 0)
    num_of_lines = np.int32(num_of_lines)

    intersectionCondition = np.zeros(num_of_lines).astype(np.intc)
    intersectionCondition_gpu = cuda.mem_alloc(intersectionCondition.nbytes)
    lines_start_gpu = cuda.mem_alloc(lines_start.nbytes)
    lines_end_gpu = cuda.mem_alloc(lines_end.nbytes)
    normDir_gpu = cuda.mem_alloc(normDir.nbytes)
    onePtOnPlane_gpu = cuda.mem_alloc(onePtOnPlane.nbytes)
    numOfPoints_gpu = cuda.mem_alloc(num_of_lines.nbytes)

    cuda.memcpy_htod(lines_start_gpu, lines_start.flatten())
    cuda.memcpy_htod(lines_end_gpu, lines_end.flatten())
    cuda.memcpy_htod(normDir_gpu, normDir)
    cuda.memcpy_htod(onePtOnPlane_gpu, onePtOnPlane)
    cuda.memcpy_htod(numOfPoints_gpu, num_of_lines)


    func_pointCheck(lines_start_gpu, lines_end_gpu, normDir_gpu, onePtOnPlane_gpu, numOfPoints_gpu, intersectionCondition_gpu,
          block=(32, 32, 1), grid=(8, 1))
    processedLinesS = np.zeros_like(lines_start)
    processedLinesE = np.zeros_like(lines_end)
    cuda.memcpy_dtoh(intersectionCondition, intersectionCondition_gpu)
    cuda.memcpy_dtoh(processedLinesS, lines_start_gpu)
    cuda.memcpy_dtoh(processedLinesE, lines_end_gpu)

    num_of_planes = np.intc((num_of_lines / 4))
    M_inex = np.matmul(data_reader.intrinsic, data_reader.extrinsic)
    startPts_ex = np.zeros((int(np.size(lines_start, 0) / 4 * 5), int(np.size(lines_start, 1)))).astype(np.float32)
    endPts_ex = np.zeros((int(np.size(lines_end, 0) / 4 * 5), int(np.size(lines_end, 1)))).astype(np.float32)
    startPts_exMapped = np.zeros_like(startPts_ex)
    endPts_exMapped = np.zeros_like(endPts_ex)
    lineIndRec = np.zeros(int(np.size(intersectionCondition) / 4 * 5)).astype(np.intc)
    bbBox = np.zeros((num_of_planes, 4)).astype(np.intc)
    planesVaalidity = np.zeros(num_of_planes, dtype=np.intc)
    planeParamRec = np.zeros((num_of_planes, 4), dtype=np.float32)

    startPts_ex_gpu = cuda.mem_alloc(startPts_ex.nbytes)
    endPts_ex_gpu = cuda.mem_alloc(endPts_ex.nbytes)
    lineIndRec_gpu = cuda.mem_alloc(lineIndRec.nbytes)
    num_of_planes_gpu = cuda.mem_alloc(num_of_planes.nbytes)
    # extrinsic_gpu = cuda.mem_alloc(data_reader.extrinsic.nbytes)
    # intrinsic_gpu = cuda.mem_alloc(data_reader.intrinsic.nbytes)
    M_inex_gpu = cuda.mem_alloc(M_inex.nbytes)
    imageSize_gpu = cuda.mem_alloc(data_reader.imageSize[renderIndex].nbytes)
    startPts_exMapped_gpu = cuda.mem_alloc(startPts_exMapped.nbytes)
    endPts_exMapped_gpu = cuda.mem_alloc(endPts_exMapped.nbytes)
    bbBox_gpu = cuda.mem_alloc(bbBox.nbytes)
    planesVaalidity_gpu = cuda.mem_alloc(planesVaalidity.nbytes)
    planeParamRec_gpu = cuda.mem_alloc(planeParamRec.nbytes)
    lines_startOrg_gpu = cuda.mem_alloc(lines_start.nbytes)
    lines_endOrg_gpu = cuda.mem_alloc(lines_end.nbytes)
    # cuda.memcpy_htod(extrinsic_gpu, data_reader.extrinsic.flatten())
    # cuda.memcpy_htod(intrinsic_gpu, data_reader.intrinsic.flatten())

    cuda.memcpy_htod(M_inex_gpu, M_inex.flatten())
    cuda.memcpy_htod(num_of_planes_gpu, num_of_planes)
    cuda.memcpy_htod(imageSize_gpu, data_reader.imageSize[renderIndex])
    cuda.memcpy_htod(lines_startOrg_gpu, lines_start.flatten())
    cuda.memcpy_htod(lines_endOrg_gpu, lines_end.flatten())

    func_acceleratedFormPolygons(lines_start_gpu, lines_end_gpu, num_of_planes_gpu, intersectionCondition_gpu, startPts_ex_gpu, endPts_ex_gpu,
          lineIndRec_gpu, M_inex_gpu, imageSize_gpu, startPts_exMapped_gpu, endPts_exMapped_gpu, bbBox_gpu,
          planesVaalidity_gpu, planeParamRec_gpu, lines_startOrg_gpu, lines_endOrg_gpu, block=(32, 32, 1), grid=(8, 1))
    cuda.memcpy_dtoh(lineIndRec, lineIndRec_gpu)
    cuda.memcpy_dtoh(startPts_ex, startPts_ex_gpu)
    cuda.memcpy_dtoh(endPts_ex, endPts_ex_gpu)
    cuda.memcpy_dtoh(startPts_exMapped, startPts_exMapped_gpu)
    cuda.memcpy_dtoh(endPts_exMapped, endPts_exMapped_gpu)
    cuda.memcpy_dtoh(bbBox, bbBox_gpu)
    cuda.memcpy_dtoh(planesVaalidity, planesVaalidity_gpu)
    cuda.memcpy_dtoh(planeParamRec, planeParamRec_gpu)

    imgMask = np.zeros((data_reader.imageSize[renderIndex][0], data_reader.imageSize[renderIndex][1]),
                       dtype=np.float32)
    imgMask.fill(1e20)
    imgMask_gpu = cuda.mem_alloc(imgMask.nbytes)
    cuda.memcpy_htod(imgMask_gpu, imgMask.flatten())

    num_of_lines_expand = np.intc(num_of_lines / 4 * 5)
    num_of_lines_expand_gpu = cuda.mem_alloc(num_of_lines_expand.nbytes)
    cuda.memcpy_htod(num_of_lines_expand_gpu, num_of_lines_expand)
    line2dParam = np.zeros((num_of_lines_expand, 3), dtype=np.float32)
    line2dParam_gpu = cuda.mem_alloc(line2dParam.nbytes)


    funcLinParamCal(startPts_exMapped_gpu, endPts_exMapped_gpu, lineIndRec_gpu, num_of_lines_expand_gpu,
                    line2dParam_gpu, block=(32, 32, 1), grid=(8, 1))
    cuda.memcpy_dtoh(line2dParam, line2dParam_gpu)

    startPts_exMappedNew = np.zeros_like(startPts_exMapped)
    endPts_exMappedNew = np.zeros_like(endPts_exMapped)
    lineIndRecNew = np.zeros_like(lineIndRec)
    startPts_exMappedNew_gpu = cuda.mem_alloc(startPts_exMappedNew.nbytes)
    endPts_exMappedNew_gpu = cuda.mem_alloc(endPts_exMappedNew.nbytes)
    lineIndRecNew_gpu = cuda.mem_alloc(lineIndRecNew.nbytes)

    func_IntersectLine(startPts_exMapped_gpu, endPts_exMapped_gpu, lineIndRec_gpu, num_of_lines_expand_gpu,
                       imageSize_gpu, startPts_exMappedNew_gpu, endPts_exMappedNew_gpu, lineIndRecNew_gpu,
                       block=(32, 32, 1), grid=(8, 1))
    cuda.memcpy_dtoh(lineIndRecNew, lineIndRecNew_gpu)
    cuda.memcpy_dtoh(startPts_exMappedNew, startPts_exMappedNew_gpu)
    cuda.memcpy_dtoh(endPts_exMappedNew, endPts_exMappedNew_gpu)
    # print(startPts_exMappedNew)

    invM_inex_gpu = cuda.mem_alloc(M_inex.nbytes)
    invM = gpuWrappedFunc.naiveMatrixInverse(M_inex)
    cuda.memcpy_htod(invM_inex_gpu, invM.flatten())

    aMinv = np.zeros_like(planeParamRec)
    aMinv_gpu = cuda.mem_alloc(aMinv.nbytes)


    funcLineAff(num_of_planes_gpu, invM_inex_gpu, planeParamRec_gpu, planesVaalidity_gpu, aMinv_gpu, block=(32, 32, 1),
                grid=(1, 1))
    cuda.memcpy_dtoh(aMinv, aMinv_gpu)


    newBbox = np.zeros_like(bbBox)
    planesVaalidity_new = np.zeros_like(planesVaalidity)
    newBbox_gpu = cuda.mem_alloc(newBbox.nbytes)
    planesVaalidity_new_gpu = cuda.mem_alloc(planesVaalidity_new.nbytes)

    func_computeNewBbox(startPts_exMappedNew_gpu, endPts_exMappedNew_gpu, lineIndRecNew_gpu, num_of_planes_gpu,
                        imageSize_gpu, newBbox_gpu, planesVaalidity_new_gpu, block=(32, 32, 1), grid=(1, 1))
    cuda.memcpy_dtoh(newBbox, newBbox_gpu)
    cuda.memcpy_dtoh(planesVaalidity_new, planesVaalidity_new_gpu)
    cuda.memcpy_dtoh(lineIndRecNew, lineIndRecNew_gpu)





    # Added visibility check
    planeCoveredRec = np.zeros((valCount, 1), dtype = np.intc)
    planeBdInsRec_gpu = cuda.mem_alloc(planeBdInsRec.nbytes)
    planeCoveredRec_gpu = cuda.mem_alloc(planeCoveredRec.nbytes)

    cuda.memcpy_htod(planeBdInsRec_gpu, planeBdInsRec)
    func_depthRender(num_of_planes_gpu, aMinv_gpu, imageSize_gpu, startPts_exMapped_gpu, endPts_exMapped_gpu, bbBox_gpu,
          planesVaalidity_gpu, line2dParam_gpu, lineIndRec_gpu, imgMask_gpu, planeBdInsRec_gpu, planeCoveredRec_gpu, block=(32, 32, 1), grid=(10, 1))
    cuda.memcpy_dtoh(imgMask, imgMask_gpu)
    cuda.memcpy_dtoh(planeCoveredRec, planeCoveredRec_gpu)
    if np.sum(planeCoveredRec) != np.size(planeCoveredRec):
        print("Warning, sequence %s, frame %d, visibility wrong" % (data_reader.sequenceName, renderIndex))
        # raise Warning('Visibility value is wrong')

    imgLineMask = np.zeros_like(imgMask)
    imgLineMask_gpu = cuda.mem_alloc(imgLineMask.nbytes)

    func_lineRender(num_of_lines_expand_gpu, aMinv_gpu, imageSize_gpu, startPts_exMappedNew_gpu, endPts_exMappedNew_gpu,
                    planesVaalidity_new_gpu, line2dParam_gpu, lineIndRecNew_gpu, imgMask_gpu, imgLineMask_gpu,
                    block=(32, 32, 1), grid=(10, 1))
    cuda.memcpy_dtoh(imgLineMask, imgLineMask_gpu)




    image = mpimg.imread(data_reader.rgbFilePaths[renderIndex])
    r = np.copy(image[:, :, 0])
    g = np.copy(image[:, :, 1])
    b = np.copy(image[:, :, 2])
    r_gpu = cuda.mem_alloc(r.nbytes)
    g_gpu = cuda.mem_alloc(g.nbytes)
    b_gpu = cuda.mem_alloc(b.nbytes)
    depthMap = np.zeros_like(imgMask)

    cuda.memcpy_htod(r_gpu, r.flatten())
    cuda.memcpy_htod(g_gpu, g.flatten())
    cuda.memcpy_htod(b_gpu, b.flatten())


    func_integration(imageSize_gpu, imgMask_gpu, imgLineMask_gpu, r_gpu, g_gpu, b_gpu, block=(32, 32, 1), grid=(10, 1))
    cuda.memcpy_dtoh(r, r_gpu)
    cuda.memcpy_dtoh(g, g_gpu)
    cuda.memcpy_dtoh(b, b_gpu)
    cuda.memcpy_dtoh(depthMap, imgMask_gpu)
    integratedImage = np.stack((r, g, b), axis=2)


    rgb_toSv = Image.fromarray((integratedImage * 255).astype('uint8'))
    rgb_toSv.save(svPath + "/rgb_" + str(renderIndex), "JPEG")

class DataProvider:
    def __init__(self):
        sequenceName_set = [
            '2011_09_30_drive_0018_sync',
            '2011_09_26_drive_0096_sync',
            '2011_09_26_drive_0104_sync',
            '2011_09_30_drive_0033_sync',
            '2011_09_26_drive_0117_sync',
            '2011_10_03_drive_0034_sync',
            '2011_10_03_drive_0027_sync',
        ]
        self.data_reader = dict()
        for sequence in sequenceName_set:
            self.data_reader[sequence] = dataReader.dataReader(sequence)
    def getReader(self, sequenceName):
        return self.data_reader[sequenceName]

class videoSequence:
    def __init__(self, renderedRgb, renderedDepth, gtTransition, gtVisibility, isValid):
        self.imgnum = len(renderedRgb)
        self.rgb = renderedRgb
        self.depth = renderedDepth
        self.gtTransition = gtTransition
        self.gtVisibility = gtVisibility
        self.isValid = isValid


class GPURender:
    def __init__(self):
        self.func_integration = gpuWrappedFunc.wrapIntegration()
        self.funcLinParamCal = gpuWrappedFunc.lineParamCalculator()
        self.func_IntersectLine = gpuWrappedFunc.wrapIntersectLineseg()
        self.funcLineAff = gpuWrappedFunc.wraplineAff()
        self.func_computeNewBbox = gpuWrappedFunc.wrapComputeNewBox()
        self.func_depthRender = gpuWrappedFunc.wrapDepthRender()
        self.func_lineRender = gpuWrappedFunc.wrapLineRender()
        self.func_pointCheck = gpuWrappedFunc.wrapPointCheck()
        self.func_acceleratedFormPolygons = gpuWrappedFunc.wrapAccleratedFormPolygons()
    def renderSingleImage(self, data_reader, bdCompInd):

        bdComp = data_reader.buildingComp[bdCompInd]
        curBotPolygon = bdComp.botPolygon
        curAngle = bdComp.angles
        curHeight = bdComp.height
        curTransition = bdComp.transition * 0
        re = getBdCompPlanePts(curBotPolygon, curAngle, curHeight, curTransition)
        triPlane_list = list()
        buildingIndsRecList = list()
        triPlane_list.append(re)
        bdIndicesTemp = np.empty(np.size(re, 0))
        bdIndicesTemp.fill(bdCompInd)
        buildingIndsRecList.append(bdIndicesTemp)

        rangeVisibility_array = data_reader.buildingComp[bdCompInd].range_visibilityArray[0 : len(data_reader.trs_grid2oxts)]
        renderedRgb = dict()
        renderedDepth = dict()
        for idx, isValid in enumerate(rangeVisibility_array):
            if isValid == 1:
                renderIndex = idx
                integratedImage, depthMap = singleFrameRender(data_reader, renderIndex, triPlane_list, buildingIndsRecList, self.func_integration, self.funcLinParamCal, self.func_IntersectLine,
                                         self.funcLineAff, self.func_computeNewBbox, self.func_depthRender, self.func_lineRender,
                                         self.func_pointCheck, self.func_acceleratedFormPolygons)
                renderedRgb[idx] = integratedImage
                renderedDepth[idx] = depthMap
        return videoSequence(renderedRgb, renderedDepth, curTransition, bdComp.visibility, sum(bdComp.visibility) > 0)


    def renderAndSaveSingleImage(self, data_reader, renderIndex, svPath):
        renderAndSaveImage(data_reader, renderIndex, svPath, self.func_integration, self.funcLinParamCal, self.func_IntersectLine,
                                         self.funcLineAff, self.func_computeNewBbox, self.func_depthRender, self.func_lineRender,
                                         self.func_pointCheck, self.func_acceleratedFormPolygons)
        return

class tt_struct:
    def __init__(self):
        self.testSeq = ['2011_09_26_drive_0117_sync']
        self.trainSeq = [
            '2011_09_30_drive_0018_sync',
            '2011_09_26_drive_0096_sync',
            '2011_09_26_drive_0104_sync',
            '2011_09_30_drive_0033_sync',
        ]
        self.dataprovider = DataProvider()
        self.gpurender = GPURender()

    def getRandomTrainSample(self):
        sequenceid = np.random.randint(0,len(self.trainSeq))
        rndDatareder = self.dataprovider.getReader(self.trainSeq[sequenceid])
        rndBdid = random.choice(rndDatareder.valueList)
        return self.gpurender.renderSingleImage(rndDatareder, rndBdid)

    def getRandomTestTransSample(self):
        while(True):
            sequenceid = np.random.randint(0,len(self.testSeq))
            rndDatareder = self.dataprovider.getReader(self.testSeq[sequenceid])
            rndBdid = random.choice(rndDatareder.valueList)
            if np.sum(rndDatareder.buildingComp[rndBdid].visibility) > 0:
                return self.gpurender.renderSingleImage(rndDatareder, rndBdid)

    def getRandomTestVisiSample(self):
        sequenceid = np.random.randint(0,len(self.testSeq))
        rndDatareder = self.dataprovider.getReader(self.testSeq[sequenceid])
        rndBdid = random.choice(rndDatareder.valueList)
        return self.gpurender.renderSingleImage(rndDatareder, rndBdid)

    def getSequentialTest(self, seqName, idx):
        tmpReader = self.dataprovider.getReader(seqName)
        return self.gpurender.renderSingleImage(tmpReader, tmpReader.valueList[idx])
    def changeDataReaderContent(self, seqName, seqIdx, transition, visibility):
        tmpReader = self.dataprovider.getReader(seqName)
        visibility_expand = tmpReader.buildingComp[tmpReader.valueList[seqIdx]].range_visibilityArray
        selector1 = visibility_expand == 1
        selector1[selector1] = np.logical_and(selector1[selector1], visibility)
        visibility_tmp = np.zeros_like(visibility_expand)
        visibility_tmp[selector1] = 1
        tmpReader.buildingComp[tmpReader.valueList[seqIdx]].visibility = visibility_tmp
        tmpReader.buildingComp[tmpReader.valueList[seqIdx]].transition = transition
        return
    def changeDataReaderTransition(self, seqName, bdIdx, transition):
        tmpReader = self.dataprovider.getReader(seqName)
        tmpReader.buildingComp[bdIdx].transition = transition
        return
    def renderSepecificSequence(self, seqName):
        prefixPath = "/media/shengjie/other/KITTI_scene_understanding/python_code/BuildingLocalization/resultsVisualization"
        svPath_folder = os.path.join(prefixPath, seqName)
        try:
            os.mkdir(svPath_folder)
        except OSError:
            pass
        tmpReader = self.dataprovider.getReader(seqName)
        for i in range(len(tmpReader.trs_grid2oxts)):
            self.gpurender.renderAndSaveSingleImage(tmpReader, i, svPath_folder)
        print("Finish Render Seq: %s" % seqName)
    def getSpecificSeqBdNum(self, seqName):
        tmpReader = self.dataprovider.getReader(seqName)
        return len(tmpReader.buildingComp)
    def replaceSavedDataProvider(self):
        self.dataprovider = pickle.load(open(
            "/media/shengjie/other/KITTI_scene_understanding/python_code/BuildingLocalization/savedDataProvider/dataprovider.p",
            "rb"))

"""
tt_struct_test = tt_struct()
trainSeq = [
            '2011_09_30_drive_0018_sync',
            '2011_09_26_drive_0096_sync',
            '2011_09_26_drive_0104_sync',
            '2011_09_30_drive_0033_sync',
            '2011_09_26_drive_0117_sync',
        ]
for content in trainSeq:
    tt_struct_test.renderSepecificSequence(content)
    
"""


"""
tt_struct_test = tt_struct()
for i in range(10):
    a = tt_struct_test.getRandomTrainSample()
    b = tt_struct_test.getRandomTestTransSample()
    c = tt_struct_test.getRandomTestVisiSample()
    print("The %dth it" % i)
"""


"""
sequenceName = '2011_09_26_drive_0117_sync'
dataprovider = DataProvider()
gpurender = GPURender()
videoSequence = gpurender.renderSingleImage(dataprovider.getReader(sequenceName), 1726)
"""




"""
plt.figure()
for rgbIdx in renderedRgb:
    plt.imshow(renderedRgb[rgbIdx])
    plt.show()
for depthIdx in renderedDepth:
    plt.imshow(renderedDepth[depthIdx], cmap = 'gray')
    plt.show()
"""