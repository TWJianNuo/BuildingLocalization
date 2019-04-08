import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
from PIL import Image
import dataReader
from numba import jit
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D


def lineParamCalculator():
    mod = SourceModule("""
        #include <stdio.h>
        __global__ void doublify(float* mappedStartPoint, float* mappedEndPoint, int* lineIndRec, int* lineNum_ex, float* line2dParam)
        {
            int blockId = blockIdx.x + blockIdx.y * gridDim.x;
            int idx = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
            // if(idx == 26){
            for(int i = idx; i < *lineNum_ex; i += gridDim.x * gridDim.y * blockDim.x * blockDim.y){
                if (lineIndRec[idx] != -1){
                    int basePtIndex = i * 4;
                    // float lineParam[3];
                    
                    if(abs(mappedStartPoint[basePtIndex] - mappedEndPoint[basePtIndex]) < 1e-5){
                        line2dParam[idx*3 + 1] = 0;
                    }
                    else{
                        line2dParam[idx*3 + 0] = (mappedStartPoint[basePtIndex + 1] - mappedEndPoint[basePtIndex + 1]) / (mappedStartPoint[basePtIndex] - mappedEndPoint[basePtIndex]);
                        if(abs(line2dParam[idx*3 + 0]) > 1e10){
                            line2dParam[idx*3 + 0] = 1;
                            line2dParam[idx*3 + 1] = 0;
                        }
                    }
                    line2dParam[idx*3 + 1] = -1;
                    line2dParam[idx*3 + 2] = -line2dParam[idx*3 + 1] * mappedStartPoint[basePtIndex + 1] - line2dParam[idx*3 + 0] * mappedStartPoint[basePtIndex];
                    // printf("y:%.10f\\n", mappedStartPoint[basePtIndex + 1]);
                    // printf("sT:%.10f\\n", -line2dParam[idx*3 + 0] * mappedStartPoint[basePtIndex]);
                    // printf("fV:%.10f\\n", line2dParam[idx*3 + 2]);
                    // line2dParam[idx*3 + 0] = lineParam[0];
                    // line2dParam[idx*3 + 1] = lineParam[1];
                    // line2dParam[idx*3 + 2] = lineParam[2];
                    float testVal = abs(mappedStartPoint[basePtIndex] * line2dParam[idx*3 + 0] + mappedStartPoint[basePtIndex+1] * line2dParam[idx*3 + 1] + line2dParam[idx*3 + 2]); // + abs(mappedEndPoint[basePtIndex] * line2dParam[idx*3 + 0] + mappedEndPoint[basePtIndex+1] * line2dParam[idx*3 + 1] + line2dParam[idx*3 + 2]);
                    // printf("X val diff: %f\\n", mappedStartPoint[basePtIndex] - mappedEndPoint[basePtIndex]);
                    // printf("Y val diff: %f\\n", mappedStartPoint[basePtIndex + 1] - mappedEndPoint[basePtIndex + 1]);
                    // printf("First term: %f\\n", mappedStartPoint[basePtIndex] * line2dParam[idx*3 + 0]);
                    // printf("Second term: %f\\n", mappedStartPoint[basePtIndex+1] * line2dParam[idx*3 + 1]);
                    // printf("Third term: %f\\n", line2dParam[idx*3 + 2]);
                    // printf("error: %f\\n", testVal);
                    // printf("x:%f, y:%f, px:%f, py:%f, pz:%f\\n", mappedStartPoint[basePtIndex], mappedStartPoint[basePtIndex+1], line2dParam[idx*3 + 0], line2dParam[idx*3 + 1], line2dParam[idx*3 + 2]);
                    if (testVal > 1e-3){
                        printf("Fail: %f\\n", testVal);
                    }
                }
            }
            // }
            // printf("lineNum_ex is %d\\n", *lineNum_ex);
        }
    """)
    func = mod.get_function("doublify")
    return func

def wrapDepthRender():
    mod = SourceModule("""
        #include <stdio.h>
        __global__ void doublify(int* numOfPlanes, float* aMinv, int* imageSize, float* mappedStartPoint, float* mappedEndPoint, int* bbBox, int* planesValidity, float*line2dParam, int* lineValidity, float* imgMask, int* planeBdInsRec, int* planeCoveredRec)
        {
            int blockId = blockIdx.x + blockIdx.y * gridDim.x;
            int idx = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
            float largNum = 3.4e10;
            for(int i = idx; i < imageSize[0] * imageSize[1]; i += gridDim.x * gridDim.y * blockDim.x * blockDim.y){
                int imgy = i / imageSize[1];
                int imgx = i - imgy * imageSize[1];
                // if ((imgy == 187) && (imgx == 450)){
                float imgyf = imgy;
                float imgxf = imgx;
                //if((imgy == 182) && (imgx == 591)){
                for(int k = 0; k < *numOfPlanes; k++){
                    if(planesValidity[k] != 0){
                        int bboxBaseInd = k * 4;
                        if ( (imgx >= bbBox[bboxBaseInd + 0]) && (imgx <= bbBox[bboxBaseInd + 1]) && (imgy >= bbBox[bboxBaseInd + 2]) && (imgy <= bbBox[bboxBaseInd + 3]) ){
                            int baseLineidx = k * 5;
                            int intersectionNum = 0;
                            // float minDepth = 1e10;
                            for(int p = 0; p < 5; p++){
                                if(lineValidity[baseLineidx + p] != -1){
                                    float t1 = mappedStartPoint[k * 4 * 5 + p * 4 + 1];
                                    float t2 = mappedEndPoint[k * 4 * 5 + p * 4 + 1];
                                    if (abs(t1 - round(t1)) < 1e-5){
                                        t1 = t1 + 1e-5;
                                    }
                                    if (abs(t2 - round(t2)) < 1e-5){
                                        t2 = t2 + 1e-5;
                                    }
                                    bool jud1 = (t1 - imgyf) * (t2 - imgyf) < 0;
                                    // bool jud1 = (mappedStartPoint[k * 4 * 5 + p * 4 + 1] - imgyf) * (mappedEndPoint[k * 4 * 5 + p * 4 + 1] - imgyf) < 0;
                                    // bool jud1p1 = ((mappedStartPoint[k * 4 * 5 + p * 4 + 1] - imgyf) > -1e-15) && ((mappedEndPoint[k * 4 * 5 + p * 4 + 1] - imgyf) < 1e-15);
                                    // bool jud1p2 = ((mappedStartPoint[k * 4 * 5 + p * 4 + 1] - imgyf) < 1e-15) && ((mappedEndPoint[k * 4 * 5 + p * 4 + 1] - imgyf) > -1e-15);
                                    // bool jud1p1 = ((mappedStartPoint[k * 4 * 5 + p * 4 + 1] - imgyf) > 0) && ((mappedEndPoint[k * 4 * 5 + p * 4 + 1] - imgyf) < 0);
                                    // bool jud1p2 = ((mappedStartPoint[k * 4 * 5 + p * 4 + 1] - imgyf) < 0) && ((mappedEndPoint[k * 4 * 5 + p * 4 + 1] - imgyf) > 0);
                                    // bool jud1 = jud1p1 || jud1p2;
                                    // if((k == 12) && (p == 0)){
                                    //     printf("p is %d, jud1 is %d, val1 is %f, val2 is %.22f\\n", p, jud1, (mappedStartPoint[k * 4 * 5 + p * 4 + 1] - imgyf), (mappedEndPoint[k * 4 * 5 + p * 4 + 1] - imgyf));
                                    // }
                                    bool jud2 = false;
                                    float tempVal1 = imgxf * line2dParam[k * 5 * 3 + p * 3 + 0] + imgyf * line2dParam[k * 5 * 3 + p * 3 + 1] + line2dParam[k * 5 * 3 + p * 3 + 2];
                                    float tempVal2 = largNum * line2dParam[k * 5 * 3 + p * 3 + 0] + imgyf * line2dParam[k * 5 * 3 + p * 3 + 1] + line2dParam[k * 5 * 3 + p * 3 + 2];
                                    if (((tempVal1 > 0) && (tempVal2 < 0)) || ((tempVal1 < 0) && (tempVal2 > 0))){
                                        jud2 = true;
                                    }
                                    if(jud1 && jud2){
                                        intersectionNum += 1;
                                        // if(k == 12){
                                        //     printf("Cross the %dth line.\\n", p);
                                        // }
                                    }
                                }
                            }
                            if (intersectionNum % 2 == 1){
                                planeCoveredRec[planeBdInsRec[k]] = 1;
                                int pPind = k * 4;
                                // printf("PlaneInd is: %d\\n", k);
                                float p = (aMinv[pPind + 0] * imgxf + aMinv[pPind + 1] * imgyf + aMinv[pPind + 2]);
                                float depth = - (aMinv[pPind + 3] / p);
                                // printf("depthVal:%f, \\t, val1:%f\\n", depth, (aMinv[pPind + 0] * imgxf + aMinv[pPind + 1] * imgyf + aMinv[pPind + 2]) );
                                if (depth > 0){
                                    if (imgMask[i] > depth){
                                        imgMask[i] = depth;
                                    }
                                }
                                else{
                                    // printf("Warning, negative depth value calculated.\\n");
                                }
                            }
                        }
                    } 
                }
                //}
                //}
            }
        }
    """)
    func = mod.get_function("doublify")
    return func

def wraplineAff():
    mod = SourceModule("""
        #include <stdio.h>
        __global__ void doublify(int* numOfPlanes, float* invM, float* planeParamRec, int* planesVaalidity, float* aMinv)
        {
            int blockId = blockIdx.x + blockIdx.y * gridDim.x;
            int idx = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
            for(int i = idx; i < *numOfPlanes; i += gridDim.x * gridDim.y * blockDim.x * blockDim.y){
                if(planesVaalidity[i] != 0){
                    int baseInd = i * 4;
                    for(int j = 0; j < 4; j++){
                        float tmpSum = 0;
                        for(int k = 0; k < 4; k++){
                            tmpSum += planeParamRec[baseInd + k] * invM[k * 4 + j];
                        }
                        aMinv[baseInd + j] = tmpSum;
                    }
                }
            }
        }
    """)
    func = mod.get_function("doublify")
    return func
def wrapIntersectLineseg():
    mod = SourceModule("""
        #include <stdio.h>
        __global__ void doublify(float* mappedStartPoint, float* mappedEndPoint, int* lineIndRec, int* lineNum_ex, int* imgSize, float* mappedStartPointnew, float* mappedEndPointNew, int* lineIndRecNew)
        {
            int blockId = blockIdx.x + blockIdx.y * gridDim.x;
            int idx = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
            for(int i = idx; i < *lineNum_ex; i += gridDim.x * gridDim.y * blockDim.x * blockDim.y){
                float imgW = imgSize[0] - 1;
                float imgL = imgSize[1] - 1;
                float p[2];
                float r[2];
                float q[2];
                float s[2];
                float q_p[2];
                float rxs;
                lineIndRecNew[i] = lineIndRec[i];
                if(lineIndRecNew[i] != -1){
                    int lineBaseInd = i * 4;
                    bool jud1 = (mappedStartPoint[lineBaseInd + 0] > imgL - 1) || (mappedStartPoint[lineBaseInd + 0] < 0) || (mappedStartPoint[lineBaseInd + 1] > imgW - 1) || (mappedStartPoint[lineBaseInd + 0] < 0);
                    bool jud2 = (mappedEndPoint[lineBaseInd + 0] > imgL - 1) || (mappedEndPoint[lineBaseInd + 0] < 0) || (mappedEndPoint[lineBaseInd + 1] > imgW - 1) || (mappedEndPoint[lineBaseInd + 0] < 0);
                    
                    mappedStartPointnew[lineBaseInd + 0] = mappedStartPoint[lineBaseInd + 0];
                    mappedStartPointnew[lineBaseInd + 1] = mappedStartPoint[lineBaseInd + 1];
                    mappedEndPointNew[lineBaseInd + 0] = mappedEndPoint[lineBaseInd + 0];
                    mappedEndPointNew[lineBaseInd + 1] = mappedEndPoint[lineBaseInd + 1];
                    if (jud1 || jud2){
                        lineIndRecNew[i] = -1;
                        for(int j = 0; j < 4; j++){
                            if(j == 0){
                                p[0] = 0;
                                p[1] = imgW;
                                r[0] = imgL;
                                r[1] = 0;
                            }
                            else if(j == 1){
                                p[0] = imgL;
                                p[1] = imgW;
                                r[0] = 0;
                                r[1] = -imgW;
                            }
                            else if(j == 2){
                                p[0] = imgL;
                                p[1] = 0;
                                r[0] = -imgL;
                                r[1] = 0;
                            }
                            else if(j == 3){
                                p[0] = 0;
                                p[1] = 0;
                                r[0] = 0;
                                r[1] = imgW;
                            }
                            q[0] = mappedStartPoint[lineBaseInd + 0];
                            q[1] = mappedStartPoint[lineBaseInd + 1];
                            s[0] = mappedEndPoint[lineBaseInd + 0] -  mappedStartPoint[lineBaseInd + 0];
                            s[1] = mappedEndPoint[lineBaseInd + 1] -  mappedStartPoint[lineBaseInd + 1];
                            q_p[0] = q[0] - p[0];
                            q_p[1] = q[1] - p[1];
                            rxs = r[0] * s[1] - r[1] * s[0];
                            // if (abs(rxs) < 1e-3){
                                // printf("Warning: unexpected 2d line relationship.\\n");
                            // }
                            float t = (q_p[0] * s[1] - q_p[1] * s[0]) / rxs;
                            float u = (q_p[0] * r[1] - q_p[1] * r[0]) / rxs;
                            // printf("t value is: %f, u value is: %f\\n", t, u);
                            if (abs(rxs) > 1e-3){
                                if(t >= 0 && t <= 1 && u >= 0 && u <= 1){
                                    lineIndRecNew[i] = lineIndRec[i];
                                    float replaced[2];
                                    replaced[0] = p[0] + t * r[0];
                                    replaced[1] = p[1] + t * r[1];
                                    if(j == 0){
                                        if(mappedStartPoint[lineBaseInd + 1] > imgW){
                                            mappedStartPointnew[lineBaseInd + 0] = replaced[0];
                                            mappedStartPointnew[lineBaseInd + 1] = replaced[1];
                                        }
                                        else{
                                            mappedEndPointNew[lineBaseInd + 0] = replaced[0];
                                            mappedEndPointNew[lineBaseInd + 1] = replaced[1];
                                        }
                                    }
                                    else if(j == 1){
                                        if(mappedStartPoint[lineBaseInd + 0] > imgL){
                                            mappedStartPointnew[lineBaseInd + 0] = replaced[0];
                                            mappedStartPointnew[lineBaseInd + 1] = replaced[1];
                                        }
                                        else{
                                            mappedEndPointNew[lineBaseInd + 0] = replaced[0];
                                            mappedEndPointNew[lineBaseInd + 1] = replaced[1];
                                        }
                                    }
                                    else if(j == 2){
                                        if(mappedStartPoint[lineBaseInd + 1] < 0){
                                            mappedStartPointnew[lineBaseInd + 0] = replaced[0];
                                            mappedStartPointnew[lineBaseInd + 1] = replaced[1];
                                        }
                                        else{
                                            mappedEndPointNew[lineBaseInd + 0] = replaced[0];
                                            mappedEndPointNew[lineBaseInd + 1] = replaced[1];
                                        }
                                    }
                                    else if(j == 3){
                                        if(mappedStartPoint[lineBaseInd + 0] < 0){
                                            mappedStartPointnew[lineBaseInd + 0] = replaced[0];
                                            mappedStartPointnew[lineBaseInd + 1] = replaced[1];
                                        }
                                        else{
                                            mappedEndPointNew[lineBaseInd + 0] = replaced[0];
                                            mappedEndPointNew[lineBaseInd + 1] = replaced[1];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    """)
    func = mod.get_function("doublify")
    return func


def wrapComputeNewBox():
    mod = SourceModule("""
        #include <stdio.h>
        #include <math.h>
        __global__ void doublify(float* mappedStartPointnew, float* mappedEndPointNew, int* lineIndRecNew, int* numOfPlanes, int* imageSize, int* newBbox, int* planesValidity_new)
        {
            int blockId = blockIdx.x + blockIdx.y * gridDim.x;
            int idx = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
            // if (idx == 3){
            for(int i = idx; i < *numOfPlanes; i += gridDim.x * gridDim.y * blockDim.x * blockDim.y){
                int basedIndex_forExpandedPts = i * 4 * 5;
                planesValidity_new[i] = 0;
                int maxX = -99999999;
                int minX = 99999999;
                int maxY = -99999999;
                int minY = 99999999;
                for(int j = 0; j < 5; j++){
                    if(lineIndRecNew[i * 5 + j] != -1){ 
                        // printf("Start Pt:%f,\\t,%f\\n",mappedStartPointnew[basedIndex_forExpandedPts + j * 4 + 0],mappedStartPointnew[basedIndex_forExpandedPts + j * 4 + 1]);
                        // printf("End Pt:%f,\\t,%f\\n",mappedEndPointNew[basedIndex_forExpandedPts + j * 4 + 0],mappedEndPointNew[basedIndex_forExpandedPts + j * 4 + 1]);
                        // Plane is valid only when there exists a valid line
                        planesValidity_new[i] = 1;
                        if (mappedStartPointnew[basedIndex_forExpandedPts + j * 4 + 0] > (float)maxX){
                            maxX = ceil(mappedStartPointnew[basedIndex_forExpandedPts + j * 4 + 0]);
                        }
                        if (mappedStartPointnew[basedIndex_forExpandedPts + j * 4 + 0] < (float)minX){
                            minX = floor(mappedStartPointnew[basedIndex_forExpandedPts + j * 4 + 0]);
                        }
                        if (mappedStartPointnew[basedIndex_forExpandedPts + j * 4 + 1] > (float)maxY){
                            maxY = ceil(mappedStartPointnew[basedIndex_forExpandedPts + j * 4 + 1]);
                        }
                        if (mappedStartPointnew[basedIndex_forExpandedPts + j * 4 + 1] < (float)minY){
                            minY = floor(mappedStartPointnew[basedIndex_forExpandedPts + j * 4 + 1]);
                        }
                        if (mappedEndPointNew[basedIndex_forExpandedPts + j * 4 + 0] > (float)maxX){
                            maxX = ceil(mappedEndPointNew[basedIndex_forExpandedPts + j * 4 + 0]);
                        }
                        if (mappedEndPointNew[basedIndex_forExpandedPts + j * 4 + 0] < (float)minX){
                            minX = floor(mappedEndPointNew[basedIndex_forExpandedPts + j * 4 + 0]);
                        }
                        if (mappedEndPointNew[basedIndex_forExpandedPts + j * 4 + 1] > (float)maxY){
                            maxY = ceil(mappedEndPointNew[basedIndex_forExpandedPts + j * 4 + 1]);
                        }
                        if (mappedEndPointNew[basedIndex_forExpandedPts + j * 4 + 1] < (float)minY){
                            minY = floor(mappedEndPointNew[basedIndex_forExpandedPts + j * 4 + 1]);
                        }
                    }
                }
                if(planesValidity_new[i] == 1){
                    // printf("Entered plane %d.\\n", i);
                    maxX < 0 ? maxX = 0: maxX = maxX;
                    maxX > imageSize[1] - 1 ? maxX = (int)(imageSize[1] - 1): maxX = maxX;
                    minX < 0 ? minX = 0: minX = minX;
                    minX > imageSize[1] - 1 ? minX = (int)(imageSize[1] - 1): minX = minX;
                    maxY < 0 ? maxY = 0: maxY = maxY;
                    maxY > imageSize[0] - 1 ? maxY = (int)(imageSize[0] - 1): maxY = maxY;
                    minY < 0 ? minY = 0: minY = minY;
                    minY > imageSize[0] - 1 ? minY = (int)(imageSize[0] - 1): minY = minY;
                    newBbox[i * 4] = minX;
                    newBbox[i * 4 + 1] = maxX;
                    newBbox[i * 4 + 2] = minY;
                    newBbox[i * 4 + 3] = maxY;
                    if (!( ((maxX - minX) > 0) && ((maxY - minY) > 0) )){
                        // printf("maxX: %d, minX: %d, maxY: %d, minY: %d\\n", maxX, minX, maxY, minY);
                        // printf("diff1: %d, diff2: %d\\n", (int)(maxX - minX), (int)(maxY - minY));
                        planesValidity_new[i] = 0;
                    }
                }
            }
            // }
        }
    """)
    func = mod.get_function("doublify")
    return func

def wrapLineRender():
    mod = SourceModule("""
        #include <stdio.h>
        __global__ void doublify(int* numofLines, float* aMinv, int* imageSize, float* mappedStartPointnew, float* mappedEndPointNew, int* planesValidity, float*line2dParam, int* lineValidity, float* depthMap, int* imgMask)
        {
            int blockId = blockIdx.x + blockIdx.y * gridDim.x;
            int idx = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
            int maxPixelNum = imageSize[0] * imageSize[1];
            // if (idx == 0){
            for(int i = idx; i < *numofLines; i += gridDim.x * gridDim.y * blockDim.x * blockDim.y){
                if(lineValidity[i] != -1){
                    int planeInd = i / 5;
                    int aMinvBaseInd = planeInd * 4;
                    float diffx = mappedStartPointnew[i * 4 + 0] - mappedEndPointNew[i * 4 + 0];
                    float diffy = mappedStartPointnew[i * 4 + 1] - mappedEndPointNew[i * 4 + 1];
                    float bigger;
                    abs(diffx) > abs(diffy) ? bigger = abs(diffx) : bigger = abs(diffy); 
                    int steps = round(bigger);
                    if(steps > 0){
                        float stepx = diffx / (float)steps;
                        float stepy = diffy / (float)steps;
                        float imgx;
                        float imgy;
                        int imgxs[3];
                        int imgys[3];
                        int linearInd;
                        float depth;
                        for(int k = 0; k < steps; k++){
                            imgx = round(mappedEndPointNew[i * 4 + 0] + (float)(k * stepx));
                            imgy = round(mappedEndPointNew[i * 4 + 1] + (float)(k * stepy));
                            depth = - (aMinv[aMinvBaseInd + 3] / (aMinv[aMinvBaseInd + 0] * imgx + aMinv[aMinvBaseInd + 1] * imgy + aMinv[aMinvBaseInd + 2]));
                            if(imgx >= 0 && imgx < imageSize[1] && imgy >=0 && imgy < imageSize[0]){
                                linearInd = imgy * imageSize[1] + imgx;
                                // if(depth - depthMap[linearInd] < 0.4f){
                                imgxs[0] = imgx - 1; imgxs[1] = imgx; imgxs[2] = imgx + 1;
                                imgys[0] = imgy - 1; imgys[1] = imgy; imgys[2] = imgy + 1;
                                for(int m = 0; m < 3; m++){
                                    for(int n = 0; n < 3; n++){
                                        if(imgxs[m] >= 0 && imgxs[m] < imageSize[1] && imgys[n] >=0 && imgys[n] < imageSize[0]){
                                            linearInd = imgys[n] * imageSize[1] + imgxs[m];
                                            if(depth - depthMap[linearInd] >= 0.5f){
                                                imgMask[linearInd] = 2;
                                            }
                                            // else{
                                            //     imgMask[linearInd] = 2;
                                            // }
                                        }
                                    }
                                }
                                // }
                            }
                            // if (linearInd < maxPixelNum){
                                // imgMask[linearInd] = 1;
                                // printf("Error occurs: idx:%d\\t,imgx: %f\\t,imgy: %f\\t,linearInd:%d\\n", idx, ,imgx, imgy, linearInd);
                            // }
                            // printf("imgx : %f\\t,imgy: %f\\t,linearInd:%d\\n", imgx, imgy, linearInd);
                            // imgMask[linearInd] = 1;
                            // if (depth - depthMap[linearInd] < 0.4f){
                            //     imgMask[linearInd] = 1;
                            // }
                        }
                    }
                }
            }
            for(int i = idx; i < *numofLines; i += gridDim.x * gridDim.y * blockDim.x * blockDim.y){
                if(lineValidity[i] != -1){
                    int planeInd = i / 5;
                    int aMinvBaseInd = planeInd * 4;
                    float diffx = mappedStartPointnew[i * 4 + 0] - mappedEndPointNew[i * 4 + 0];
                    float diffy = mappedStartPointnew[i * 4 + 1] - mappedEndPointNew[i * 4 + 1];
                    float bigger;
                    abs(diffx) > abs(diffy) ? bigger = abs(diffx) : bigger = abs(diffy); 
                    int steps = round(bigger);
                    if(steps > 0){
                        float stepx = diffx / (float)steps;
                        float stepy = diffy / (float)steps;
                        float imgx;
                        float imgy;
                        int imgxs[3];
                        int imgys[3];
                        int linearInd;
                        float depth;
                        for(int k = 0; k < steps; k++){
                            imgx = round(mappedEndPointNew[i * 4 + 0] + (float)(k * stepx));
                            imgy = round(mappedEndPointNew[i * 4 + 1] + (float)(k * stepy));
                            depth = - (aMinv[aMinvBaseInd + 3] / (aMinv[aMinvBaseInd + 0] * imgx + aMinv[aMinvBaseInd + 1] * imgy + aMinv[aMinvBaseInd + 2]));
                            if(imgx >= 0 && imgx < imageSize[1] && imgy >=0 && imgy < imageSize[0]){
                                linearInd = imgy * imageSize[1] + imgx;
                                // if(depth - depthMap[linearInd] < 0.4f){
                                imgxs[0] = imgx - 1; imgxs[1] = imgx; imgxs[2] = imgx + 1;
                                imgys[0] = imgy - 1; imgys[1] = imgy; imgys[2] = imgy + 1;
                                for(int m = 0; m < 3; m++){
                                    for(int n = 0; n < 3; n++){
                                        if(imgxs[m] >= 0 && imgxs[m] < imageSize[1] && imgys[n] >=0 && imgys[n] < imageSize[0]){
                                            linearInd = imgys[n] * imageSize[1] + imgxs[m];
                                            if(depth - depthMap[linearInd] < 0.5f){
                                                imgMask[linearInd] = 1;
                                            }
                                            // else{
                                            //     imgMask[linearInd] = 2;
                                            // }
                                        }
                                    }
                                }
                                // }
                            }
                            // if (linearInd < maxPixelNum){
                                // imgMask[linearInd] = 1;
                                // printf("Error occurs: idx:%d\\t,imgx: %f\\t,imgy: %f\\t,linearInd:%d\\n", idx, ,imgx, imgy, linearInd);
                            // }
                            // printf("imgx : %f\\t,imgy: %f\\t,linearInd:%d\\n", imgx, imgy, linearInd);
                            // imgMask[linearInd] = 1;
                            // if (depth - depthMap[linearInd] < 0.4f){
                            //     imgMask[linearInd] = 1;
                            // }
                        }
                    }
                }
            }
            // }
        }
    """)
    func = mod.get_function("doublify")
    return func


def wrapIntegration():
    mod = SourceModule("""
        #include <stdio.h>
        __global__ void doublify(int* imageSize, float* depthMap, int* imgMask, float* r, float* g, float* b)
        {
            int blockId = blockIdx.x + blockIdx.y * gridDim.x;
            int idx = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
            int maxPixelNum = imageSize[0] * imageSize[1];
            float tmpVal;
            float ratio = 0.8;
            for(int i = idx; i < maxPixelNum; i += gridDim.x * gridDim.y * blockDim.x * blockDim.y){
                if (depthMap[i] < 1e10){
                    g[i] = g[i] * ratio + 1 - ratio;
                    r[i] = r[i] * ratio;
                    b[i] = b[i] * ratio;
                }
                else{
                    depthMap[i] = 0;
                }
                if (imgMask[i] != 0){
                    if (imgMask[i] == 1){
                        g[i] = 0;
                        r[i] = 1;
                        b[i] = 0;
                    }
                    else if (imgMask[i] == 2){
                        g[i] = 0;
                        r[i] = 0;
                        b[i] = 1;
                    }
                }
            }
        }
    """)
    func = mod.get_function("doublify")
    return func


def wrapPointCheck():
    mod = SourceModule("""
      #include <stdio.h>
      __global__ void doublify(float* startPts, float* endPts, float* normDir, float* onePtOnPlane, int* numOfPoints, int* intersectionCondition)
      {
        // Every thread deal with one line
        int blockId = blockIdx.x + blockIdx.y * gridDim.x;
        int i = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

        // if (i == 8){
        for(int idx = i; idx < *numOfPoints; idx += gridDim.x * gridDim.y * blockDim.x * blockDim.y){
            // printf("idx is: %d\\n", idx);
            int ptIndexB = idx * 4;
            float mulRe1 = (startPts[ptIndexB] - onePtOnPlane[0]) * normDir[0] + (startPts[ptIndexB + 1] - onePtOnPlane[1]) * normDir[1] + (startPts[ptIndexB + 2] - onePtOnPlane[2]) * normDir[2];
            float mulRe2 = (endPts[ptIndexB] - onePtOnPlane[0]) * normDir[0] + (endPts[ptIndexB + 1] - onePtOnPlane[1]) * normDir[1] + (endPts[ptIndexB + 2] - onePtOnPlane[2]) * normDir[2];
            // printf("mulRe1:%f\\t,mulRe2:%f\\n", mulRe1, mulRe2);
            // printf("Pts1:%f,%f,%f\\n", startPts[ptIndexB],startPts[ptIndexB+1],startPts[ptIndexB+2]);
            // printf("Pts2:%f,%f,%f\\n", endPts[ptIndexB],endPts[ptIndexB+1],endPts[ptIndexB+2]);
            if ((mulRe1 > 0) && (mulRe2 > 0)){
                intersectionCondition[idx] = 1;
            }
            else if((mulRe1 < 0) && (mulRe2 < 0)){
                intersectionCondition[idx] = 2;
                startPts[ptIndexB] = 0;
                startPts[ptIndexB+1] = 0;
                startPts[ptIndexB+2] = 0;
                endPts[ptIndexB] = 0;
                endPts[ptIndexB+1] = 0;
                endPts[ptIndexB+2] = 0;
            }
            else{
                intersectionCondition[idx] = 3;
                // printf("Entered.\\n");
                float w_pointer[3];
                w_pointer[0] = startPts[ptIndexB] - onePtOnPlane[0];
                w_pointer[1] = startPts[ptIndexB + 1] - onePtOnPlane[1];
                w_pointer[2] = startPts[ptIndexB + 2] - onePtOnPlane[2];
                float u_pointer[3];
                u_pointer[0] = endPts[ptIndexB] - startPts[ptIndexB];
                u_pointer[1] = endPts[ptIndexB + 1] - startPts[ptIndexB + 1];
                u_pointer[2] = endPts[ptIndexB + 2] - startPts[ptIndexB + 2];
                float s = - (normDir[0] * w_pointer[0] + normDir[1] * w_pointer[1] + normDir[2] * w_pointer[2]) / (normDir[0] * u_pointer[0] + normDir[1] * u_pointer[1] + normDir[2] * u_pointer[2]);
                float intersectedPt[3];
                intersectedPt[0] = startPts[ptIndexB] + s * u_pointer[0];
                intersectedPt[1] = startPts[ptIndexB + 1] + s * u_pointer[1];
                intersectedPt[2] = startPts[ptIndexB + 2] + s * u_pointer[2];
                if(mulRe1 <= 0){
                    startPts[ptIndexB] = intersectedPt[0];
                    startPts[ptIndexB + 1] = intersectedPt[1];
                    startPts[ptIndexB + 2] = intersectedPt[2];
                }
                else{
                    endPts[ptIndexB] = intersectedPt[0];
                    endPts[ptIndexB + 1] = intersectedPt[1];
                    endPts[ptIndexB + 2] = intersectedPt[2];
                }
                // float lineDirOrg[3];
                // float norm1 = sqrt(u_pointer[0]^2 + u_pointer[1]^2 + u_pointer[2]^2)
                // lineDirOrg[0] = u_pointer[0] / norm1;
                // lineDirOrg[1] = u_pointer[1] / norm1;
                // lineDirOrg[2] = u_pointer[2] / norm1;
                // float lineDirNewew[3];
                // lineDirNewew[0] = 
            }
        }
        //}
      }
      """)
    func = mod.get_function("doublify")
    return func

def wrapAccleratedFormPolygons():
    mod = SourceModule("""
          #include <stdio.h>
          __global__ void doublify(float* startPts, float* endPts, int* numOfPlanes, int* intersectionCondition, float* startPts_ex, float* endPts_ex, int* lineIndRec, float* M, int* imageSize, float* mappedStartPoint, float* mappedEndPoint, int* bbBox, int* planesValidity, float* planeParamRec, float* orgStartPts, float* orgEndPts)
          {
            // Every thread deal with one plane
            int blockId = blockIdx.x + blockIdx.y * gridDim.x;
            int p = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
            
            for(int idx = p; idx < *numOfPlanes; idx += gridDim.x * gridDim.y * blockDim.x * blockDim.y){
            // if (idx < *numOfPlanes){
            // if (idx == 2){
                int baseIndex_forCondition = idx * 4;
                int baseIndex_forPts = idx * 4 * 4;
                int basedIndex_forExpandedPts = idx * 5 * 4;
                int countExp = 0;
                int first3Rec = -1;
                int second3Rec = -1;
                bool hasBehind = false;

                int tempCount = 0;
                for(int k = 0; k < 4; k++){
                    if(intersectionCondition[k + baseIndex_forCondition] == 3){
                        if(tempCount == 0){
                            first3Rec = k;
                            tempCount = tempCount + 1;
                        }
                        else if(tempCount == 1){
                            second3Rec = k;
                            tempCount = tempCount + 1;
                        }
                    }
                   else if(intersectionCondition[k + baseIndex_forCondition] == 2){
                        hasBehind = true;
                   }
                }
                for(int i = 0; i < 5; i++){
                    lineIndRec[idx * 5 + i] = -1;
                }
                for(int i = 0; i < 4; i++){
                    if(intersectionCondition[i + baseIndex_forCondition] == 1){
                        startPts_ex[basedIndex_forExpandedPts + countExp * 4] = startPts[baseIndex_forPts + i * 4];
                        startPts_ex[basedIndex_forExpandedPts + countExp * 4 + 1] = startPts[baseIndex_forPts + i * 4 + 1];
                        startPts_ex[basedIndex_forExpandedPts + countExp * 4 + 2] = startPts[baseIndex_forPts + i * 4 + 2];
                        startPts_ex[basedIndex_forExpandedPts + countExp * 4 + 3] = startPts[baseIndex_forPts + i * 4 + 3];
                        endPts_ex[basedIndex_forExpandedPts + countExp * 4] = endPts[baseIndex_forPts + i * 4];
                        endPts_ex[basedIndex_forExpandedPts + countExp * 4 + 1] = endPts[baseIndex_forPts + i * 4 + 1];
                        endPts_ex[basedIndex_forExpandedPts + countExp * 4 + 2] = endPts[baseIndex_forPts + i * 4 + 2];
                        endPts_ex[basedIndex_forExpandedPts + countExp * 4 + 3] = endPts[baseIndex_forPts + i * 4 + 3];
                        lineIndRec[idx * 5 + countExp] = idx;
                        countExp = countExp + 1;
                        // printf("i : %d,\\t countExp : %d\\n", i, countExp);
                    }
                    else if(intersectionCondition[i + baseIndex_forCondition] == 3){
                        float entryRecStart[4];
                        float entryRecEnd[4];
                        // bool btc = ((i == second3Rec) && ( (intersectionCondition[i + baseIndex_forCondition - 1] == 2) || (intersectionCondition[i + baseIndex_forCondition - 1] == 3)  ));
                        bool btc = ((i == second3Rec) && ( (intersectionCondition[i + baseIndex_forCondition - 1] == 2) || ((intersectionCondition[i + baseIndex_forCondition - 1] == 3) && (hasBehind == false))  ));
                        bool bbc = ((i == second3Rec) && (btc == false));
                        for(int k = 0; k < 4; k++){
                            entryRecStart[k] = startPts[baseIndex_forPts + i * 4 + k];
                            entryRecEnd[k] = endPts[baseIndex_forPts + i * 4 + k];
                        }
                        if (btc){
                            startPts_ex[basedIndex_forExpandedPts + countExp * 4] = endPts[baseIndex_forPts + first3Rec * 4];
                            startPts_ex[basedIndex_forExpandedPts + countExp * 4 + 1] = endPts[baseIndex_forPts + first3Rec * 4 + 1];
                            startPts_ex[basedIndex_forExpandedPts + countExp * 4 + 2] = endPts[baseIndex_forPts + first3Rec * 4 + 2];
                            startPts_ex[basedIndex_forExpandedPts + countExp * 4 + 3] = endPts[baseIndex_forPts + first3Rec * 4 + 3];
                            endPts_ex[basedIndex_forExpandedPts + countExp * 4] = startPts[baseIndex_forPts + second3Rec * 4];
                            endPts_ex[basedIndex_forExpandedPts + countExp * 4 + 1] = startPts[baseIndex_forPts + second3Rec * 4 + 1];
                            endPts_ex[basedIndex_forExpandedPts + countExp * 4 + 2] = startPts[baseIndex_forPts + second3Rec * 4 + 2];
                            endPts_ex[basedIndex_forExpandedPts + countExp * 4 + 3] = startPts[baseIndex_forPts + second3Rec * 4 + 3];
                            lineIndRec[idx * 5 + countExp] = idx;
                            countExp = countExp + 1;
                        }
                        for(int k = 0; k < 4; k++){
                            startPts_ex[basedIndex_forExpandedPts + countExp * 4 + k] = entryRecStart[k];
                            endPts_ex[basedIndex_forExpandedPts + countExp * 4 + k] = entryRecEnd[k];
                        }
                        lineIndRec[idx * 5 + countExp] = idx;
                        countExp = countExp + 1;

                        // if (i == second3Rec && intersectionCondition[i + baseIndex_forCondition - 1] != 2){
                        if (bbc){
                            startPts_ex[basedIndex_forExpandedPts + countExp * 4] = endPts[baseIndex_forPts + second3Rec * 4];
                            startPts_ex[basedIndex_forExpandedPts + countExp * 4 + 1] = endPts[baseIndex_forPts + second3Rec * 4 + 1];
                            startPts_ex[basedIndex_forExpandedPts + countExp * 4 + 2] = endPts[baseIndex_forPts + second3Rec * 4 + 2];
                            startPts_ex[basedIndex_forExpandedPts + countExp * 4 + 3] = endPts[baseIndex_forPts + second3Rec * 4 + 3];
                            endPts_ex[basedIndex_forExpandedPts + countExp * 4] = startPts[baseIndex_forPts + first3Rec * 4];
                            endPts_ex[basedIndex_forExpandedPts + countExp * 4 + 1] = startPts[baseIndex_forPts + first3Rec * 4 + 1];
                            endPts_ex[basedIndex_forExpandedPts + countExp * 4 + 2] = startPts[baseIndex_forPts + first3Rec * 4 + 2];
                            endPts_ex[basedIndex_forExpandedPts + countExp * 4 + 3] = startPts[baseIndex_forPts + first3Rec * 4 + 3];
                            lineIndRec[idx * 5 + countExp] = idx;
                            countExp = countExp + 1;
                        }
                    }
                }
                if(countExp > 0){
                    // float mappedPtsStart[5][4] = {0};
                    // float mappedPtsEnd[5][4] = {0};
                    for(int i = 0; i < countExp; i++){
                        int resourceMBase = basedIndex_forExpandedPts + i * 4;
                        for(int j = 0; j < 4; j++){
                            float val1 = 0;
                            float val2 = 0;
                            for(int k = 0; k < 4; k++){
                                val1 += M[j * 4 + k] * startPts_ex[resourceMBase + k];
                                val2 += M[j * 4 + k] * endPts_ex[resourceMBase + k];
                            }
                            // mappedPtsStart[i][j] = val1;
                            // mappedPtsEnd[i][j] = val2;
                            mappedStartPoint[basedIndex_forExpandedPts + i * 4 + j] = val1;
                            mappedEndPoint[basedIndex_forExpandedPts + i * 4 + j] = val2;
                        }
                        mappedStartPoint[basedIndex_forExpandedPts + i * 4 + 0] /= mappedStartPoint[basedIndex_forExpandedPts + i * 4 + 2];
                        mappedStartPoint[basedIndex_forExpandedPts + i * 4 + 1] /= mappedStartPoint[basedIndex_forExpandedPts + i * 4 + 2];
                        mappedEndPoint[basedIndex_forExpandedPts + i * 4 + 0] /= mappedEndPoint[basedIndex_forExpandedPts + i * 4 + 2];
                        mappedEndPoint[basedIndex_forExpandedPts + i * 4 + 1] /= mappedEndPoint[basedIndex_forExpandedPts + i * 4 + 2];
                    }
                    // int bounidngBox[4];
                    int maxX = -9999999;
                    int minX = 99999999;
                    int maxY = -9999999;
                    int minY = 9999999;
                    for(int i = 0; i < 5; i++){
                        if (mappedStartPoint[basedIndex_forExpandedPts + i * 4 + 0] > (float)maxX){
                            maxX = ceil(mappedStartPoint[basedIndex_forExpandedPts + i * 4 + 0]);
                        }
                        if (mappedStartPoint[basedIndex_forExpandedPts + i * 4 + 0] < (float)minX){
                            minX = floor(mappedStartPoint[basedIndex_forExpandedPts + i * 4 + 0]);
                        }
                        if (mappedStartPoint[basedIndex_forExpandedPts + i * 4 + 1] > (float)maxY){
                            maxY = ceil(mappedStartPoint[basedIndex_forExpandedPts + i * 4 + 1]);
                        }
                        if (mappedStartPoint[basedIndex_forExpandedPts + i * 4 + 1] < (float)minY){
                            minY = floor(mappedStartPoint[basedIndex_forExpandedPts + i * 4 + 1]);
                        }



                        if (mappedEndPoint[basedIndex_forExpandedPts + i * 4 + 0] > (float)maxX){
                            maxX = ceil(mappedEndPoint[basedIndex_forExpandedPts + i * 4 + 0]);
                        }
                        if (mappedEndPoint[basedIndex_forExpandedPts + i * 4 + 0] < (float)minX){
                            minX = floor(mappedEndPoint[basedIndex_forExpandedPts + i * 4 + 0]);
                        }
                        if (mappedEndPoint[basedIndex_forExpandedPts + i * 4 + 1] > (float)maxY){
                            maxY = ceil(mappedEndPoint[basedIndex_forExpandedPts + i * 4 + 1]);
                        }
                        if (mappedEndPoint[basedIndex_forExpandedPts + i * 4 + 1] < (float)minY){
                            minY = floor(mappedEndPoint[basedIndex_forExpandedPts + i * 4 + 1]);
                        }
                        // mappedStartPoint[basedIndex_forExpandedPts + i * 4 + 0] > (float)maxX ? maxX = mappedStartPoint[basedIndex_forExpandedPts + i * 4 + 0] : maxX = maxX;
                        // mappedStartPoint[basedIndex_forExpandedPts + i * 4 + 0] < (float)minX ? minX = mappedStartPoint[basedIndex_forExpandedPts + i * 4 + 0] : minX = minX;
                        // mappedStartPoint[basedIndex_forExpandedPts + i * 4 + 1] > (float)maxY ? maxY = mappedStartPoint[basedIndex_forExpandedPts + i * 4 + 1] : maxY = maxY;
                        // mappedStartPoint[basedIndex_forExpandedPts + i * 4 + 1] < (float)minY ? minY = mappedStartPoint[basedIndex_forExpandedPts + i * 4 + 1] : minY = minY;
                        // a<b ? printf("a is less") : printf("a is greater");
                        // mappedStartPoint[basedIndex_forExpandedPts + i * 4 + 0]
                    }
                    // printf("maxX:%d\\t,minX:%d\\t,maxY:%d\\t,minY:%d\\t,\\n", maxX, minX, maxY, minY);
                    maxX < 0 ? maxX = 0: maxX = maxX;
                    maxX > imageSize[1] - 1 ? maxX = imageSize[1] - 1: maxX = maxX;
                    minX < 0 ? minX = 0: minX = minX;
                    minX > imageSize[1] - 1 ? minX = imageSize[1] - 1: minX = minX;
                    maxY < 0 ? maxY = 0: maxY = maxY;
                    maxY > imageSize[0] - 1 ? maxY = imageSize[0] - 1: maxY = maxY;
                    minY < 0 ? minY = 0: minY = minY;
                    minY > imageSize[0] - 1 ? minY = imageSize[0] - 1: minY = minY;
                    bbBox[idx * 4] = minX;
                    bbBox[idx * 4 + 1] = maxX;
                    bbBox[idx * 4 + 2] = minY;
                    bbBox[idx * 4 + 3] = maxY;
                    if ((maxX - minX > 0) && (maxY - minY > 0)){
                        planesValidity[idx] = 1;
                        // Calculate the plane param
                        float planeParam[4];
                        float dir1[3];
                        float dir2[3];
                        for(int p = 0; p < 3; p++){
                            dir1[p] = orgEndPts[baseIndex_forPts + 0 * 4 + p] - orgStartPts[baseIndex_forPts + 0 * 4 + p];
                            dir2[p] = orgEndPts[baseIndex_forPts + 1 * 4 + p] - orgStartPts[baseIndex_forPts + 1 * 4 + p];
                        }
                        // printf("dir1:%f,%f,%f\\n",dir1[0],dir1[1],dir1[2]);
                        // printf("dir2:%f,%f,%f\\n",dir2[0],dir2[1],dir2[2]);
                        planeParam[0] = dir1[1] * dir2[2] - dir1[2] * dir2[1];
                        planeParam[1] = -dir1[0] * dir2[2] + dir1[2] * dir2[0];
                        planeParam[2] = dir1[0] * dir2[1] - dir1[1] * dir2[0];
                        // printf("planeParam:%f,%f,%f\\n",planeParam[0],planeParam[1],planeParam[2]);
                        // normalize the planeParam
                        float norm = sqrt(planeParam[0] * planeParam[0] + planeParam[1] * planeParam[1] + planeParam[2] * planeParam[2]);
                        // printf("cross product re is: %f\\n", planeParam[0] * dir1[0] + planeParam[1] * dir1[1] + planeParam[2] * dir1[2]);
                        // printf("cross product re is: %f\\n", planeParam[0] * dir2[0] + planeParam[1] * dir2[1] + planeParam[2] * dir2[2]);
                        planeParam[0] = planeParam[0] / norm;
                        planeParam[1] = planeParam[1] / norm;
                        planeParam[2] = planeParam[2] / norm;
                        planeParam[3] = -(planeParam[0] * startPts[baseIndex_forPts + 0] + planeParam[1] * startPts[baseIndex_forPts + 1] + planeParam[2] * startPts[baseIndex_forPts + 2]);
                        // printf("planeParam:%f,%f,%f,%f\\n",planeParam[0],planeParam[1],planeParam[2],planeParam[3]);
                        for(int p = 0; p < 4; p++){
                            planeParamRec[idx * 4 + p] = planeParam[p];
                        }
                    }

                    // planeParamRec[idx * 4] = 0;
                    // printf("maxX:%d\\t,minX:%d\\t,maxY:%d\\t,minY:%d\\t,\\n", maxX, minX, maxY, minY);
                    // printf("mappedPtsStart: \\n");
                    // for(int i = 0; i < 4; i++){
                    //     for(int j = 0; j < 4; j++){
                    //         printf("%f\\t", mappedPtsStart[i][j]);
                    //     }
                    // }
                    // for(int i = 0; i < countExp; i++){
                    //     for(int j = 0; j < 4; j++){
                    //         mappedStartPoint[basedIndex_forExpandedPts + i * 4 + j] = mappedPtsStart[i][j];
                    //         mappedEndPoint[basedIndex_forExpandedPts + i * 4 + j] = mappedPtsEnd[i][j];
                    //     } 
                    // }
                    // printf("\\nmappedStartPoint: \\n");
                    // for(int i = 0; i < 4; i++){
                    //     for(int j = 0; j < 4; j++){
                    //         printf("%f\\t", mappedStartPoint[basedIndex_forExpandedPts + i * 4 + j]);
                    //     }
                    // }
                }
            //}
            }
          }
          """)
    func = mod.get_function("doublify")
    return func


@jit(nopython=True)
def naiveMatrixInverse(inpuM):
    return np.linalg.inv(inpuM).astype(np.float32)