
def renderAndSaveImage(data_reader, renderIndex, svPath, func_integration, funcLinParamCal, func_IntersectLine, funcLineAff, func_computeNewBbox, func_depthRender, func_lineRender, func_pointCheck, func_acceleratedFormPolygons):
    triPlane_list = list()
    planeBdInsRecList = list()
    valCount = 0
    for idx, bdCompInd in enumerate(data_reader.buildingComp):
        # if data_reader.buildingComp[bdCompInd].visibility[renderIndex]:
        if data_reader.buildingComp[bdCompInd].range_visibilityArray[renderIndex]:
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