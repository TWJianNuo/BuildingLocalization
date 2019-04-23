import json
infoDict = dict()
infoDict['prefixPath'] = '/media/shengjie/other/KITTI_scene_understanding/python_code/BuildingLocalization'
infoDict['vgg_LSTM_sideViewViewAngleAligned'] = 'trainData/sideViewAligned'
infoDict['vgg_LSTM_sideView64_256_withoutBlack'] = 'trainData/sideView64_256_withoutBlack'
infoDict['vgg_LSTM_globalImg_256_73'] = 'trainData/globalImg_256_73'
infoDict['vgg_LSTM_globalImg_localCropped_256_256'] = 'trainData/globalImg_localCropped_256_256'
infoDict['vgg_conv3d_globalImg_256_73'] = 'trainData/globalImg_256_73'
infoDict['allSeq'] = [
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
with open('jsonParam.json', 'w') as outfile:
    json.dump(infoDict, outfile)