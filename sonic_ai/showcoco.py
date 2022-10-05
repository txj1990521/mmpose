import os
import random

import cv2
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

# 1、定义数据集路径
cocoRoot = "/data/txj/mmpose/data/crowdpose"
# dataType = "val2017"
# annFile = os.path.join(cocoRoot, f'annotations/instances_{dataType}.json')

annFile = os.path.join(cocoRoot, f'annotations/mmpose_crowdpose_trainval.json')
print(f'Annotation file: {annFile}')

# 2、为实例注释初始化COCO的API
coco = COCO(annFile)

# 3、采用不同函数获取对应数据或类别
# ids = coco.getCatIds('person')[0]  # 采用getCatIds函数获取"person"类别对应的IDcoco = {COCO} <pycocotools.coco.COCO object at 0x7f8fefbf7040>
# print(f'"person" 对应的序号: {ids}')
# id = coco.getCatIds(
#     ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip",
#      "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "head", "neck"])[
#     0]  # 获取某一类的所有图片，比如获取包含dog的所有图片
# imgIds = coco.catToImgs[id]
# print(f'包含dog的图片共有：{len(imgIds)}张, 分别是：', imgIds)

cats = coco.loadCats(1)  # 采用loadCats函数获取序号对应的类别名称
print(f'"1" 对应的类别名称: {cats}')

imgIds = coco.getImgIds(catIds=[1])  # 采用getImgIds函数获取满足特定条件的图片（交集），获取包含person的所有图片
print(f'包含person的图片共有：{len(imgIds)}张')

# 4、将图片进行可视化
for m in range(len(imgIds)):
    imgId = imgIds[m]
    imgInfo = coco.loadImgs(imgId)[0]
    print(f'图像{imgId}的信息如下：\n{imgInfo}')

    imPath = os.path.join(cocoRoot, 'images', imgInfo['file_name'])
    im = cv2.imread(imPath)
    plt.axis('off')
    plt.imshow(im)
    plt.show()

    plt.imshow(im)
    plt.axis('off')
    annIds = coco.getAnnIds(imgIds=imgInfo['id'])  # 获取该图像对应的anns的Id

    anns = coco.loadAnns(annIds)
    print(f'图像{imgInfo["id"]}包含{len(anns)}个ann对象，分别是:\n{annIds}')

    coco.showAnns(anns)
    # print(f'ann{annIds[3]}对应的mask如下：')
    step = 3
    point_size = 1
    
    thickness = 4  # 可以为 0 、4、8
    j = 0

    for i in range(len(anns)):
        bbox_x1 = int((anns[i]['bbox'][0]))
        bbox_y1 = int((anns[i]['bbox'][1]))
        bbox_x2 = bbox_x1 + int((anns[i]['bbox'][2]))
        bbox_y2 = bbox_y1 + int((anns[i]['bbox'][3]))
        cv2.rectangle(im, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), (255, 0, 255), 1)
        b = [anns[i]['keypoints'][j:j + step] for j in range(0, len(anns[i]['keypoints']), step)]
        for k in range(len(b)):
            point_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            keypoint_x, keypoint_y = b[k][:-1]
            cv2.circle(im, (keypoint_x, keypoint_y), point_size, point_color, thickness)

    cv2.imwrite('/data/txj/mmpose/mmpose_crowdpose_test_data/' + imgInfo['file_name'], im)
