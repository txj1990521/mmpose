import os
import sys
import glob
import json
import shutil
import argparse
import numpy as np
from PIL import Image
import os.path as osp
import cv2 as cv
from tqdm import tqdm
from labelme import utils
from sklearn.model_selection import train_test_split

image_type = ''
annotationsPath = ''
trainPath = ''
vlaPath = ''


class Labelme2coco_keypoints():
    def __init__(self, dictConfig):
        """
        Lableme 关键点数据集转 COCO 数据集的构造函数:

        Args
            args：命令行输入的参数
                - class_name 根类名字

        """

        self.classname_to_id = {dictConfig['class_name']: 1}
        self.images = []
        self.annotations = []
        self.categories = []
        self.ann_id = 0
        self.img_id = 0

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    def read_jsonfile(self, path):
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)

    def _get_box(self, points):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        return [min_x, min_y, max_x - min_x, max_y - min_y]

    def _get_keypoints(self, points, keypoints, num_keypoints):
        """
        解析 labelme 的原始数据， 生成 coco 标注的 关键点对象

        例如：
            "keypoints": [
                67.06149888292556,  # x 的值
                122.5043507571318,  # y 的值
                1,                  # 相当于 Z 值，如果是2D关键点 0：不可见 1：表示可见。
                82.42582269256718,
                109.95672933232304,
                1,
                ...,
            ],

        """

        if points[0] == 0 and points[1] == 0:
            visable = 0
        else:
            visable = 1
            num_keypoints += 1
        keypoints.extend([points[0], points[1], visable])
        return keypoints, num_keypoints

    def _image(self, obj, path):
        """
        解析 labelme 的 obj 对象，生成 coco 的 image 对象

        生成包括：id，file_name，height，width 4个属性

        示例：
             {
                "file_name": "training/rgb/00031426.jpg",
                "height": 224,
                "width": 224,
                "id": 31426
            }

        """

        image = {}
        global image_type
        image_type = obj['imagePath'][-3:]
        image['file_name'] = os.path.basename(path).replace(".json", "." + image_type)
        # image['file_name'] = path.replace(".json", "." + image_type)
        image['height'] = obj['imageHeight']
        image['width'] = obj['imageWidth']

        # img_x = utils.img_b64_to_arr(obj['imageData'])  # 获得原始 labelme 标签的 imageData 属性，并通过 labelme 的工具方法转成 array
        # if len(img_x.shape) == 2:
        #     image['height'] = img_x.shape[0]
        #     image['width'] = img_x.shape[1]
        # else:
        #     image['height'], image['width'] = img_x.shape[:-1]  # 获得图片的宽高

        # self.img_id = int(os.path.basename(path).split(".json")[0])
        self.img_id = self.img_id + 1
        image['id'] = self.img_id

        return image

    def _annotation(self, bboxes_list, keypoints_list, json_path, ismakeKeypoint, dictConfig):
        """
        生成coco标注

        Args：
            bboxes_list： 矩形标注框
            keypoints_list： 关键点
            json_path：json文件路径

        """
        if ismakeKeypoint != 1:
            pass
        else:
            if len(keypoints_list) > dictConfig['join_num'] * len(bboxes_list):
                print('you loss {} keypoint(s) with file {}'.format(
                    dictConfig['join_num'] * len(bboxes_list) - len(keypoints_list),
                    json_path))
                print('Please check ！！！')
                sys.exit()
        i = 0
        for object in bboxes_list:
            annotation = {}
            keypoints = []
            num_keypoints = 0

            label = object['label']
            bbox = object['points']
            annotation['id'] = self.ann_id
            annotation['image_id'] = self.img_id
            annotation['category_id'] = int(self.classname_to_id[label])
            annotation['iscrowd'] = 0
            annotation['area'] = 1.0
            annotation['segmentation'] = [np.asarray(bbox).flatten().tolist()]
            annotation['bbox'] = self._get_box(bbox)
            if ismakeKeypoint != 1:
                pass
            else:
                for keypoint in keypoints_list[i * dictConfig['join_num']: (i + 1) * dictConfig['join_num']]:
                    point = keypoint['points']
                    annotation['keypoints'], num_keypoints = self._get_keypoints(point[0], keypoints, num_keypoints)
                annotation['num_keypoints'] = num_keypoints

            i += 1
            self.ann_id += 1
            self.annotations.append(annotation)

    def _init_categories(self, dictConfig):
        """
        初始化 COCO 的 标注类别

        例如：
        "categories": [
            {
                "supercategory": "hand",
                "id": 1,
                "name": "hand",
                "keypoints": [
                    "wrist",
                    "thumb1",
                    "thumb2",
                    ...,
                ],
                "skeleton": [
                ]
            }
        ]
        """

        for name, id in self.classname_to_id.items():
            category = {}

            category['supercategory'] = name
            category['id'] = id
            category['name'] = name
            # 4 个关键点数据
            category['keypoint'] = dictConfig['total_classname']
            # category['keypoint'] = [str(i + 1) for i in range(args.join_num)]

            self.categories.append(category)

    # iSmakekeypoint是是否生成keypoint即骨骼点数据。0:为不生成 1:生成
    def to_coco(self, json_path_list, iSmakekeypoint, dictConfig):
        """
        Labelme 原始标签转换成 coco 数据集格式，生成的包括标签和图像

        Args：
            json_path_list：原始数据集的目录

        """

        self._init_categories(dictConfig)

        for json_path in tqdm(json_path_list):
            obj = self.read_jsonfile(json_path)  # 解析一个标注文件
            self.images.append(self._image(obj, json_path))  # 解析图片
            shapes = obj['shapes']  # 读取 labelme shape 标注

            bboxes_list, keypoints_list = [], []
            for shape in shapes:
                if shape['shape_type'] == 'rectangle':  # bboxs
                    bboxes_list.append(shape)  # keypoints
                elif shape['shape_type'] == 'point':
                    if iSmakekeypoint != 1:
                        pass
                    else:
                        keypoints_list.append(shape)

            self._annotation(bboxes_list, keypoints_list, json_path, iSmakekeypoint, dictConfig)

        keypoints = {}
        keypoints['info'] = {'description': 'Lableme Dataset', 'version': 1.0, 'year': 2021}
        keypoints['license'] = ['BUAA']
        keypoints['images'] = self.images
        keypoints['annotations'] = self.annotations
        keypoints['categories'] = self.categories
        return keypoints


def labelme2coco_process(dictConfig):
    """
    初始化COCO数据集的文件夹结构；
    coco - annotations  #标注文件路径
         - train        #训练数据集
         - val          #验证数据集
    Args：
        base_path：数据集放置的根路径
    """
    base_path = dictConfig['output']
    # 创建文件夹
    global annotationsPath, trainPath, vlaPath
    annotationsPath = os.path.join(base_path, dictConfig['project_name'], "annotations")
    trainPath = os.path.join(base_path, dictConfig['project_name'], "train")
    vlaPath = os.path.join(base_path, dictConfig['project_name'], "val")
    if not os.path.exists(annotationsPath):
        os.makedirs(annotationsPath)
    if not os.path.exists(trainPath):
        os.makedirs(trainPath)
    if not os.path.exists(vlaPath):
        os.makedirs(vlaPath)

    # 是否生成骨骼点数据
    isMakeKeyPointData = 1

    labelme_path = dictConfig['input']
    saved_coco_path = dictConfig['output']
    # 分离数据
    json_list_path = glob.glob(labelme_path + "/*.json")
    train_path, val_path = train_test_split(json_list_path, test_size=dictConfig['ratio'])
    print('{} for training'.format(len(train_path)),
          '\n{} for testing'.format(len(val_path)))
    print('Start transform please wait ...')

    l2c_train = Labelme2coco_keypoints(dictConfig)  # 构造数据集生成类
    # 生成训练集
    train_keypoints = l2c_train.to_coco(train_path, isMakeKeyPointData, dictConfig)
    l2c_train.save_coco_json(train_keypoints,
                             os.path.join(saved_coco_path, dictConfig['project_name'], "annotations",
                                          "keypoints_train.json"))
    # 生成验证集
    l2c_val = Labelme2coco_keypoints(dictConfig)
    val_instance = l2c_val.to_coco(val_path, isMakeKeyPointData, dictConfig)
    l2c_val.save_coco_json(val_instance, os.path.join(saved_coco_path, dictConfig['project_name'], "annotations",
                                                      "keypoints_val.json"))

    # # 拷贝 labelme 的原始图片到训练集和验证集里面
    global image_type
    for file in train_path:
        shutil.copy(file.replace("json", image_type), trainPath)
    for file in val_path:
        shutil.copy(file.replace("json", image_type), vlaPath)
    print('生成完毕！')
    sys.exit()
