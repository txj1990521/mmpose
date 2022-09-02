# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
import warnings
import cv2
from tqdm import tqdm
from argparse import ArgumentParser
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo

project_name = 'CYS.220652-温州瑞浦顶盖焊检测/02-关键点/B2'
dataset_path = f'/data2/5-标注数据/{project_name}'
file_root = dataset_path  # 当前文件夹下的所有图片
Run_config = "configs2/CYS.220301-密封钉检测/密封钉配置文件.py"
Pose_checkpoint = '/data/14-调试数据/txj/CYS.220301-密封钉检测/02-关键点/CYS.220652-温州瑞浦顶盖焊检测/02-关键点/20220902_113330.pth'

Result_path = 'InferResult/' + project_name
# bbox = [457.5, 0, 10, 613]
bbox = [0, 0, 600, 2600]


def main():
    """Visualize the demo images.

    Require the json_file containing boxes.
    """
    global Run_config, Pose_checkpoint, Result_path
    parser = ArgumentParser()
    parser.add_argument('--pose_config', type=str, default=Run_config, help='Config file for detection')
    parser.add_argument('--pose_checkpoint', type=str, default=Pose_checkpoint, help='Checkpoint file')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--out-img-root',
        type=str,
        default=Result_path,
        help='Root of the output img file. '
             'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    args = parser.parse_args()
    assert args.show or (args.out_img_root != '')
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    # optional
    return_heatmap = True
    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None
    file_list = os.listdir(file_root)

    # process each image
    for i in tqdm(range(len(file_list))):
        # get bounding box annotations
        image_id = i + 1
        # 读取到图片的名字
        if 'json' not in file_list[i] and '@eaDir' not in file_list[i] and 'ini' not in file_list[i]:
            image_name_new = file_root + "/" + file_list[i]
            # print(image_name_new)
            # src = cv2.imread(image_name_new)
            # h, w = src.shape[:-1]
            # make project bounding boxes产生检测框
            myproject_results = [{'bbox': bbox}]

            # test a single image, with a list of bboxes
            pose_results, returned_outputs = inference_top_down_pose_model(
                pose_model,
                image_name_new,
                myproject_results,
                bbox_thr=None,
                format='xywh',
                dataset=dataset,
                dataset_info=dataset_info,
                return_heatmap=return_heatmap,
                outputs=output_layer_names)

            if args.out_img_root == '':
                out_file = None
            else:
                os.makedirs(args.out_img_root, exist_ok=True)
                out_file = os.path.join(args.out_img_root, (''.join(file_list[i].split('.')[:-1]) + '.jpg'))
            # 显示检测结果
            vis_pose_result(
                pose_model,
                image_name_new,
                pose_results,
                dataset=dataset,
                dataset_info=dataset_info,
                kpt_score_thr=args.kpt_thr,
                radius=args.radius,
                thickness=args.thickness,
                show=args.show,
                out_file=out_file)
    print("推理完毕！")
    sys.exit()


if __name__ == '__main__':
    main()
