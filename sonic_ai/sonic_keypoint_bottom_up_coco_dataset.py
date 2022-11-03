import os
import os.path as osp
import tempfile
import time
import warnings

import json_tricks as json
import numpy as np
from mmcv import Config
from mmdet.datasets.pipelines import Compose
from xtcocotools.cocoeval import COCOeval

from mmpose.datasets.builder import DATASETS
from mmpose.datasets.datasets.bottom_up.bottom_up_coco import BottomUpCocoDataset


@DATASETS.register_module()
class SonicBottomUpPoseDataset(BottomUpCocoDataset):
    def __init__(self,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 test_mode=False,
                 dataset_path_list='',  # 数据集路径 string，list
                 label_path='',  # 映射表路径
                 init_pipeline=None,  # 数据预处理的pipeline
                 timestamp=None,  # 时间戳，将会成为生成的文件名
                 eval_pipeline=None,  # 训练评估的pipeline
                 copy_pred_bad_case_path='/data/14-调试数据/过漏检数据',  # 过漏检数据的路径
                 right_labels=None,  # OK的label，默认为['OK', "mark孔OK"]
                 start=None,  # 数据的开始，0~1
                 end=None,  # 数据的结束，0~1
                 times=None,  # 重复次数
                 ignore_labels=None,  # 屏蔽的label，默认为['屏蔽']
                 ):
        if dataset_info is None:
            warnings.warn(
                'dataset_info is missing. '
                'Check https://github.com/open-mmlab/mmpose/pull/663 '
                'for details.', DeprecationWarning)
            cfg = Config.fromfile('configs/_base_/datasets/crowdpose.py')
            dataset_info = cfg._cfg_dict['dataset_info']
        if right_labels is None:
            right_labels = ['OK', "mark孔OK"]
        self.right_labels = right_labels
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.ann_file = os.path.join(self.tmp_dir.name, "ann_file.json")
        self.image_info = {}
        self.pipeline = pipeline
        self.test_mode = test_mode
        self.eval_pipeline = eval_pipeline
        self.data = dict(
            dataset_path_list=dataset_path_list,
            label_path=label_path,
            ann_save_path=self.ann_file,
            start=start,
            end=end,
            times=times,
            ignore_labels=ignore_labels)
        compose = Compose(init_pipeline)
        compose(self.data)
        self.copy_pred_bad_case_path = copy_pred_bad_case_path

        super(BottomUpCocoDataset, self).__init__(
            self.ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode)
        if timestamp is not None:
            self.timestamp = timestamp
        else:
            self.timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        # if self.test_mode:
        #     self.compose = Compose(self.eval_pipeline)
        #     for idx in range(len(self)):
        #         self.get_ann_info(idx)
        self.compose = None
        self.ann_info['use_different_joint_weights'] = False
        print(f'=> num_images: {self.num_images}')

    def _do_python_keypoint_eval(self, res_file):
        """Keypoint evaluation using COCOAPI."""

        stats_names = [
            'AP', 'AP .5', 'AP .75', 'AR', 'AR .5', 'AR .75', 'AP(E)', 'AP(M)',
            'AP(H)'
        ]

        with open(res_file, 'r') as file:
            res_json = json.load(file)
            if not res_json:
                info_str = list(zip(stats_names, [
                    0,
                ] * len(stats_names)))
                return info_str

        coco_det = self.coco.loadRes(res_file)
        coco_eval = COCOeval(
            self.coco,
            coco_det,
            'keypoints_crowd',
            self.sigmas,
            use_area=False)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        info_str = list(zip(stats_names, coco_eval.stats))

        return info_str

    def _get_joints(self, anno):
        """Get joints for all people in an image."""
        num_people = len(anno)
        self.ann_info['num_joints'] = 50
        if self.ann_info['scale_aware_sigma']:
            joints = np.zeros((num_people, self.ann_info['num_joints'], 4),
                              dtype=np.float32)
        else:
            joints = np.zeros((num_people, self.ann_info['num_joints'], 3),
                              dtype=np.float32)

        for i, obj in enumerate(anno):
            keypoints_tmp = np.array(obj['keypoints']).reshape([-1, 3])
            for j in range(len(joints[0]) - len(keypoints_tmp)):
                keypoints_tmp = np.row_stack((keypoints_tmp, np.array([0, 0, 0])))
            joints[i, :, :3] = keypoints_tmp
            if self.ann_info['scale_aware_sigma']:
                # get person box
                box = obj['bbox']
                size = max(box[2], box[3])
                sigma = size / self.base_size * self.base_sigma
                if self.int_sigma:
                    sigma = int(np.ceil(sigma))
                assert sigma > 0, sigma
                joints[i, :, 3] = sigma

        return joints

    def _get_single(self, idx):
        """Get anno for a single image.

        Args:
            idx (int): image idx

        Returns:
            dict: info for model training
        """
        coco = self.coco
        img_id = self.img_ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anno = coco.loadAnns(ann_ids)

        mask = self._get_mask(anno, idx)
        anno = [
            obj.copy() for obj in anno
            if obj['iscrowd'] == 0 or obj['num_keypoints'] > 0
        ]

        joints = self._get_joints(anno)
        mask_list = [mask.copy() for _ in range(self.ann_info['num_scales'])]
        joints_list = [
            joints.copy() for _ in range(self.ann_info['num_scales'])
        ]

        db_rec = {}
        db_rec['dataset'] = self.dataset_name
        db_rec['image_file'] = osp.join(self.img_prefix, self.id2name[img_id])
        db_rec['mask'] = mask_list
        db_rec['joints'] = joints_list

        return db_rec
