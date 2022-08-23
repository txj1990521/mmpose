import os
import tempfile
import time
import traceback
import os.path as osp
import warnings
from mmcv import Config
import numpy as np
from mmpose.datasets.datasets.top_down.topdown_coco_dataset import TopDownCocoDataset
from mmpose.datasets.builder import DATASETS
from mmdet.datasets.pipelines.compose import Compose
from xtcocotools.coco import COCO


# dataConfig = 'configs2/base/CDJ_data.py'


@DATASETS.register_module()
class SonicKeyPointDataset(TopDownCocoDataset):
    def __init__(
            self,
            data_cfg,
            pipeline,
            dataset_path_list,  # 数据集路径 string，list
            label_path,  # 映射表路径
            init_pipeline,  # 数据预处理的pipeline
            filter_empty_gt=True,  # 当使用LoadOKPathList时要置为False
            timestamp=None,  # 时间戳，将会成为生成的文件名
            eval_pipeline=None,  # 训练评估的pipeline
            copy_pred_bad_case_path='/data/14-调试数据/过漏检数据',  # 过漏检数据的路径
            right_labels=None,  # OK的label，默认为['OK', "mark孔OK"]
            start=None,  # 数据的开始，0~1
            end=None,  # 数据的结束，0~1
            times=None,  # 重复次数
            ignore_labels=None,  # 屏蔽的label，默认为['屏蔽']
            dataset_info=None,
            test_mode=False,
            coco_style=True,
            *args,
            **kwargs):
        if dataset_info is None:
            warnings.warn(
                'dataset_info is missing. '
                'Check https://github.com/open-mmlab/mmpose/pull/663 '
                'for details.', DeprecationWarning)
            cfg = Config.fromfile('configs/_base_/datasets/coco.py')
            dataset_info = cfg._cfg_dict['dataset_info']

        """当dataset的参数和pipeline的参数有冲突时，以dataset的参数为主"""
        if right_labels is None:
            right_labels = ['OK', "mark孔OK"]
        self.right_labels = right_labels
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.ann_file = os.path.join(self.tmp_dir.name, "ann_file.json")
        self.image_info = {}

        self.pipeline = pipeline
        self.test_mode = test_mode

        self.use_gt_bbox = data_cfg['use_gt_bbox']
        self.bbox_file = data_cfg['bbox_file']
        self.det_bbox_thr = data_cfg.get('det_bbox_thr', 0.0)
        self.use_nms = data_cfg.get('use_nms', True)
        self.soft_nms = data_cfg['soft_nms']
        self.nms_thr = data_cfg['nms_thr']
        self.oks_thr = data_cfg['oks_thr']
        self.vis_thr = data_cfg['vis_thr']

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
        # self.ann_file = self.data['ann_save_path']
        self.img_prefix = dataset_path_list
        self.copy_pred_bad_case_path = copy_pred_bad_case_path

        super().__init__(
            data_cfg=data_cfg,
            pipeline=pipeline,
            ann_file=self.ann_file,
            img_prefix=dataset_path_list,
            dataset_info=dataset_info,
            test_mode=test_mode * args,
            **kwargs)

        if timestamp is not None:
            self.timestamp = timestamp
        else:
            self.timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        self.compose = None

        if self.test_mode:
            self.compose = Compose(self.eval_pipeline)
            for idx in range(len(self)):
                self.get_ann_info(idx)

        self.db = self._get_db()

    def _load_coco_keypoint_annotation_kernel(self, img_id):
        img_ann = self.coco.loadImgs(img_id)[0]
        width = img_ann['width']
        height = img_ann['height']
        num_joints = self.ann_info['num_joints']

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        objs = self.coco.loadAnns(ann_ids)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            if 'bbox' not in obj:
                continue
            x, y, w, h = obj['bbox']
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width - 1, x1 + max(0, w - 1))
            y2 = min(height - 1, y1 + max(0, h - 1))
            if ('area' not in obj or obj['area'] > 0) and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                valid_objs.append(obj)
        objs = valid_objs

        bbox_id = 0
        rec = []
        for obj in objs:

            if 'keypoints' not in obj:
                continue
            if max(obj['keypoints']) == 0:
                continue
            if 'num_keypoints' in obj and obj['num_keypoints'] == 0:
                continue
            joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
            joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)

            keypoints = np.array(obj['keypoints']).reshape(-1, 3)
            if num_joints != obj['num_keypoints']:
                for i in range(num_joints - obj['num_keypoints']):
                    keypoints = np.row_stack((keypoints, np.array([0, 0, 0])))

            joints_3d[:, :2] = keypoints[:, :2]
            joints_3d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3])

            center, scale = self._xywh2cs(*obj['clean_bbox'][:4])
            if len(self.img_prefix) == 1:
                image_file = osp.join(self.img_prefix[0], self.id2name[img_id])
            else:
                image_file = osp.join(self.img_prefix, self.id2name[img_id])
            rec.append({
                'image_file': image_file,
                'center': center,
                'scale': scale,
                'bbox': obj['clean_bbox'][:4],
                'rotation': 0,
                'joints_3d': joints_3d,
                'joints_3d_visible': joints_3d_visible,
                'dataset': self.dataset_name,
                'bbox_score': 1,
                'bbox_id': bbox_id
            })
            bbox_id = bbox_id + 1

        return rec

    def _get_db(self):
        """Load dataset."""
        #       if (not self.test_mode) or self.use_gt_bbox:
        # use ground truth bbox
        gt_db = self._load_coco_keypoint_annotations()
        #       else:
        # use bbox from detection
        #           gt_db = self._load_coco_person_detection_results()
        return gt_db
