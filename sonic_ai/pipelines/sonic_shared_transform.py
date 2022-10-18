import os

import cv2
import numpy as np
from PIL import Image
from mmcv.parallel import DataContainer as DC

from mmpose.datasets.builder import PIPELINES
from mmpose.datasets.pipelines.shared_transform import Collect

'''
收集训练的结果数据，并且可以保存heatmap图像到TrainPointImage中
'''


@PIPELINES.register_module()
class SonicCollect(Collect):
    def __init__(self, keys, meta_keys, meta_name='img_metas', make_heatmap=False):
        self.keys = keys
        self.meta_keys = meta_keys
        self.meta_name = meta_name

        self.makeHeatmap = make_heatmap
        self.index = 0

    def __call__(self, results):
        """Performs the Collect formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
              to the next transform in pipeline.
        """
        if 'ann_info' in results:
            results.update(results['ann_info'])

        data = {}
        for key in self.keys:
            if isinstance(key, tuple):
                assert len(key) == 2
                key_src, key_tgt = key[:2]
            else:
                key_src = key_tgt = key
            data[key_tgt] = results[key_src]

        meta = {}
        if len(self.meta_keys) != 0:
            for key in self.meta_keys:
                if isinstance(key, tuple):
                    assert len(key) == 2
                    key_src, key_tgt = key[:2]
                else:
                    key_src = key_tgt = key
                meta[key_tgt] = results[key_src]
        if 'bbox_id' in results:
            meta['bbox_id'] = results['bbox_id']

        data[self.meta_name] = DC(meta, cpu_only=True)
        data_tmp = data
        images = results['img']
        meta_tmp = meta
        h, w = images.shape[1:3]
        vis = images.cpu().numpy()
        vis -= vis.min()
        vis /= vis.max()
        vis *= 255
        vis = vis.transpose(1, 2, 0).astype(np.uint8)
        vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        if 'target' in data_tmp.keys():
            a = data_tmp['target'].copy()
            a = a.max(axis=0)
            a *= 255
            a = cv2.resize(a, (w, h))
            vis[:, :, 1] = a
            save_path = 'TrainPointImage/' + '/'.join(meta_tmp['image_file'].split('/')[4:-1])
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            Image.fromarray(vis).save(save_path + '/' + ''.join(
                meta_tmp['image_file'].split('/')[-1].split('.')[:-1]) + '_trainmap.png')
        else:
            for i in range(len(data_tmp['targets'])):
                a = data_tmp['targets'][i].copy()
                a = a.max(axis=0)
                a *= 255
                a = cv2.resize(a, (w, h))
                vis[:, :, 1] = a
                save_path = 'TrainPointImage/' + '/'.join(results['image_file'].split('/')[4:-1])
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                Image.fromarray(vis).save(save_path + '/' + str(
                    self.index) + '_' + ''.join(
                    results['image_file'].split('/')[-1].split('.')[:-1]) + '_' + str(
                    i) + '_' + '_trainmap.png')
        self.index = self.index + 1
        return data
