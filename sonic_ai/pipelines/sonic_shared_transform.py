from mmcv.parallel import DataContainer as DC
from mmpose.datasets.builder import PIPELINES
from mmpose.datasets.pipelines.shared_transform import Collect
from PIL import Image
import cv2
import json
import numpy as np
import os


@PIPELINES.register_module()
class SonicCollect(Collect):
    def __init__(self, keys, meta_keys, meta_name='img_metas', make_heatmap=False):
        self.keys = keys
        self.meta_keys = meta_keys
        self.meta_name = meta_name

        self.makeHeatmap = make_heatmap

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
        h, w = results['img'].shape[1:3]
        vis = results['img'].cpu().numpy()
        vis -= vis.min()
        vis /= vis.max()
        vis *= 255
        vis = vis.transpose(1, 2, 0).astype(np.uint8)
        vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        a = data['target'].copy()
        a = a.max(axis=0)
        a *= 255
        a = cv2.resize(a, (w, h))
        vis[:, :, 1] = a
        save_path = 'TrainPointImage/' + '/'.join(meta['image_file'].split('/')[4:-1])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        Image.fromarray(vis).save(save_path+ '/' + ''.join(
            meta['image_file'].split('/')[-1].split('.')[:-1]) + '_trainmap.png')
        return data
