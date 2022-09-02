from mmcv.parallel import DataContainer as DC
from mmpose.datasets.builder import PIPELINES
from mmpose.datasets.pipelines.shared_transform import Collect
import cv2 as cv
import json


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

        img_resize_w = results['image_size'][0]
        img_resize_h = results['image_size'][1]
        img = cv.imread(meta['image_file'])
        img = cv.resize(img, (img_resize_w, img_resize_h))
        for i in range(len(meta['joints_3d'])):
            circle_x = meta['joints_3d'][i][0]
            circle_y = meta['joints_3d'][i][1]
            cv.circle(img, (int(circle_x), int(circle_y)), 5, (0, 255, 0), -1)
        cv.imwrite(
            'TrainPointImage/' + meta['image_file'].split('/')[-2] + '/' + meta['image_file'].split('/')[-1].replace(
                '.png',
                '') + '_trainmap.png',
            img)
        # if self.makeHeatmap:
        #     print()
        return data
