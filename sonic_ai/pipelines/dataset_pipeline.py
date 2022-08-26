import os.path as osp

import mmcv
import numpy as np
from mmpose.datasets.builder import PIPELINES
from mmdet.datasets.pipelines.loading import LoadImageFromFile


@PIPELINES.register_module()
class Load3DImageFromFile(LoadImageFromFile):

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = osp.join(
                results['img_prefix'], results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        h_path = filename.replace('_Lum.tiff', '_H.tiff')
        lum_path = filename.replace('_H.tiff', '_Lum.tiff')

        img_bytes = self.file_client.get(h_path)
        img_h = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, channel_order=self.channel_order)

        img_bytes = self.file_client.get(lum_path)
        img_lum = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, channel_order=self.channel_order)

        img = np.zeros(img_h.shape + (3, ), dtype=np.uint16)
        img[:, :, 0] = img_lum
        img[:, :, 1] = img_h
        img[:, :, 2] = img_lum

        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results


@PIPELINES.register_module()
class CopyChannel:

    def __init__(self, target_channel=5, add_noise=True, noise_std=0.01, overwrite_shape=False):
        self.target_channel = target_channel
        self.add_noise = add_noise
        self.noise_std = noise_std
        self.overwrite_shape = overwrite_shape

    def __call__(self, results):

        img = results['img']
        target_channel = self.target_channel
        if len(img.shape) == 2:
            img = np.resize(img, img.shape + (1,))
        new_img = np.repeat(img, np.ceil(target_channel / img.shape[-1]), axis=-1)
        new_img = new_img[:, :, :target_channel]
        if self.add_noise:
            for i in range(3, self.target_channel):
                new_img[:, :, i] += np.random.random(
                    new_img.shape[:2]) * self.noise_std
        results['img'] = new_img
        if self.overwrite_shape:
            results['img_shape'] = img.shape
            results['ori_shape'] = img.shape
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(target_channel={self.target_channel}, add_noise={self.add_noise}, noise_std={self.noise_std})'
        return repr_str


@PIPELINES.register_module()
class RandomAdd:

    def __init__(
            self, channel=1, random_range=2000, ratio=0.5, *args, **kwargs):
        self.channel = channel
        self.random_range = random_range
        self.ratio = ratio

    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            img = results[key]
            if np.random.random() <= self.ratio:
                img_channel = img[:, :, self.channel]
                np.add(
                    img_channel[img_channel > 0],
                    np.random.randint(-self.random_range, self.random_range),
                    out=img_channel[img_channel > 0],
                    casting="unsafe")
            results[key] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class SubMean:

    def __init__(self, channel=1, non_zero=True, *args, **kwargs):
        self.channel = channel
        self.non_zero = non_zero

    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img = img.astype(np.float32)
            img_channel = img[:, :, self.channel]
            if self.non_zero:
                img_channel[img_channel > 0] -= img_channel[img_channel > 0].mean()
            else:
                img_channel -= img_channel.mean()
            results[key] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class AddMean:

    def __init__(self, channel=1, *args, **kwargs):
        self.channel = channel

    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img_channel = img[:, :, self.channel]
            img_channel[img_channel == 0] += img_channel[img_channel > 0].mean().astype(img.dtype)
            results[key] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
