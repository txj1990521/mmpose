import time

from configs2.base.base_sonic_dataset import Setinference_channel
from sonic_ai.pipelines.init_pipeline import LoadCategoryList

# 服务器路径
_base_ = ['../base/default_runtime.py',
          '../base/schedule_sonic.py',
          '../base/base_sonic_dataset.py',
          './CYS.210905-极耳翻折-CATL-测试层数骨骼点配置-统一类-统一编号(新版).py']
# 服务器路径
project_name = 'BatteryPoleEar'
dataset_path = '/data/14-调试数据/txj/BatteryPoleEar/data/blue_reduce_image_lableme'
label_path = dataset_path + '/label.ini'
dataset_path_list = [f'{dataset_path}']
num_classes = len(
    LoadCategoryList()(results={'label_path': label_path})['point_list'])
current_channel = Setinference_channel[:num_classes]
Setdataset_channel = [
    current_channel,
]
Setinference_channel = current_channel
dataset_type = 'SonicBottomUpPoseDataset'
save_model_path = '/data/14-调试数据/txj/CYS.210905-极耳翻折-CATL/BatteryPoleEar'
badcase_path = save_model_path
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
total_epochs = 200
checkpoint_config = dict(interval=5)
evaluation = dict(interval=1000, metric='mAP', save_best='AP')
num_people = 50

channel_cfg = dict(
    num_output_channels=1,
    dataset_joints=1,
    dataset_channel=[[0, ]],
    inference_channel=[0])

# model settings
model = dict(
    type='AssociativeEmbedding',
    pretrained='https://download.openmmlab.com/mmpose/'
               'pretrain_models/hrnet_w32-36af842e.pth',
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4,),
                num_channels=(64,)),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256))),
    ),
    keypoint_head=dict(
        type='AESimpleHead',
        in_channels=32,
        num_joints=50,
        num_deconv_layers=0,
        tag_per_joint=True,
        with_ae_loss=[True],
        extra=dict(final_conv_kernel=1, ),
        loss_keypoint=dict(
            type='MultiLossFactory',
            num_joints=50,
            num_stages=1,
            ae_loss_type='exp',
            with_ae_loss=[True],
            push_loss_factor=[0.01],
            pull_loss_factor=[0.001],
            with_heatmaps_loss=[True],
            heatmaps_loss_factor=[1.0])),
    train_cfg=dict(),
    test_cfg=dict(
        num_joints=50,
        max_num_people=50,
        scale_factor=[1],
        with_heatmaps=[True],
        with_ae=[True],
        project2image=True,
        align_corners=False,
        nms_kernel=5,
        nms_padding=2,
        tag_per_joint=True,
        detection_threshold=0.9,
        tag_threshold=1,
        use_detection_val=True,
        ignore_too_much=False,
        adjust=True,
        refine=True,
        flip_test=False))

data_cfg = dict(
    image_size=[256, 5120],
    base_size=512,
    heatmap_size=[[64, 1280]],
    base_sigma=2,
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    num_scales=1,
    scale_aware_sigma=False,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='BottomUpRandomAffine',
        rot_factor=0,
        scale_factor=[0.75, 1.5],
        scale_type='short',
        trans_factor=0),
    # dict(type='BottomUpRandomFlip', flip_prob=0.5),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='BottomUpGenerateTarget',
        sigma=2,
        max_num_people=num_people,
    ),
    dict(
        type='Collect',
        keys=['img', 'joints', 'targets', 'masks'],
        meta_keys=[]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='BottomUpGetImgSize', test_scale_factor=[1]),
    dict(
        type='BottomUpResizeAlign',
        transforms=[
            dict(type='ToTensor'),
            dict(
                type='NormalizeTensor',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'aug_data', 'test_scale_factor', 'base_size',
            'center', 'scale', 'flip_index'
        ]),
]

test_pipeline = val_pipeline

train_init_pipeline = [
    dict(type='CopyData2Local', target_dir='/data/公共数据缓存', run_rsync=True),
    dict(type='LoadCategoryList', ignore_labels=['屏蔽']),
    dict(type='LoadPathList'),
    dict(type='SplitData', start=0, end=0.8, key='json_path_list'),
    dict(type='LoadJsonDataList'),
    dict(type='LoadLabelmeDataset'),
    dict(type='StatCategoryCounter'),
    dict(type='CopyData', times=1),
    dict(type='Labelme2COCOKeypoints', bbox_full_image=False),
    dict(type='CopyErrorPath', copy_error_file_path='/data/14-调试数据/txj'),
    dict(type='SaveJson'),
]

test_init_pipeline = [
    dict(type='CopyData2Local', target_dir='/data/公共数据缓存', run_rsync=True),
    dict(type='LoadCategoryList', ignore_labels=['屏蔽']),
    dict(type='LoadPathList'),
    dict(type='SplitData', start=0, end=0.8, key='json_path_list'),
    dict(type='LoadJsonDataList'),
    dict(type='LoadLabelmeDataset'),
    dict(type='StatCategoryCounter'),
    dict(type='CopyData', times=1),
    dict(type='Labelme2COCOKeypoints', bbox_full_image=False),
    dict(type='CopyErrorPath', copy_error_file_path='/data/14-调试数据/txj'),
    dict(type='SaveJson'),
]
data = dict(
    persistent_workers=False,
    samples_per_gpu=16,
    workers_per_gpu=0,
    val_dataloader=dict(samples_per_gpu=32),
    test_dataloader=dict(samples_per_gpu=32),
    train=dict(
        type=dataset_type,
        label_path=label_path,
        init_pipeline=train_init_pipeline,
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        img_prefix=dataset_path,
        dataset_path_list=dataset_path_list,
        dataset_info={{_base_.dataset_info}}),

    val=dict(
        type=dataset_type,
        label_path=label_path,
        init_pipeline=test_init_pipeline,
        dataset_path_list=dataset_path_list,
        data_cfg=data_cfg,
        img_prefix=dataset_path,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}},
        timestamp=timestamp,
    ),
    test=dict(
        type=dataset_type,
        label_path=label_path,
        init_pipeline=test_init_pipeline,
        dataset_path_list=dataset_path_list,
        data_cfg=data_cfg,
        img_prefix=dataset_path,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}},
        timestamp=timestamp,
    ))
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

runner = dict(
    save_model_path=f"{save_model_path}/{project_name}",
    timestamp=timestamp,
    max_epochs=1)
LoadCategoryList = None
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=50,
    warmup_ratio=0.001,
    step=[170, 200])
