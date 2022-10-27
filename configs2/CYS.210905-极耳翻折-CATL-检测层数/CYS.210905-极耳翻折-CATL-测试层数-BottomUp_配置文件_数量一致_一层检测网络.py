import time
log_level = 'INFO'  # 日志记录级别
custom_imports = dict(imports=['sonic_ai.losses.sonic_multi_loss_factory', 'sonic_ai.pipelines.sonic_shared_transform'],
                      allow_failed_imports=True)
_base_ = [
    '/data/txj/mmpose/configs/_base_/default_runtime.py',
    './CYS.210905-极耳翻折-CATL-测试层数骨骼点配置-统一类-统一编号(新版).py'
]
save_model_path = '/data/14-调试数据/txj/CYS.210905-极耳翻折-CATL/'
project_name = 'BatteryPoleEar'
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
checkpoint_config = dict(interval=50)
evaluation = dict(interval=10000, metric='mAP', save_best='AP')

num_people = 50
optimizer = dict(
    type='Adam',
    lr=0.0015,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=55,
    warmup_ratio=0.001,
    step=[350, 370])
total_epochs = 500
channel_cfg = dict(
    num_output_channels=1,
    dataset_joints=1,
    # dataset_channel=[
    #     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    #      30, 31, 32, 33, 34, 35, 36],
    # ],
    # inference_channel=[
    #     0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    #     30, 31, 32, 33, 34, 35, 36])
    dataset_channel=[[0, ]],
    inference_channel=[0])

data_cfg = dict(
    # image_size=1024,
    # base_size=512,
    # heatmap_size=[256],
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

data_root = '/data/14-调试数据/txj/BatteryPoleEar/data/blue_reduce_image_result/coco'
data = dict(
    workers_per_gpu=2,
    train_dataloader=dict(samples_per_gpu=24),
    val_dataloader=dict(samples_per_gpu=2),
    test_dataloader=dict(samples_per_gpu=2),
    train=dict(
        type='BottomUpCrowdPoseDataset',
        ann_file=f'{data_root}/annotations/keypoints_train.json',
        img_prefix=f'{data_root}/train/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='BottomUpCrowdPoseDataset',
        ann_file=f'{data_root}/annotations/keypoints_val.json',
        img_prefix=f'{data_root}/val/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='BottomUpCrowdPoseDataset',
        ann_file=f'{data_root}/annotations/keypoints_val.json',
        img_prefix=f'{data_root}/val/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
# runner = dict(
#     save_model_path=f"{save_model_path}/{project_name}",
#     timestamp=timestamp,
#     max_epochs=1)
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
