# 本地路径
# _base_ = ['E:/MMpose/mmpose/configs2/CYS.220317-雅康-欣旺达切叠一体机/骨骼点配置.py']

# 服务器路径
_base_ = ['./骨骼点配置.py']

dataset_info_test = {{_base_.dataset_info}}

# 本地路径
# data_root = 'D:/CYS.220317-雅康-欣旺达切叠一体机/result/CYS.220317-雅康-欣旺达切叠一体机'  # NG数据
# 服务器路径
data_root = '/data/14-调试数据/ypw/CYS.220317-雅康-欣旺达切叠一体机'


JointNum = 2
custom_imports = dict(
    imports=[
        "sonic_ai.topdown_custom_dataset"],
    allow_failed_imports=True)

Setdataset_channel = [[0, 1], ]
Setinference_channel = [0, 1]

log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=10)
evaluation = dict(interval=10, metric='mAP', save_best='AP')

optimizer = dict(
    type='Adam',
    lr=5e-4,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[170, 200])
total_epochs = 200
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

channel_cfg = dict(
    num_output_channels=JointNum,
    dataset_joints=JointNum,
    dataset_channel=Setdataset_channel,
    inference_channel=Setinference_channel)

# model settings
model = dict(
    type='TopDown',
    pretrained='torchvision://resnet18',
    backbone=dict(type='ResNet', depth=18),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=512,
        out_channels=channel_cfg['num_output_channels'],
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11))

data_cfg = dict(
    image_size=[512, 512],
    heatmap_size=[128, 128],
    # heatmap_size=[48, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=False,
    det_bbox_thr=0.0,
    # bbox_file='data/coco/person_detection_results/'
    # 'COCO_val2017_detections_AP_H_56_person.json',
    bbox_file=None,
    configFile=_base_,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetBboxCenterScale', padding=1.25),  # 将 bbox 从 [x, y, w, h] 转换为中心和缩放
    dict(type='TopDownRandomShiftBboxCenter', shift_factor=0.16, prob=0.3),  # 随机移动 bbox 中心。
    dict(type='TopDownRandomFlip', flip_prob=0.5),  # 随机图像翻转的数据增强
    dict(
        type='TopDownHalfBodyTransform',  # 半身数据增强
        num_joints_half_body=8,
        prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    # 随机缩放和旋转的数据增强，rot_factor旋转的角度，scale_factor缩放的数据增强系数
    dict(type='TopDownAffine'),  # 仿射变换图像进行输入
    dict(type='ToTensor'),  # 将图像转换为pytorch的变量tensor
    dict(
        type='NormalizeTensor',  # 归一化
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=2),  # 生成目标热图
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetBboxCenterScale', padding=1.25),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs'
        ]),
]

test_pipeline = val_pipeline

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=0,
    val_dataloader=dict(samples_per_gpu=32),
    test_dataloader=dict(samples_per_gpu=32),
    train=dict(
        type='TopDownCustomDataset',
        ann_file=f'{data_root}/annotations/keypoints_train.json',
        img_prefix=f'{data_root}/train/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='TopDownCustomDataset',
        ann_file=f'{data_root}/annotations/keypoints_val.json',
        img_prefix=f'{data_root}/val/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='TopDownCustomDataset',
        ann_file=f'{data_root}/annotations/keypoints_val.json',
        img_prefix=f'{data_root}/val/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
