import time

from configs2.base.base_sonic_dataset import Setinference_channel

# 服务器路径
_base_ = ['../base/default_runtime.py',
          '../base/schedule_sonic.py',
          '../base/base_sonic_dataset.py',
          './CYS.210905-极耳翻折-CATL-测试层数骨骼点配置-统一类-统一编号.py']

# 服务器路径
# project_name = 'CYS.210905-极耳翻折-CATL/切割10'
# dataset_path = f'/data2/4-标注任务/{project_name}'
project_name = 'BatteryPoleEar'
dataset_path = f'/data/14-调试数据/txj/BatteryPoleEar/data/image'
label_path = dataset_path + '/label.ini'
dataset_path_list = [f'{dataset_path}']
# num_classes = len(
#     LoadCategoryList()(results={'label_path': label_path})['point_list'])
num_classes = 50
current_channel = Setinference_channel[:num_classes]
Setdataset_channel = [
    current_channel,
]
Setinference_channel = current_channel
save_model_path = '/data/14-调试数据/txj/CYS.210905-极耳翻折-CATL/'
badcase_path = save_model_path
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
total_epochs = 70
checkpoint_config = dict(interval=10)
evaluation = dict(interval=1000, metric='mAP', save_best='AP')

channel_cfg = dict(
    num_output_channels=num_classes,
    dataset_joints=num_classes,
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
        flip_test=False,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11))

data_cfg = dict(
    image_size=[1024, 2560],
    heatmap_size=[256, 640],
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
    bbox_file=None,
    configFile=_base_,
)

train_pipeline = [
    dict(type='LoadImageFromFile', color_type='unchanged'),
    dict(type='CopyChannel', target_channel=3, overwrite_shape=True, add_noise=False),
    dict(type='TopDownAffine'),  # 仿射变换图像进行输入
    dict(type='ToTensor'),  # 将图像转换为pytorch的变量tensor
    dict(
        type='NormalizeTensor',  # 归一化
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='sonicTopDownGenerateTarget', sigma=2),  # 生成目标热图
    dict(
        type='SonicCollect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile', color_type='unchanged'),
    dict(type='CopyChannel', target_channel=3, overwrite_shape=True, add_noise=False),
    # dict(type='TopDownGetBboxCenterScale', padding=1.25),  # 将 bbox 从 [x, y, w, h] 转换为中心和缩放
    dict(type='TopDownAffine'),  # 仿射变换图像进行输入
    dict(type='ToTensor'),  # 将图像转换为pytorch的变量tensor
    dict(
        type='NormalizeTensor',  # 归一化
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
    samples_per_gpu=4,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=32),
    test_dataloader=dict(samples_per_gpu=32),
    train=dict(
        label_path=label_path,
        init_pipeline=train_init_pipeline,
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_path_list=dataset_path_list,
        dataset_info={{_base_.dataset_info}}),

    val=dict(
        label_path=label_path,
        init_pipeline=test_init_pipeline,
        dataset_path_list=dataset_path_list,
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}},
        timestamp=timestamp, ),
    test=dict(
        label_path=label_path,
        init_pipeline=test_init_pipeline,
        dataset_path_list=dataset_path_list,
        data_cfg=data_cfg,
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
    warmup_iters=10,
    warmup_ratio=0.001,
    step=[65, 70])
