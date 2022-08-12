custom_imports = dict(
    imports=[],
    allow_failed_imports=True)

dataset_type = 'SonicDataset'


train_init_pipeline = [
    dict(type='CopyData2Local', target_dir='/data/公共数据缓存', run_rsync=True),
    dict(type='LoadCategoryList', ignore_labels=['屏蔽']),
    dict(type='LoadPathList'),
    dict(type='SplitData', start=0, end=0.8, key='json_path_list'),
    dict(type='LoadJsonDataList'),
    dict(type='LoadLabelmeDataset'),
    dict(type='StatCategoryCounter'),
    dict(type='CopyData', times=1),
    dict(type='Labelme2Coco'),
    # dict(type='LoadOKPathList'),
    # dict(type='ShuffleCocoImage'),
    dict(type='CopyErrorPath', copy_error_file_path='/data/14-调试数据/cyf'),
    dict(type='SaveJson'),
]

test_init_pipeline = [
    dict(type='CopyData2Local', target_dir='/data/公共数据缓存', run_rsync=False),
    dict(type='LoadCategoryList', ignore_labels=['屏蔽']),
    dict(type='LoadPathList'),
    dict(type='SplitData', start=0.8, end=1, key='json_path_list'),
    dict(type='LoadJsonDataList'),
    dict(type='LoadLabelmeDataset'),
    dict(type='Labelme2Coco'),
    dict(type='CopyErrorPath', copy_error_file_path='/data/14-调试数据/cyf'),
    dict(type='StatCategoryCounter'),
    dict(type='SaveJson'),
]
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
eval_pipeline = [
    dict(
        type='CocoEvaluate', metric=['bbox'], classwise=True, iou_thrs=[0, 0]),
    dict(type='ShowScores'),
    dict(type='CreateConfusionMatrix'),
    dict(type='CopyErrorCases')
]
# 准备数据
data = dict(
    persistent_workers=True,
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        pipeline=train_pipeline,
        init_pipeline=train_init_pipeline),
    train_dataloader=dict(class_aware_sampler=dict(num_sample_class=1)),
    val=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        init_pipeline=test_init_pipeline,
        eval_pipeline=eval_pipeline),
    test=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        init_pipeline=test_init_pipeline,
        eval_pipeline=eval_pipeline))

