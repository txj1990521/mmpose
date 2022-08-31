import time

custom_imports = dict(
    imports=['sonic_ai.sonic_keypoint_dataset',
             'sonic_ai.sonic_epoch_based_runner',
             'sonic_ai.pipelines.init_pipeline',
             'sonic_ai.pipelines.eval_pipeline',
             'sonic_ai.pipelines.save_pipeline',
             'sonic_ai.pipelines.after_run_pipeline',
             'sonic_ai.sonic_after_run_hook',
             'sonic_ai.pipelines.dataset_pipeline',
             'sonic_ai.pipelines.sonic_shared_transform'], allow_failed_imports=True)
dataset_type = 'SonicKeyPointDataset'
img_scale = (640, 640)
Setdataset_channel = [
    [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
]
Setinference_channel = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

train_init_pipeline = [
    dict(type='CopyData2Local', target_dir='/data/公共数据缓存', run_rsync=True),  # 将训练数据保存到本地，使用rsync
    dict(type='LoadCategoryList', ignore_labels=['屏蔽']),  # 通过文件读取映射表，这里不能logger
    dict(type='LoadPathList'),  # 读取路径下的数据
    dict(type='SplitData', start=0, end=0.8, key='json_path_list'),  # 分割数据集，用于训练集和验证集
    dict(type='LoadJsonDataList'),  # 读取数据列表中json的数据
    dict(type='LoadLabelmeDataset'),  # 通过json数据对数据进行筛选
    dict(type='CopyData', times=1),  # 将数据翻倍
    dict(type='Labelme2COCOKeypoints'),  # 将labelme数据转化为coco数据
    dict(type='CopyErrorPath', copy_error_file_path='/data/14-调试数据/cyf'),
    # 将无法参与训练的数据保存在指定路径。例如类别没有出现在映射表中，没有找到图片，json损坏等
    dict(type='SaveJson'),  # 保存coco json
]

test_init_pipeline = [
    dict(type='CopyData2Local', target_dir='/data/公共数据缓存', run_rsync=False),
    dict(type='LoadCategoryList', ignore_labels=['屏蔽']),
    dict(type='LoadPathList'),
    dict(type='SplitData', start=0.8, end=1, key='json_path_list'),
    dict(type='LoadJsonDataList'),
    dict(type='LoadLabelmeDataset'),
    dict(type='Labelme2COCOKeypoints'),
    dict(type='CopyErrorPath', copy_error_file_path='/data/14-调试数据/cyf'),
    dict(type='StatCategoryCounter'),
    dict(type='SaveJson'),
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=img_scale,
        ratio_range=[0.75, 1.25],
        keep_ratio=True),
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
    # dict(
    #     type='CocoEvaluate', metric=['bbox'], classwise=True, iou_thrs=[0, 0]),
    # dict(type='ShowScores'),
    # dict(type='CreateConfusionMatrix'),
    # dict(type='CopyErrorCases')
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
    # train_dataloader=dict(class_aware_sampler=dict(num_sample_class=1)),
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
