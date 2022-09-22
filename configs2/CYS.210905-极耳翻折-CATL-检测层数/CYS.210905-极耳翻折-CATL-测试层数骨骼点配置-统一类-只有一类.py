dataset_info = dict(
    dataset_name='custom',
    paper_info=dict(
        author='tian',
        title='Microsoft coco: Common objects in context',
        container='European conference on computer vision',
        year='20220811',
        homepage='http://cocodataset.org/',
    ),
    keypoint_info={
        0:
            dict(
                name='pliesPoint',
                id=0,
                color=[0, 128, 255],
                type='',
                swap=''),
    },
    skeleton_info={
        0:
            dict(link=('pliesPoint', 'pliesPoint'), id=0, color=[0, 255, 0]),
    },
    joint_weights=[
        1.,
    ],
    sigmas=[
        0.026,
    ])
