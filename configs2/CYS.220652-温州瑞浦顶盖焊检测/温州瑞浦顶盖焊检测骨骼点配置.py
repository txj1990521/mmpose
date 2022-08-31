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
                name='A21',
                id=0,
                color=[0, 128, 255],
                type='upper',
                swap='A22'),
        1:
            dict(
                name='A22',
                id=1,
                color=[128, 0, 255],
                type='lower',
                swap='A21'),

    },
    skeleton_info={
        0:
            dict(link=('A21', 'A22'), id=0, color=[0, 255, 0]),
    },
    joint_weights=[
        1., 1.
    ],
    sigmas=[
        0.026, 0.026
    ])
