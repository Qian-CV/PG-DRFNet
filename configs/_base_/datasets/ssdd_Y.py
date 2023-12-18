# dataset settings
dataset_type = 'mmdet.CocoDataset'
data_root = '/media/ubuntu/nvidia/dataset/ssdd/'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='mmdet.LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='ConvertMask2BoxType', box_type='rbox'),
    dict(type='mmdet.Resize', scale=(608, 608), keep_ratio=True),
    dict(type='mmdet.Pad', size=(608, 608), pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.PackDetInputs')
]
val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmdet.Resize', scale=(608, 608), keep_ratio=True),
    dict(type='mmdet.Pad', size=(608, 608), pad_val=dict(img=(114, 114, 114))),
    # avoid bboxes being resized
    dict(
        type='mmdet.LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='ConvertMask2BoxType', box_type='qbox'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'instances'))
]
test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmdet.Resize', scale=(608, 608), keep_ratio=True),
    dict(type='mmdet.Pad', size=(608, 608), pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

metainfo = dict(classes=('ship', ), palette=[(0, 255, 0), ])

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='train/train.json',
        data_prefix=dict(img='train/images/'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline,
        file_client_args=file_client_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='test/all/test.json',
        data_prefix=dict(img='test/all/images/'),
        test_mode=True,
        pipeline=val_pipeline,
        file_client_args=file_client_args))
test_dataloader = val_dataloader

val_evaluator = dict(type='RotatedCocoMetric', metric='bbox')

test_evaluator = val_evaluator
