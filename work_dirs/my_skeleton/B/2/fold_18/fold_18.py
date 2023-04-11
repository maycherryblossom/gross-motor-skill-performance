model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='CTRGCN_F', graph_cfg=dict(layout='coco', mode='spatial')),
    cls_head=dict(type='GCNHead', num_classes=3, in_channels=256))
dataset_type = 'PoseDataset'
ann_file = '/home/pilab/ActionRecognition/mmpickles/fast_rcnn_pid_chain/B/2/fold_18_train.pkl'
ann_file_val = '/home/pilab/ActionRecognition/mmpickles/fast_rcnn_pid_chain/B/2/fold_18_val.pkl'
ann_file_test = '/home/pilab/ActionRecognition/mmpickles/fast_rcnn_pid_chain/B/2/fold_18_test.pkl'
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(
        type='Flip',
        flip_ratio=0.5,
        left_kp=[1, 3, 5, 7, 9, 11, 13, 15],
        right_kp=[2, 4, 6, 8, 10, 12, 14, 16]),
    dict(type='RandomScale', scale=0.1),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSample', clip_len=80),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSample', clip_len=80, num_clips=2, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSample', clip_len=80, num_clips=10, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=32,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=2),
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type='PoseDataset',
            ann_file=
            '/home/pilab/ActionRecognition/mmpickles/fast_rcnn_pid_chain/B/2/fold_18_train.pkl',
            pipeline=[
                dict(type='PreNormalize2D'),
                dict(
                    type='Flip',
                    flip_ratio=0.5,
                    left_kp=[1, 3, 5, 7, 9, 11, 13, 15],
                    right_kp=[2, 4, 6, 8, 10, 12, 14, 16]),
                dict(type='RandomScale', scale=0.1),
                dict(type='GenSkeFeat', dataset='coco', feats=['j']),
                dict(type='UniformSample', clip_len=80),
                dict(type='PoseDecode'),
                dict(type='FormatGCNInput', num_person=2),
                dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
                dict(type='ToTensor', keys=['keypoint'])
            ],
            split=None)),
    val=dict(
        type='PoseDataset',
        ann_file=
        '/home/pilab/ActionRecognition/mmpickles/fast_rcnn_pid_chain/B/2/fold_18_val.pkl',
        pipeline=[
            dict(type='PreNormalize2D'),
            dict(type='GenSkeFeat', dataset='coco', feats=['j']),
            dict(
                type='UniformSample', clip_len=80, num_clips=2,
                test_mode=True),
            dict(type='PoseDecode'),
            dict(type='FormatGCNInput', num_person=2),
            dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['keypoint'])
        ],
        split=None),
    test=dict(
        type='PoseDataset',
        ann_file=
        '/home/pilab/ActionRecognition/mmpickles/fast_rcnn_pid_chain/B/2/fold_18_test.pkl',
        pipeline=[
            dict(type='PreNormalize2D'),
            dict(type='GenSkeFeat', dataset='coco', feats=['j']),
            dict(
                type='UniformSample',
                clip_len=80,
                num_clips=10,
                test_mode=True),
            dict(type='PoseDecode'),
            dict(type='FormatGCNInput', num_person=2),
            dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['keypoint'])
        ],
        split=None))
optimizer = dict(
    type='SGD', lr=0.0002, momentum=0.9, weight_decay=5e-05, nesterov=True)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 30
checkpoint_config = dict(interval=30)
evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 2))
log_config = dict(
    interval=1,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
log_level = 'INFO'
work_dir = './work_dirs/fast_rcnn_pid_chain/try4/B/2/fold_18'
find_unused_parameters = True
dist_params = dict(backend='nccl')
gpu_ids = range(0, 2)
