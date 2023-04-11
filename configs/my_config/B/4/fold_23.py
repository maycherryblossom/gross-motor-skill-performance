model = dict(
    type='RecognizerGCN',
    backbone=dict(
		type='CTRGCN_F',
        # frozenset=4
        graph_cfg=dict(layout='coco', mode='spatial')),
    cls_head=dict(type='GCNHead', num_classes=3, in_channels=256))

dataset_type = 'PoseDataset'
ann_file = './01_skeleton_data_extraction/my_skeleton/B/4/fold_23_train.pkl'
ann_file_val = './01_skeleton_data_extraction/my_skeleton/B/4/fold_23_val.pkl'
ann_file_test = './01_skeleton_data_extraction/my_skeleton/B/4/fold_23_test.pkl'

left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]


train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    # dict(type='RandomRot'),
    dict(type='RandomScale', scale=0.1),
    # dict(type='RandomGaussianNoise'),
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
        dataset=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline, split=None)),
    val=dict(type=dataset_type, ann_file=ann_file_val, pipeline=val_pipeline, split=None),
    test=dict(type=dataset_type, ann_file=ann_file_test, pipeline=test_pipeline, split=None))

# optimizer
optimizer = dict(type='SGD', lr=0.0002, momentum=0.9, weight_decay=0.00005, nesterov=True)
# optimizer = dict(type='AdamW', lr=1.2, betas=(0.9, 0.999), weight_decay=0.0005,
#                  paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
#                                                  'relative_position_bias_table': dict(decay_mult=0.),
#                                                  'norm': dict(decay_mult=0.),
#                                                  'backbone': dict(lr_mult=0.1)}))
# optimizer = dict(type='AdamW', lr=0.05, weight_decay=0.0003)

optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(policy='CosineRestart', periods=[500, 1000, 1500, 2000, 2500, 3000], restart_weights=[1, 0.85, 0.7, 0.55, 0.4, 0.25], min_lr=0, by_epoch=False)
# lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
# lr_config = dict(policy='step', step=[5, 15, 25], by_epoch=True)
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 30
checkpoint_config = dict(interval=30)
evaluation = dict(interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 2))
log_config = dict(interval=1, hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

# runtime settings
# workflow = [('train', 1)]
log_level = 'INFO'
work_dir = './work_dirs/my_skeleton/B/4/fold_23'

# load_from= '/home/pilab/ActionRecognition/pyskl/zoo/j.pth'
find_unused_parameters = True