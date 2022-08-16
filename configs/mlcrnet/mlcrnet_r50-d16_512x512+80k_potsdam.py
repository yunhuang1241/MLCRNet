_base_ = [
    '../_base_/models/mlcrnet_r50-d16.py',
    '../_base_/datasets/potsdam.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
model = dict(
    decode_head=dict(
        num_classes=6))
data = dict(
    samples_per_gpu=8,
    train = dict(data_root = '/home/hzf/datasets/potsdam'),
    val = dict(data_root = '/home/hzf/datasets/potsdam'),
    test = dict(data_root = '/home/hzf/datasets/potsdam'),
)
find_unused_parameters = True
