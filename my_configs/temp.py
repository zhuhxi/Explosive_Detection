
data_root = 'data/explosive_dataset_coco/'
dataset_type = 'CocoDataset'
classes = ('explosive',)


ann_file='data/explosive_dataset_coco/annotations/instances_val.json',
data_prefix=dict(img='val/'),
ann_file='data/explosive_dataset_coco/annotations/instances_test.json',
data_prefix=dict(img='test/'),
ann_file='data/explosive_dataset_coco/annotations/instances_train.json',
data_prefix=dict(img='train/'),
metainfo=dict(classes=classes),