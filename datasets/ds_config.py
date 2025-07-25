dataset_root = "s3://ljh-data/croco"
dataset_names = ['habitat_release',]  # ARKitScenes, MegaDepth, 3DStreetView, IndoorVL
dataset_nori_index_dir = "s3://ljh-data/croco/nori_index"
dataset_nori_volume_dir = "s3://ljh-data/croco/nori_volume"
transforms_operations = "crop224+acolor"
batch_size = 256
seed = 42  # None for not specifying