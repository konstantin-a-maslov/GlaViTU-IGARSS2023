import dataloaders
import dataloaders.transformations
import h5py
import os


data_dir = os.path.join("/data", "massive", "glacier_outlines")
train_dataset_path = os.path.join(data_dir, "precomputed", "train_ps384_igarss23.hdf5")
val_dataset_path = os.path.join(data_dir, "precomputed", "val_ps384_igarss23.hdf5")
test_dataset_path = os.path.join(data_dir, "precomputed", "test_ps384_igarss23.hdf5")

predictions_dir = os.path.join(data_dir, "predictions")

n_outputs = 2
patch_size = 384
batch_size = 8

features = ["optical", "dem", "sar"]
labels = "groundtruth"
input_shapes = {
    "optical": (patch_size, patch_size, 6),
    "dem": (patch_size, patch_size, 2),
    "sar": (patch_size, patch_size, 2),
}

train_sampler_builder = dataloaders.RandomSampler
train_sampler_args = {
    "dataset": h5py.File(train_dataset_path, "r"), 
    "patch_size": patch_size, 
    "features": features,
    "labels": labels
}
train_plugins = [
    dataloaders.Augmentation([
        dataloaders.transformations.random_vertical_flip(),
        dataloaders.transformations.random_horizontal_flip(),
        dataloaders.transformations.random_rotation(),
        dataloaders.transformations.crop_and_scale(patch_size=patch_size)
    ])
]

val_sampler_builder = dataloaders.ConsecutiveSampler
val_sampler_args = {
    "dataset": h5py.File(val_dataset_path, "r"), 
    "patch_size": patch_size, 
    "features": features,
    "labels": labels
}
val_plugins = []
