import h5py
import numpy as np
import config
import dataloaders
import utils
import os
from tqdm import tqdm


def read_tile(tile):
    pad_height = tile.attrs["padding_height"]
    pad_width = tile.attrs["padding_width"]
    features = {_: np.array(tile[_])[np.newaxis, ...] for _ in tile.keys() if _ != "groundtruth"}
    groundtruth = np.array(tile["groundtruth"])
    return features, groundtruth, (pad_height, pad_width)


def apply_model(model, features):
    patch_size = config.data.patch_size
    _, height, width, _ = features["optical"].shape
    weighted_prob = np.zeros((height, width, config.data.n_outputs))
    weights = gaussian_kernel(patch_size)[..., np.newaxis]
    counts = np.zeros((height, width, 1))

    row = 0
    while row + patch_size <= height:
        col = 0 
        while col + patch_size <= width:
            patch = {}
            for feature, arr in features.items():
                if len(arr.shape) == 4:
                    patch[feature] = arr[:, row:row + patch_size, col:col + patch_size, :]
                else:
                    patch[feature] = arr
            patch_prob = model.predict(patch)[0]
            weighted_prob[row:row + patch_size, col:col + patch_size, :] += (weights * patch_prob)
            counts[row:row + patch_size, col:col + patch_size, :] += weights
            col += (patch_size // 2)
        
        row += (patch_size // 2)
    
    prob = weighted_prob / counts
    return prob


def gaussian_kernel(size, mu=0, sigma=1):
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    distance = np.sqrt(x**2 + y**2)
    kernel = np.exp(-(distance - mu)**2 / 2/ sigma**2) / np.sqrt(2 / np.pi) / sigma
    return kernel


def main():
    model_builder = config.model.model_builder
    model_args = config.model.model_args
    weights_path = os.path.join("weights", f"{config.model.model_name}_weights.h5")
    model = utils.build_model(
        model_builder, model_args, config.data, 
        weights_path=weights_path, mode="testing"
    )

    for tile_name in tqdm(test_dataset.keys()):
        tile = test_dataset[tile_name]
        features, true, (pad_height, pad_width) = read_tile(tile)

        prob = apply_model(model, features)
        pred = np.argmax(prob, axis=-1)

        prob = prob[pad_height:-pad_height, pad_width:-pad_width, :]
        pred = pred[pad_height:-pad_height, pad_width:-pad_width]
        true = true[pad_height:-pad_height, pad_width:-pad_width, -1]

        group = predictions_dataset.create_group(tile_name)
        group.create_dataset("prob", data=prob)
        group.create_dataset("pred", data=pred)
        group.create_dataset("true", data=true)
        

if __name__ == "__main__":
    utils.update_config_from_cli(config)
    test_dataset = h5py.File(config.data.test_dataset_path, "r")
    predictions_dataset_dir = os.path.join(config.data.predictions_dir, config.model.model_name)
    if not os.path.exists(predictions_dataset_dir):
        os.makedirs(predictions_dataset_dir, exist_ok=True)
    predictions_dataset_path = os.path.join(predictions_dataset_dir, "predictions.hdf5")
    if os.path.isfile(predictions_dataset_path):
        os.remove(predictions_dataset_path)
    predictions_dataset = h5py.File(predictions_dataset_path, "w")
    main()
    predictions_dataset.close()
    test_dataset.close()
