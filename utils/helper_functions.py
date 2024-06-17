import bisect
import torch
import yaml
import random
import os
import numpy as np
from pyproj import Proj, transform

# Define the existing CRS and the target CRS
PROJ4_H8 = "+proj=geos +over +lon_0=140.700 +lat_0=0.000 +a=6378137.000 +f=0.0033528129638281333 +h=35785863.0"
PROJ4_4326 = "+proj=longlat +datum=WGS84 +no_defs"  # This is for EPSG:4326
# Define the transformation function
TRANSFORM_H8_4326 = Proj(PROJ4_H8), Proj(PROJ4_4326)


def get_sliced_data(array, time_steps, bands):
    """Get the sliced image data by slicing across the time_steps and bands.

    Args:
        idx: index of the data point

    """
    time_steps_array = array[time_steps, :, :, :]
    bands_array = time_steps_array[:, bands, :, :]

    return bands_array


def get_4326_fromh8(lon_array, lat_array):
    """Get the 4326 coordinates from the H8 coordinates."""

    new_longitude_array, new_latitude_array = transform(
        TRANSFORM_H8_4326[0], TRANSFORM_H8_4326[1], lon_array, lat_array
    )

    return new_longitude_array, new_latitude_array


def get_standardized_data(features: np.array, means: list, stds: list):
    """Standardises the features bases on the provided means and stds"""

    features_copy = (
        features.copy()
    )  # Copy the features to avoid modifying the original array for debugging purposes
    for idx, values in enumerate(zip(means, stds)):
        mean = values[0]
        std = values[1]
        features_copy[:, idx, :, :] = (features[:, idx, :, :] - mean) / std        

    return features_copy

def get_eval_standardized_data(features: np.array, means: list, stds: list):
    """Standardises the features bases on the provided means and stds"""

    features_copy = (
        features.copy()
    )  # Copy the features to avoid modifying the original array for debugging purposes
    for idx, values in enumerate(zip(means, stds)):
        mean = values[0]
        std = values[1]
        features_copy[idx, :, :] = (features[idx, :, :] - mean) / std        

    return features_copy


def revert_feature_standardization(features: np.array, means: list, stds: list):
    """Reverts standardisation, e.g. for plotting batched image data"""
    for idx, values in enumerate(zip(means, stds)):
        mean = values[0]
        std = values[1]
        features[idx, :, :] = (features[idx, :, :] * std) + mean                # with no timeseries
        # features[:, idx, :, :] = (features[:, idx, :, :] * std) + mean        # with timeseries

    return features


def get_one_hot_encoding(classes, categorical_array: np.array):
    """Returns an one-hot encoded array with one channel for each class"""

    out = np.zeros(
        (len(classes), categorical_array.shape[1], categorical_array.shape[2])
    )
    for idx, cls in enumerate(classes):
        out[idx, :, :] = np.where(categorical_array == cls, 1, 0)

    return out


def fold_worldcover(classes, categorical_one_hot: np.array):
    """Creates a single channel array containing the ESA WorldCover classes from a one-hot encoded array"""

    out = np.zeros((categorical_one_hot.shape[1], categorical_one_hot.shape[1]))
    for idx, cls in enumerate(classes):
        out[categorical_one_hot[idx, :, :] == 1] = cls

    return out


def get_file_path(idx, split, data_dir):
    """Get the h5 file path and preceeding index for a given index.

    Args:
        idx: index of the data point

    Returns:
        file path of the h5 file

    """

    # XXX: Make sure to have a if condition once the test data is ready : if self.split="train" else "test" ansd have a diffrent mapping for test data
    if split == "train" or split == "val":
        # Dictionary mapping index to file path
        idx_h5_mapping = {
            12839: f"{data_dir}/training_dynamic_data_0_12840.h5",
            25679: f"{data_dir}/training_dynamic_data_12840_25680.h5",
            38519: f"{data_dir}/training_dynamic_data_25680_38520.h5",
            51359: f"{data_dir}/training_dynamic_data_38520_51360.h5",
            64199: f"{data_dir}/training_dynamic_data_51360_64200.h5",
            77039: f"{data_dir}/training_dynamic_data_64200_77040.h5",
            89879: f"{data_dir}/training_dynamic_data_77040_89880.h5",
            102719: f"{data_dir}/training_dynamic_data_89880_102720.h5",
            115559: f"{data_dir}/training_dynamic_data_102720_115560.h5",
            128399: f"{data_dir}/training_dynamic_data_115560_128400.h5",
            141239: f"{data_dir}/training_dynamic_data_128400_141240.h5",
            154079: f"{data_dir}/training_dynamic_data_141240_154080.h5",
            166919: f"{data_dir}/training_dynamic_data_154080_166920.h5",
            179759: f"{data_dir}/training_dynamic_data_166920_179760.h5",
            192599: f"{data_dir}/training_dynamic_data_179760_192600.h5",
            205439: f"{data_dir}/training_dynamic_data_192600_205440.h5",
            218279: f"{data_dir}/training_dynamic_data_205440_218280.h5",
            231119: f"{data_dir}/training_dynamic_data_218280_231120.h5",
            243959: f"{data_dir}/training_dynamic_data_231120_243960.h5",
            256799: f"{data_dir}/training_dynamic_data_243960_256800.h5",
            257548: f"{data_dir}/training_dynamic_data_256800_257549.h5",
        }

    else:  # split=="test"

        # XXX: has to be changed when using the proper test dataset.
        # Dictionary mapping index to file path
        idx_h5_mapping = {
            12839: f"{data_dir}/testing_dynamic_data_0_12840.h5",
            25679: f"{data_dir}/testing_dynamic_data_12840_25680.h5",
            38519: f"{data_dir}/testing_dynamic_data_25680_38520.h5",
            51359: f"{data_dir}/testing_dynamic_data_38520_51360.h5",
            64199: f"{data_dir}/testing_dynamic_data_51360_64200.h5",
            77039: f"{data_dir}/testing_dynamic_data_64200_77040.h5",
            89879: f"{data_dir}/testing_dynamic_data_77040_89880.h5",
            102719: f"{data_dir}/testing_dynamic_data_89880_102720.h5",
            109460: f"{data_dir}/testing_dynamic_data_102720_109461.h5",
        }

    # Get the sorted keys
    sorted_keys = sorted(idx_h5_mapping.keys())

    # Find the index where idx should be inserted to maintain sorted order
    idx_index = bisect.bisect_left(sorted_keys, idx)
    if idx_index == 0:
        preceeding_index = -1
    else:
        preceeding_index = sorted_keys[idx_index - 1]

    # Get the closest key
    h5_key = sorted_keys[idx_index]

    # Return the file path corresponding to the closest key
    return idx_h5_mapping[h5_key], preceeding_index


def get_input_channels_count(time_steps, bands, aux_data=None, normalize_bands=False):
    """Get the number of input channels based on the time_steps, bands and aux_data.

    Args:
        time_steps: number of time steps
        bands: list of bands
        aux_data: list of auxillary data

    Returns:
        number of input channels

    """
    if normalize_bands:
        no_channels = 1
    else:
        no_channels = len(bands)

    if time_steps == 1 or time_steps == 4:

        if aux_data is None:
            return no_channels
        else:
            for data in aux_data:
                if data == "BIOMES":
                    no_channels += 15
                elif data == "LANDCOVER":
                    no_channels += 11
                elif data == "DEM":
                    no_channels += 1
                elif data == "LOCATION":
                    no_channels += 3
                elif data == "TIME":
                    no_channels += 4

            return no_channels
        


def save_experiment_logs(args, best_metric_logger, WEIGHTS_LOGS_DIR):
    """
    Save the hyperparameters and final metrics to a yaml file.
    """

    # training_best_epoch , training_best_metrics = best_metric_logger.best_epoch, best_metric_logger.best_epoch_metrics
    test_metrics = best_metric_logger.test_metrics
    log_dict ={**vars(args) **test_metrics}

    # Convert tensors to python numbers
    for key, value in log_dict.items():
        if isinstance(value, torch.Tensor):
            log_dict[key] = value.item()



    # Define filename
    filename = f'{WEIGHTS_LOGS_DIR}/args_metrics.yaml'

    if not os.path.exists(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save hyperparameters and final metrics to a yaml file
    with open(filename, 'w') as outfile:
        yaml.dump(log_dict, outfile, default_flow_style=False, allow_unicode=True)

    print(f"Saved hyperparameters and final metrics to {filename}")


def get_random_crop(array_height: int, array_width: int, crop_size: int):
    """Computes the indices for a random crop window of the desired size

    We need to ensure that
        - the crop doesn't exceed the arrays extent
        - the center values containing the AGBD information are still included
    """

    row_start_max = array_height - crop_size
    col_start_max = array_width - crop_size

    row_start = random.choice(range(0, row_start_max))
    col_start = random.choice(range(0, col_start_max))

    row_stop = row_start + crop_size
    col_stop = col_start + crop_size

    assert (row_stop - row_start) == (col_stop - col_start) == crop_size

    return row_start, row_stop, col_start, col_stop


def get_crop_indices_for_val(idx):
    pass


def get_crop_indices_for_test(idx):
    pass