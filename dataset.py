"""ActiveFire dataset."""

import glob
import h5py
import random
import datetime
import yaml
from typing import Callable, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from utils.helper_functions import (
    get_standardized_data,
    get_one_hot_encoding,
    get_4326_fromh8,
)

# get global mean and std of all bands for training data to standardize the data
config_file = "config/global_mean_std.yaml"

with open(config_file, "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

GLOBAL_BAND_MEANS = config["GLOBAL_MEAN_SUBSET"]
GLOBAL_BAND_STDS = config["GLOBAL_STD_SUBSET"]

TRAINING_DATASET_SIZE = config[
    "TRAINING_DATA_SIZE"
]  # XXX: has to be changed when using the whole dataset. Then change this name to TRAINING_DATA_SIZE/TRAINING_DATA_SIZE_PROTOTYPE
TESTING_DATASET_SIZE = config[
    "TESTING_DATA_SIZE"
]# XXX: has to be changed when using the test dataset. Then change this name to TEST_DATA_SIZE


class H8Dataset(Dataset):
    """Dataset instance for the Active Fire dataset."""

    def __init__(
        self,
        data_dir: str,
        static_data_dir: str,
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        split: str = "train",
        time_steps: int = 1,
        bands: list = [0],
        aux_data: Optional[list] = None,
        normalize_bands: bool = False,
    ):
        """
        Initialize ActiveFire dataset instance introduced in "Active fire detection in Landsat-8 imagery: A large-scale dataset and a deep-learning study".

        Args:
            data_dir (str): Path to the traning/valdiation data or test data
            static_data_dir (str): Path to the static data
            transforms (Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]], optional): Transformations. Defaults to None.
            split (str, optional): Split "train" "val" or "test". Defaults to "train".
            time_steps (int): Time steps to use. Should be either 1 or 4. Defaults to 1.
            bands (list): Bands to use. Defaults to [0].
            aux_data (list, optional): Auxiliary data to use. Defaults to empty list [].
            normalize_bands (bool, optional): Normalize bands. Defaults to False.

        """

        # check if the valid arguments are passed
        assert split in [
            "train",
            "val",
            "test",
        ], f"Split should be either 'train', 'val' or 'test' but got {split}"
        assert time_steps in [
            1,
            4,
        ], f"Time steps should be 1 or 4 to but got {time_steps}"
        assert all(
            x in range(0, 6) for x in bands
        ), f"Bands should be in the range of 0 to 5 but got {bands}"
        if aux_data is not None:
            assert all(
                x in ["DEM", "LANDCOVER", "BIOMES", "LOCATION", "TIME"]
                for x in aux_data
            ), f"Auxiliary data should be in ['DEM', 'LANDCOVER', 'BIOMES', 'LOCATION', 'TIME'] but got {aux_data}"

        self.split = split
        self.normalize_bands = normalize_bands
        if self.split == "train" or self.split == "val":
            self.data_dir = data_dir
            self.ids = []
            for i in range(0, TRAINING_DATASET_SIZE):
                self.ids.append(i)

            random.seed(42)
            random.shuffle(self.ids) # shuffle the ids to have random samples for train and val sets

            total_training_samples = len(self.ids)
            train_idx = int(0.8 * total_training_samples)

            if self.split == "train":
                self.ids = self.ids[:train_idx]
                print("Train", len(self.ids))
            if self.split == "val":
                self.ids = self.ids[train_idx:]     
                print("Validation", len(self.ids))

        if self.split == "test":
            self.data_dir = data_dir
            self.ids = []

            # XXX: has to be changed when using the proper test dataset.
            for i in range(0, TESTING_DATASET_SIZE):
                self.ids.append(i)
            
            print("Test", len(self.ids))

        self.transforms = transforms
        self.bands = bands
        self.time_steps = time_steps

        self.global_band_means_stds = np.column_stack(
            (GLOBAL_BAND_MEANS, GLOBAL_BAND_STDS)
        )
        self.aux_data = aux_data
        self.static_data_h5_path = sorted(
            glob.glob(f"{static_data_dir}/*static_data.h5")
        )

        # get the unique classes of landcover and biomes which are used for one-hot encoding
        if self.aux_data is not None:
            self.lc_classes = np.array(
                [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100], dtype=np.int8
            )
            self.biomes_classes = np.array(
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], dtype=np.int8
            )

    def __len__(self):
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset

        """
        return len(self.ids)

    def __getitem__(self, idx):
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index

        """
        # get the file which is related to the index
        h5_path = self.data_dir
        h5_idx = self.ids[idx]

        # open the h5 file and read the corresponding image and mask of the h5_idx
        with h5py.File(h5_path, "r") as f:
            # get ahi_data for h5_idx
            idx_sliced_sample = f["input_features"]["ahi_data"][
                h5_idx, 0 : self.time_steps, self.bands, :, :
            ]

            # get the raster window id for h5_idx (used to get auxiliary data from static data h5 file)
            idx_rw_id = f["input_features"]["raster_window_id"][h5_idx]

            # get timestamps for h5_idx
            idx_timestamps = [
                timestamp[0].decode()
                for timestamp in f["input_features"]["timestamps_str"][h5_idx]
            ]  # shape of timestamps (4,)
            # get sample year from timestamps for h5_idx
            idx_sample_year = idx_timestamps[0].split("/")[0]

            # get mask for h5_idx
            mask = f["labels"]["labels_data"][h5_idx, :, :]  # shape of mask (256, 256)
            mask = mask[np.newaxis, :, :]

        # standardize the image using global band mean and standard deviation
        global_means_stds = self.global_band_means_stds[self.bands]
        global_mean, global_std = list(global_means_stds[:, 0]), list(
            global_means_stds[:, 1]
        )
        standardized_image = get_standardized_data(
            idx_sliced_sample, global_mean, global_std
        )  # shape of standardized_image (t, b, 256, 256) where t is timesteps and b is bands

        if self.normalize_bands:
            standardized_image = (idx_sliced_sample[:,0,:,:] - idx_sliced_sample[:,1,:,:]) / (idx_sliced_sample[:,0,:,:] + idx_sliced_sample[:,1,:,:])
            standardized_image = standardized_image[np.newaxis,:,:,:]
    

        if self.aux_data is not None:
            with h5py.File(self.static_data_h5_path[0], "r") as a_f:

                for aux in self.aux_data:
                    if aux == "DEM":
                        # needs standardization
                        dem = a_f["static_features"]["copdem"][idx_rw_id]
                        standardized_dem = (dem - np.mean(dem)) / np.std(
                            dem
                        )
                        standardized_dem = standardized_dem[np.newaxis, :, :]
                        # broadcasting the standardized_dem to the shape of standardized_image
                        standardized_dem = np.broadcast_to(
                            standardized_dem, (standardized_image.shape[0], 1, 256, 256)
                        )
                        standardized_image = np.concatenate(
                            (standardized_image, standardized_dem), axis=1
                        )

                    if aux == "LANDCOVER":
                        if idx_sample_year == "2020":
                            landcover = a_f["static_features"]["landcover_2020"][
                                idx_rw_id, :, :
                            ]

                        else:
                            landcover = a_f["static_features"]["landcover_2021"][
                                idx_rw_id, :, :
                            ]

                        landcover_one_hot = get_one_hot_encoding(
                            self.lc_classes, landcover
                        )  # shape of landcover_one_hot (11, 256, 256)
                        landcover_one_hot = landcover_one_hot[np.newaxis, :, :, :]
                        # broadcasting the landcover_one_hot to the shape of standardized_image
                        landcover_one_hot = np.broadcast_to(
                            landcover_one_hot,
                            (standardized_image.shape[0], 11, 256, 256),
                        )
                        standardized_image = np.concatenate(
                            (standardized_image, landcover_one_hot), axis=1
                        )

                    if aux == "BIOMES":
                        biomes = a_f["static_features"]["biomes"][idx_rw_id, :, :]
                        biomes_one_hot = get_one_hot_encoding(
                            self.biomes_classes, biomes
                        )  # shape of biomes_one_hot (15, 256, 256)
                        biomes_one_hot = biomes_one_hot[np.newaxis, :, :, :]
                        # broadcasting the biomes_one_hot to the shape of standardized_image
                        biomes_one_hot = np.broadcast_to(
                            biomes_one_hot, (standardized_image.shape[0], 15, 256, 256)
                        )
                        standardized_image = np.concatenate(
                            (standardized_image, biomes_one_hot), axis=1
                        )

                    if aux == "LOCATION":
                        location_lat_t = a_f["static_features"]["lat"][idx_rw_id, :, :][
                            0
                        ]
                        location_lon_t = a_f["static_features"]["lon"][idx_rw_id, :, :][
                            0
                        ]

                        # change from h8 projection to 4326
                        location_lon, location_lat = get_4326_fromh8(
                            location_lon_t, location_lat_t
                        )

                        # transform coordinates to range -1 to 1 and make longitude cyclic
                        lat = location_lat / 90
                        lon_sin = np.sin(2 * np.pi * location_lon / 360)
                        lon_cos = np.cos(2 * np.pi * location_lon / 360)
                        coords = np.stack([lat, lon_sin, lon_cos], axis=0)
                        coords = coords[
                            np.newaxis, :, :, :
                        ]  # shape of coords (3, 256, 256)
                        # broadcasting the coords to the shape of standardized_image
                        coords = np.broadcast_to(
                            coords, (standardized_image.shape[0], 3, 256, 256)
                        )
                        standardized_image = np.concatenate(
                            (standardized_image, coords), axis=1
                        )

                    if aux == "TIME":
                        doy = (
                            datetime.datetime.strptime(
                                idx_timestamps[0][:10], "%Y/%m/%d"
                            )
                            .timetuple()
                            .tm_yday
                        )
                        hod = int(idx_timestamps[0][16:18])

                        doy_sin = np.sin(2 * np.pi * doy / 365.0)
                        doy_cos = np.cos(2 * np.pi * doy / 365.0)

                        hod_sin = np.sin(2 * np.pi * hod / 24.0)
                        hod_cos = np.cos(2 * np.pi * hod / 24.0)

                        arr_doy_sin = np.full((256, 256), doy_sin, dtype=np.float32)
                        arr_doy_cos = np.full((256, 256), doy_cos, dtype=np.float32)

                        arr_hod_sin = np.full((256, 256), hod_sin, dtype=np.float32)
                        arr_hod_cos = np.full((256, 256), hod_cos, dtype=np.float32)

                        doy_hod = np.stack(
                            [arr_doy_sin, arr_doy_cos, arr_hod_sin, arr_hod_cos], axis=0
                        )  # shape of doy_hod (4, 256, 256)

                        doy_hod = doy_hod[np.newaxis, :, :, :]
                        # broadcasting the doy to the shape of standardized_image
                        doy_hod = np.broadcast_to(
                            doy_hod, (standardized_image.shape[0], 4, 256, 256)
                        )

                        standardized_image = np.concatenate(
                            (standardized_image, doy_hod), axis=1
                        )

        # remove the first axis of the standardized_image and mask
        if self.time_steps == 1:
            # remove the first axis of the standardized_image when time_steps is 1 to have same shape for both image and mask
            standardized_image = standardized_image[0]
        else:
            # add one more axis to the mask when time_steps is 4 to have same shape for both image and mask
            mask = mask[np.newaxis, :, :, :]
            
        # convert to torch tensors
        standardized_image = torch.from_numpy(standardized_image).float()
        mask = torch.from_numpy(mask).to(
            torch.int8
        )  # device for the tensors of image and mask are not assigned to GPU as pl.Trainer in main.py does that automatically when specified GPUs list in devices argument

        # replace nan, inf, -inf with 0. Else the loss is returning nan
        standardized_image = standardized_image.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

        sample = {"image": standardized_image, "mask": mask}

        # apply transformations if any
        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample["image"], sample["mask"], h5_idx
