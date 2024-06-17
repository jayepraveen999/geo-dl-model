import numpy as np
from helper_functions import get_file_path, calculate_fire_fraction
import h5py


test_data_dir = "/home/jayendra/thesis/geo-dl-fire-detection/train_test_split_data_files/test_split/dynamic_files"
h5_dir = f"{test_data_dir}/"

fire_fraction = calculate_fire_fraction(h5_dir)
fires = [256 * 256 * x for x in fire_fraction]
test_fire_indices = [i for i, x in enumerate(fires) if x > 0.0]

print(f"Number of test samples with fire pixels: {len(test_fire_indices)}")
no_of_samples= len(test_fire_indices)
no_of_bands = 6
timeseries_length = 4
sample_width,sample_height = 256,256   
timestamps_str = np.empty((no_of_samples, timeseries_length, 1), dtype='S20')
ahi_data = np.empty((no_of_samples, timeseries_length, no_of_bands, sample_width,sample_height), dtype=np.float32)
cloud_mask_binary = np.empty((no_of_samples, sample_width,sample_height), dtype=np.int8)
ahi_stat_p = np.empty((no_of_samples, no_of_bands, 2), dtype=np.float32)
ahi_stat_p_c = np.empty((no_of_samples, timeseries_length, no_of_bands, 2), dtype=np.float32)
fire_fraction = np.empty((no_of_samples, 1), dtype=np.float32)
cloud_fraction = np.empty((no_of_samples, 1), dtype=np.float32)
raster_window_id = np.empty((no_of_samples, 1), dtype=np.int8)
labels_data = np.empty((no_of_samples, sample_width, sample_height), dtype=np.int8)

def get_subset_fires_h5_data():

    for idx, test_idx in enumerate(test_fire_indices):
        h5_path, preceeding_index = get_file_path(test_idx, split="test", data_dir="/home/jayendra/thesis/geo-dl-fire-detection/train_test_split_data_files/test_split/dynamic_files")
        # calculate h5_idx that is relative to the start of the h5 file
        h5_idx = test_idx - (preceeding_index + 1)
        # open the h5 file and read the corresponding image and mask of the h5_idx
        with h5py.File(h5_path, "r") as f:
            ahi_data[idx, ...] = f["input_features"]["ahi_data"][h5_idx]
            cloud_mask_binary[idx, ...] = f["input_features"]["cloud_mask_binary"][h5_idx]
            ahi_stat_p[idx, ...] = f["input_features"]["ahi_stat_p"][h5_idx]
            ahi_stat_p_c[idx, ...] = f["input_features"]["ahi_stat_p_c"][h5_idx]
            fire_fraction[idx, ...] = f["input_features"]["fire_fraction"][h5_idx]
            cloud_fraction[idx, ...] = f["input_features"]["cloud_fraction"][h5_idx]
            raster_window_id[idx, ...] = f["input_features"]["raster_window_id"][h5_idx]
            timestamps_str[idx, ...] = f["input_features"]["timestamps_str"][h5_idx]
            labels_data[idx, ...] = f["labels"]["labels_data"][h5_idx]


    return timestamps_str, ahi_data, cloud_mask_binary, ahi_stat_p, ahi_stat_p_c, fire_fraction, cloud_fraction, raster_window_id, labels_data


timestamps_str, ahi_data, cloud_mask_binary, ahi_stat_p, ahi_stat_p_c, fire_fraction, cloud_fraction, raster_window_id, labels_data = get_subset_fires_h5_data()


# write to hdf5
filename = f'/home/jayendra/thesis/geo-dl-fire-detection-model/data/subset_testing_dynamic_data.h5'
# mnt_filename = f'/mnt/research/datasets/thesis-jayendra/testing_hdf_files/testing_dynamic_data_{file_naming_start}_{file_naming_end}.h5'

with h5py.File(filename, 'w') as f:

    # create input features group
    input_features = f.create_group('input_features')

    input_features.create_dataset('timestamps_str', data=timestamps_str, chunks = (1,4,1), compression="lzf")
    input_features.create_dataset('ahi_data', data=ahi_data, chunks = (1,4,6,256,256), compression="lzf")
    input_features.create_dataset('cloud_mask_binary', data=cloud_mask_binary, chunks = (1,256,256), compression="lzf")
    input_features.create_dataset('ahi_stat_p', data=ahi_stat_p, chunks = (1,6,2), compression="lzf")
    input_features.create_dataset('ahi_stat_p_c', data=ahi_stat_p_c, chunks = (1,4,6,2), compression="lzf")
    input_features.create_dataset('fire_fraction', data=fire_fraction, chunks = (1,1), compression="lzf")
    input_features.create_dataset('cloud_fraction', data=cloud_fraction, chunks = (1,1), compression="lzf")
    input_features.create_dataset('raster_window_id', data=raster_window_id, chunks = (1,1), compression="lzf")
    # also write description for each dataset 
    input_features['timestamps_str'].attrs['description'] = "Timestamps for each sample. Includes Parents timestamp at index 0 and children timestamps at index 1,2,3"
    input_features['ahi_data'].attrs['description'] = "AHI data for each sample. Includes 6 bands for each timestamp"
    input_features['cloud_mask_binary'].attrs['description'] = "Cloud mask binary for each sample. Cloud mask indicates the mask for parent timestamp"
    input_features['ahi_stat_p'].attrs['description'] = "AHI statistics for parent timestamp. Includes mean and standard deviation for each band"
    input_features['ahi_stat_p_c'].attrs['description'] = "AHI statistics for parent and child timestamp. Includes mean and standard deviation for each band"
    input_features['fire_fraction'].attrs['description'] = "Fire fraction for each sample. Fire fraction indicates the fraction of fire pixels in the parent timestamp"
    input_features['cloud_fraction'].attrs['description'] = "Cloud fraction for each sample. Cloud fraction indicates the fraction of cloud pixels in the parent timestamp"
    input_features['raster_window_id'].attrs['description'] = "Raster window id for each sample. Raster window id indicates the window id in the raster image and used as foregin key to get the static features for each sample"

    # create labels group
    labels_group = f.create_group('labels')
    labels_group.create_dataset('labels_data', data=labels_data, chunks = (1,256,256), compression="lzf")
    labels_group['labels_data'].attrs['description'] = "Labels for each sample. Labels indicate the presence of fire pixels in the parent timestamp"



print("HDF5 file created successfully for subset of testing dynamic data.")