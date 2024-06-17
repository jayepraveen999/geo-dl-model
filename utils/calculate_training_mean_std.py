import h5py
import numpy as np
import glob
import yaml


def calculate_nanmean_nanstd(h5_files):

    print(f"Total no. of h5 files found are: {len(h5_files)}")

    for h5_file in h5_files:
        h5_file = h5py.File(h5_file, "r+")

        # check if the key ahi_stat_p_updated is already present in the h5 file
        if "ahi_stat_p_updated" in h5_file["input_features"]:
            print(f"The key ahi_stat_p_updated is already present in the: {h5_file}")
            h5_file.close()
            continue

        len_dataset = len(h5_file["input_features"]["ahi_data"])
        # len_dataset = 1 #for testing
        ahi_stat_p_updated = np.empty((len_dataset, 6, 2), dtype=np.float32)
        for i in range(len_dataset):
            ahi_data = h5_file["input_features"]["ahi_data"][i, 0, :, :]
            nan_mean = np.nanmean(ahi_data, axis=(1, 2))
            nan_std = np.nanstd(ahi_data, axis=(1, 2))
            ahi_stat_p_updated[i, :, 0] = nan_mean
            ahi_stat_p_updated[i, :, 1] = nan_std

        h5_file["input_features"].create_dataset(
            "ahi_stat_p_updated", data=ahi_stat_p_updated
        )
        print(f"Successfully added the ahi_stat_p_updated key to the: {h5_file}")
        h5_file.close()


def calculate_global_nanmean_nanstd(h5_files):
    """
    This function calculates the global mean and std of the ahi_stat_p_updated feature
    """
    # we consider mean of means and mean of stds for all bands as the final value

    # first calculate the mean of means and mean of stds for all bands for each h5 file
    nan_stacks = []
    for h5_file in h5_files:
        h5_file = h5py.File(h5_file, "r")

        # sample 6212 has complete nans so this reason we still use nanmean here: Either delete this sample or use nanmean
        nan_stacks.append(
            np.nanmean(h5_file["input_features"]["ahi_stat_p_updated"], axis=0)
        )
        h5_file.close()

    # convert the list of arrays to a single array using np.stack
    nan_stacks = np.stack(nan_stacks)

    # Calculate the global mean and std
    nan_mean_std = np.nanmean(nan_stacks, axis=0)

    global_mean, global_std = (
        np.transpose(nan_mean_std)[0],
        np.transpose(nan_mean_std)[1],
    )
    return global_mean, global_std


if __name__ == "__main__":

    h5_dir = "/home/jayendra/thesis/geo-dl-fire-detection/train_test_split_data_files/train_split/dynamic_files/"
    h5_files = glob.glob(h5_dir + "*.h5")

    calculate_nanmean_nanstd(h5_files)

    # get band-wise global mean and std
    global_mean, global_std = calculate_global_nanmean_nanstd(h5_files)

    # write these values to a .yaml file in the config directory with name global_mean_std.yaml
    with open(
        "/home/jayendra/thesis/geo-dl-fire-detection-model/config/global_mean_std.yaml",
        "w",
    ) as file:
        yaml.dump(
            {"global_mean": global_mean.tolist(), "global_std": global_std.tolist()},
            file,
        )
    print(
        "Successfully written the global mean and std to the global_mean_std.yaml file"
    )
