import torch 
import numpy as np
import os
import yaml
import argparse
import h5py
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from trainer import UNetTrainer
from utils.helper_functions import get_eval_standardized_data

# get global mean and std of all bands for training data to standardize the data
config_file = "config/global_mean_std.yaml"

with open(config_file, "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

GLOBAL_BAND_MEANS = config["GLOBAL_MEAN_SUBSET"]
GLOBAL_BAND_STDS = config["GLOBAL_STD_SUBSET"]


class EvalDataset(Dataset):
    """
    Dataset class for evaluation on ground truth dataset
    """

    def __init__(self, data_dir, split="eval", bands=[0, 4, 5]):
        """
        Initialize the dataset
        """
        self.data_dir = data_dir
        self.split = split
        self.bands = bands
        self.global_band_means_stds = np.column_stack(
            (GLOBAL_BAND_MEANS, GLOBAL_BAND_STDS)
        )

        
    def __len__(self):
        """
        Return the length of the dataset
        """
        with h5py.File(self.data_dir, "r") as f:
            return len(f["ahi_data"])

    def __getitem__(self, idx):
        """
        Load the data from the h5py file and standardize the image using global band mean and standard deviation
        """

        with h5py.File(self.data_dir, "r") as f:
            data = f["ahi_data"][idx, self.bands, :, :]

        # standardize the image using global band mean and standard deviation
        global_means_stds = self.global_band_means_stds[self.bands]
        global_mean, global_std = list(global_means_stds[:, 0]), list(
            global_means_stds[:, 1]
        )
        standardized_image = get_eval_standardized_data(data, global_mean, global_std)
        # create tensor from a numpy array
        standardized_image = torch.from_numpy(standardized_image).float()
        
        # replace nan, inf, -inf with 0. Else the loss is returning nan
        standardized_image = standardized_image.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

        return standardized_image, idx


def run_inference_on_ground_truth(args):
    """
    Run inference on the provided ground truth data using best performing model weights and save the predictions in the same h5py file under the key "predictions"
    
    """
    eval_dataset = EvalDataset(data_dir=args.data_dir, bands=args.bands)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    model = UNetTrainer(
        model_type="UNet2D",
        input_ch=len(args.bands),
        enc_ch=args.encoder_ch,
    )
    
    checkpoint = torch.load('bpm_weight_file/2024-05-25_14-18_32epoch=319-val_iou=0.24.ckpt')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    predictions = [] 
    with torch.no_grad():
        for sample, idx in eval_dataloader:
            pred = model.forward(sample)
            pred_mask = (pred > 0.5).float()
            predictions.append(pred_mask[0].numpy())
            # break

    predictions = np.concatenate(predictions, axis=0)

    # add this key to the h5py
    with h5py.File(args.data_dir, "a") as f:
        f.create_dataset("predictions", data=predictions, compression="lzf")
def create_grouped_eval_dataset():
    """
    Create a new h5py file with grouped data for each fire which is better used to understand the model detections for each fire in ground truth data
    """
    if os.path.exists("data/ground_truth_data/bushfires_gad_preprocessed_flat_fire_i_e_grouped_2022_final.h5"):
        print("Grouped data already exists. Skipping the creation of grouped data...")
        return
    
    # Prepare a dictionary to hold data grouped by fire_id
    grouped_data = {}

    with h5py.File("data/ground_truth_data/bushfires_gad_preprocessed_flat_fire_i_e_2022_final.h5", 'r') as f:
        
        len_dataset = len(f["ahi_data"])
        # Iterate over each data entry
        for i in range(len_dataset):
            f_id = f["fire_id"][i] # assuming fire_id is a 2D array with one column
            r_w_id = f["raster_window_id"][i] # assuming raster_window_id is a 2D array with one column
            # merge id, r_w_id to create a unique key
            id = f"{f_id}_{r_w_id}"
            if id not in grouped_data:
                grouped_data[id] = {
                    'ahi_data': [],
                    'predictions': [],
                    'bushfire_label_data':[],
                    'h8_fire_product_data': [],
                    "cloud_mask_binary":[],
                    "timestamps":[],
                    "timestamps_index": [],
                    "area_ha": [],
                    "ignition_date": [],
                    "extinguish_date": [],
                    "fire_type": []
                    
                }
            
            # Append data to the appropriate group
            grouped_data[id]['ahi_data'].append(f["ahi_data"][i])
            grouped_data[id]['predictions'].append(f["predictions"][i])
            grouped_data[id]['bushfire_label_data'].append(f["bushfire_label_data"][i])
            grouped_data[id]['h8_fire_product_data'].append(f["h8_fire_product_data"][i])
            grouped_data[id]['cloud_mask_binary'].append(f["cloud_mask_binary"][i])
            grouped_data[id]['timestamps'].append(f["timestamps"][i])
            grouped_data[id]['timestamps_index'].append(f["timestamps_index"][i])
            grouped_data[id]['area_ha'].append(f["area_ha"][i])
            grouped_data[id]['ignition_date'].append(f["ignition_date"][i])
            grouped_data[id]['extinguish_date'].append(f["extinguish_date"][i])
            grouped_data[id]['fire_type'].append(f["fire_type"][i])


    # Create a new H5 file
    with h5py.File("data/ground_truth_data/bushfires_gad_preprocessed_flat_fire_i_e_grouped_2022_final.h5", 'w') as new_file:
        for id, data in grouped_data.items():
            group = new_file.create_group(str(id))
            group.create_dataset('ahi_data', data=np.array(data['ahi_data']), compression = "lzf")
            group.create_dataset('predictions', data=np.array(data['predictions']), compression = "lzf")
            group.create_dataset('bushfire_label_data', data=np.array(data['bushfire_label_data']), compression = "lzf")
            group.create_dataset('h8_fire_product_data', data=np.array(data['h8_fire_product_data']), compression = "lzf")
            group.create_dataset('cloud_mask_binary', data=np.array(data['cloud_mask_binary']), compression = "lzf")
            group.create_dataset('timestamps', data=np.array(data['timestamps']), compression = "lzf")
            group.create_dataset('timestamps_index', data=np.array(data['timestamps_index']), compression = "lzf")
            group.create_dataset('area_ha', data=np.array(data['area_ha']), compression = "lzf")
            group.create_dataset('ignition_date', data=np.array(data['ignition_date']), compression = "lzf")
            group.create_dataset('extinguish_date', data=np.array(data['extinguish_date']), compression = "lzf")
            group.create_dataset('fire_type', data=np.array(data['fire_type']), compression = "lzf")

            # create unions of the data that we need to comapare
            group.create_dataset('predictions_unions', data = np.max(np.array(data['predictions']), axis=0), compression = "lzf")
            group.create_dataset('bushfire_label_data_unions', data = np.max(np.array(data['bushfire_label_data']), axis=0), compression = "lzf")
            group.create_dataset('h8_fire_product_data_unions', data = np.max(np.array(data['h8_fire_product_data']), axis=0), compression = "lzf")


if __name__ == "__main__":

    # load ground truth path and relevand information for dataloader 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/ground_truth_data/bushfires_gad_preprocessed_flat_fire_i_e_2022_final.h5")
    parser.add_argument("--bands", type=list, default=[0, 4])
    parser.add_argument(
        "--encoder-ch",
        default=(32, 64, 128, 256, 512, 1024, 2048),
        type=int,
        help="Encoder channels",
    )

    # parser.add_argument("--train_val_logging_images", type=int, default=[1,1])
    args = parser.parse_args()

    with h5py.File(args.data_dir, "a") as f:
        # look for the key, if it exists stop inference
        if "predictions" in f.keys():
            print("Predictions already exist for this file. Stopping inference on provided ground truth data...")
    

    run_inference_on_ground_truth(args)
    print("Inference on ground truth data completed successfully")

    create_grouped_eval_dataset(args)
   
