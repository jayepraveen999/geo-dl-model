import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches

plt.rcParams["font.family"] = 'serif'
plt.rcParams["font.size"] = 11

# Load the grouped data
eval_grouped_data = "data/ground_truth_data/bushfires_gad_preprocessed_flat_fire_i_e_grouped_2022_final.h5"
output_dir = "outputs/evaluation_plots/"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define a function to set the background color and remove the border for the plot
def set_background_color_and_remove_border(ax, color):
    ax.set_facecolor(color)
    ax.figure.set_facecolor(color)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

with h5py.File(eval_grouped_data, 'r') as f:
    # Iterate through all the keys in the h5 file
    for key in f.keys():
        ground_truth = f[key]["bushfire_label_data_unions"][:]
        prediction = f[key]["predictions_unions"][:]
        h8_predictions = f[key]["h8_fire_product_data_unions"][:]

        # Create empty RGB images of the same size for both predictions
        color_map_pred = np.ones((256, 256, 3))  # Initialize with white
        color_map_h8 = np.ones((256, 256, 3))    # Initialize with white
        color_map_gt = np.ones((256, 256, 3))    # Initialize with white

        # True Positive (TP) - Green for model predictions
        color_map_pred[(ground_truth == 1) & (prediction == 1)] = [0, 1, 0]

        # False Positive (FP) - Red for model predictions
        color_map_pred[(ground_truth == 0) & (prediction == 1)] = [1, 0, 0]

        # False Negative (FN) - Blue for model predictions
        color_map_pred[(ground_truth == 1) & (prediction == 0)] = [0, 0, 1]

        # True Positive (TP) - Green for H8 predictions
        color_map_h8[(ground_truth == 1) & (h8_predictions == 1)] = [0, 1, 0]

        # False Positive (FP) - Red for H8 predictions
        color_map_h8[(ground_truth == 0) & (h8_predictions == 1)] = [1, 0, 0]

        # False Negative (FN) - Blue for H8 predictions
        color_map_h8[(ground_truth == 1) & (h8_predictions == 0)] = [0, 0, 1]

        # Ground Truth - Green
        color_map_gt[ground_truth == 1] = [0, 1, 0]

        fig, axes = plt.subplots(1, 4, figsize=(18, 6))

        # Set background color and remove border for each subplot
        bg_color = 'white'

        def add_black_border(ax):
            rect = patches.Rectangle((0, 0), 256, 256, linewidth=1.5, edgecolor='black', facecolor='none')
            rect.set_clip_on(False)  # Ensure rectangle is drawn fully
            ax.add_patch(rect)

                # Define legend elements
        legend_elements = [patches.Patch(facecolor=[0, 1, 0], edgecolor='black', label='TP'),
                           patches.Patch(facecolor=[1, 0, 0], edgecolor='black', label='FP'),
                           patches.Patch(facecolor=[0, 0, 1], edgecolor='black', label='FN')]

        legend_elements_gt = [patches.Patch(facecolor=[0, 1, 0], edgecolor='black', label='TP')]

        # Plot ground truth
        axes[0].imshow(color_map_gt, interpolation='nearest')
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')
        set_background_color_and_remove_border(axes[0], bg_color)
        add_black_border(axes[0])
        axes[0].legend(handles=legend_elements_gt, loc='upper right', fontsize=7)

        # Plot color-coded model predictions
        axes[1].imshow(color_map_pred, interpolation='nearest')
        axes[1].set_title('Model Prediction')
        axes[1].axis('off')
        set_background_color_and_remove_border(axes[1], bg_color)
        add_black_border(axes[1])
        axes[1].legend(handles=legend_elements, loc='upper right', fontsize=7)

        # Plot color-coded H8 predictions
        axes[2].imshow(color_map_h8, interpolation='nearest')
        axes[2].set_title('H8 Active Fire Product')
        axes[2].axis('off')
        set_background_color_and_remove_border(axes[2], bg_color)
        add_black_border(axes[2])
        axes[2].legend(handles=legend_elements, loc='upper right', fontsize=7)

        # Hide the fourth subplot (optional)
        axes[3].axis('off')
        set_background_color_and_remove_border(axes[3], bg_color)

        # Save the plot
        plot_filename = os.path.join(output_dir, f"{key.replace('/', '_')}_predictions.png")
        plt.savefig(plot_filename, bbox_inches='tight', facecolor=fig.get_facecolor())
        # plt.show()
        plt.close(fig)

        print(f"Saved plot for {key} to {plot_filename}")
        # break
