# Deep Learning Model for Active Fire Detection on Geostationary Satellite Data

## Setup Environment

1. **Install Poetry**
   - Ensure you have Poetry version 1.3.2 installed. For installation instructions, refer to the [Poetry documentation](https://python-poetry.org/docs/#installation).
   - Verify the installation with:
     ```sh
     poetry --version
     ```

2. **Install Dependencies**
   - Run the following command to install all required packages for this project:
     ```sh
     poetry install
     ```

## Download Data

- Details on the whole data preparation and generation is available at [geo-dl-data](https://github.com/jayepraveen999/geo-dl-data)
- create a `data` folder in the project dir and download all necessary data files from [google drive](https://drive.google.com/drive/folders/1J7VqqCQds-x5ulbs3LTF7SPhC3Cy9BuZ?usp=drive_link)
- Incase you change folder names or file names, please adapt to those names in the code.

## Running the Project

### Activate Environment

- Activate the environment using:
  ```sh
  poetry shell
  ```

### Running `main.py`

- The script `main.py` accepts several arguments for input features and hyperparameters based on the best performing model.

#### Using Pretrained Weights

- To test the model on the test dataset using the best performing model weights, set the `--bpm-ckpt-path` argument to `True`. This will bypass training and use the pre-existing weights:
  ```sh
  python main.py --bpm-ckpt-path=True
  ```

- To train the model from scratch, set the `--bpm-ckpt-path` argument to `False`:
  ```sh
  python main.py --bpm-ckpt-path=False
  ```

- You can also adjust the hyperparameters as needed.

### Outputs and Logs

- When you run experiments, the corresponding weight files are saved in a timestamped subdirectory under the `weights_and_logs` directory.
- Test dataset statistics, configurations, hyperparameters, and other relevant input arguments are saved as`arg_metrics.yaml` file in the same timestamped subdirectory.
- Incase if you train from scratch it necessary that you want to visualize the metrics. We use tensorboard to log and visualize different metrics. Run  `tensorboard --logdir ./weights_and_logs/lightning_logs` in the terminal and tensorboard should be visible at http://localhost:6006/ unless this port is occupied. 

### Test Dataset Analysis

- Statistics for the test dataset are displayed, and plots are saved in `outputs/test_dataset_plots`.

### Evaluation Dataset

- Predictions for the evaluation dataset are added to the existing file.
- To visualize these predictions, run:
  ```sh
  python utils/create_evaluation_plots.py
  ```
- Plots will be saved in the `output/evaluation_plots/` directory.
