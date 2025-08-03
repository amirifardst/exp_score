import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
log_path = os.path.join("logs", "app.log")
if os.path.exists(log_path): # Remove existing log file if it exists
    os.remove(log_path)

################################################# Import necessary modules #################################################
from src.utils import load_yaml
from src.utils.utils import get_all_feature_maps
from src.scores.exp_score import calculate_exp_score_nas
import numpy as np
from src.utils.correlation_calculator import get_kendall
from src.engine.trainer import train_model
from src.utils.utils import save_exp_score,save_sorted_accuracy,save_sorted_exp_score
from src.logging.logger import get_logger
from data.random_gaussion import random_gaussian_input
import tensorflow as tf
import random
import joblib
################################################# Set random seed #################################################
seed_value = 42  # your specified seed number
random.seed(seed_value)                    # Python built-in random module
np.random.seed(seed_value)                 # NumPy random seed
tf.random.set_seed(seed_value)
make_logger = get_logger(__name__)

################################################# Load configuration #################################################
config_dict = load_yaml.main(config_file_dir="yamls/config.yml")
model_name = config_dict["model"]["name"]
show_summary = config_dict["model"]["show_summary"]
num_classes = config_dict["model"]["num_classes"]
pretrained = config_dict["model"]["pretrained"]
dataset_name = config_dict["dataset"]["name"]
view_sample = config_dict["dataset"]["view_sample"]
input_shape = config_dict["dataset"]["input_shape"]
epochs = config_dict["training"]["epochs"]
batch_size = config_dict["training"]["batch_size"]
lr = config_dict["training"]["learning_rate"]
small_constant = config_dict["expressivity"]["small_constant"]

################################################# Step 1: Get feature maps #################################################
fmap_list = joblib.load('my_tensor_list.pkl')
layer_names_list = joblib.load('my_layer_names_list.pkl')
accuracy_list = joblib.load('my_test_acc_list.pkl')
model_names_list = joblib.load('my_architecture_names.pkl')

################################################# Step 2: Get expressivity_score #################################################
expressivity_score_df = calculate_exp_score_nas(model_names=model_names_list,
                                                 layer_names=layer_names_list,
                                                 feature_maps_list=fmap_list,
                                                 constant=small_constant,
                                                 show_exp_score=True)
# Save expressivity scores before training
method = "median"  # or "mean", depending on your preference
type_of_score = "Expressivity Score"  # or any other score type you want to use
################################################# Step 3: Save expressivity score for each model #################################################
for model_name, expressivity_score_df in expressivity_score_df.items():
    save_exp_score(expressivity_score_df, model_name, dataset_name)

################################################# Step 4: Save sorted accuracy and exp score of all models #################################################
accuracy_df = save_sorted_accuracy(accuracy_list, model_names_list, dataset_name)
expressivity_score_df = save_sorted_exp_score(method=method, type_of_score=type_of_score, database_name=dataset_name)

################################################# Step 5: Get Tau Kendall performance #################################################
tau, p_value, merged_df = get_kendall(accuracy_df, expressivity_score_df, method=method, type_of_score=type_of_score, database_name=dataset_name)

if __name__ == "__main__":
    pass
