import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
log_path = os.path.join("logs", "app.log")
if os.path.exists(log_path):
    os.remove(log_path)

from src.utils import load_yaml
from data import dataloading
import importlib
from src.utils.utils import get_all_feature_maps
from src.scores.exp_score import calculate_exp_score
import numpy as np
from src.engine.trainer import train_model
from src.utils.utils import save_model,save_accuracy,save_exp_score
from tensorflow.keras.models import load_model
from src.logging.logger import get_logger
from data.random_gaussion import random_gaussian_input
import glob
import tensorflow as tf
import random
make_logger = get_logger(__name__)
# Set reproduction 
seed_value = 42  # your specified seed number
random.seed(seed_value)                    # Python built-in random module
np.random.seed(seed_value)                 # NumPy random seed
tf.random.set_seed(seed_value)
make_logger = get_logger(__name__)

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

# Dynamically import the model module based on the model name from the config
model_module = f"src.models.{model_name}"
create_model = importlib.import_module(model_module).create_model

print('model_name:', model_name)



# Instantiate model
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
model = create_model(database_name=dataset_name, model_name=model_name, model_input_shape=input_shape, num_classes=num_classes, optimizer=optimizer, show_summary=show_summary)

# Get feature_maps before training
# Step 1: Generate random Gaussian input
data_input = random_gaussian_input(batch_size=batch_size, input_shape=input_shape)  # Generate random Gaussian input

# Step 2: Get feature maps from the model
feature_dict = get_all_feature_maps(model, data_input, show_maps=True, remove_unnecessary=True)

# Step 3: Get expressivity_score before training
expressivity_score_df = calculate_exp_score(model_name=model_name,
                                                                feature_maps_dict=feature_dict,
                                                                constant=small_constant,
                                                                show_exp_score=True)
# Save expressivity scores before training

save_exp_score(expressivity_score_df, model_name, dataset_name)

if __name__ == "__main__":
    pass
