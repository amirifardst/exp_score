import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
log_path = os.path.join("logs", "app.log")
if os.path.exists(log_path):
    os.remove(log_path)

from src.utils import load_yaml

from data import dataloading
import importlib
from src.utils.utils import get_all_feature_maps
from src.scores.exp_score import calculate_exp_score_nas
import numpy as np
from src.utils.rank_pred import get_rank
from src.engine.trainer import train_model
from src.utils.utils import save_model,save_accuracy,save_exp_score,save_acc_nas,save_exp
from tensorflow.keras.models import load_model
from src.logging.logger import get_logger
from data.random_gaussion import random_gaussian_input
import glob
import tensorflow as tf
import random
import pickle
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



print('model_name:', model_name)




# Step 1: Generate random Gaussian input (done before)

# Step 2: Get feature maps from the model
import joblib
feature_list = joblib.load('my_tensor_list.pkl')
my_architecture_names = joblib.load('my_architecture_names.pkl')
my_layer_names = joblib.load('my_layer_names_list.pkl')
test_accc_list = joblib.load('my_test_acc_list.pkl')
# Step 3: Get expressivity_score before training
all_expressivity_score_df = calculate_exp_score_nas(model_names=my_architecture_names, layer_names=my_layer_names,
                                                     feature_maps_list=feature_list,
                                                     constant=small_constant,
                                                     show_exp_score=True)
# Save expressivity scores before training



for architecture_name, expressivity_score_df in all_expressivity_score_df.items():
    save_exp_score(expressivity_score_df, architecture_name, dataset_name)

save_acc_nas(test_accc_list,my_architecture_names, dataset_name)
save_exp()

tau, p_value, merged_df = get_rank(
    
    acc_df_dir=r"C:\Users\amirifam\OneDrive - STMicroelectronics\Desktop\Work\Projects_by_py\exp_score\results\cifar10\nas\nas_accuracy.csv",
    df_exp_dir=r"C:\Users\amirifam\OneDrive - STMicroelectronics\Desktop\Work\Projects_by_py\exp_score\results\cifar10\nas\architecture_mean_scores.csv"
)

if __name__ == "__main__":
    pass
