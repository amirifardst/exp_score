import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
log_path = os.path.join("logs", "app.log")
if os.path.exists(log_path):
    os.remove(log_path)

from src.utils import load_yaml
from data import dataloading
from src.models.simple_cnn import create_model
from src.utils.utils import get_feature_maps
from src.utils.exp_score import calculate_exp_score
import numpy as np
from src.engine.trainer import train_model



config_dict = load_yaml.main(config_file_dir="yamls/config.yml")

model_name = config_dict["model"]["name"]
dataset_name = config_dict["dataset"]["name"]
view_sample = config_dict["dataset"]["view_sample"]
num_classes = config_dict["model"]["num_classes"]
input_shape = config_dict["dataset"]["input_shape"]
show_summary = config_dict["model"]["show_summary"]
small_constant = config_dict["expressivity"]["small_constant"]
optimizer = config_dict["training"]["optimizer"]
epochs = config_dict["training"]["epochs"]
batch_size = config_dict["training"]["batch_size"]

print('model_name:', model_name)
# Load dataset
(train_images, train_labels), (test_images, test_labels) = dataloading.load_data(dataset_name, view_sample)

# Preprocess images
train_images, test_images = dataloading.normalize_images(train_images, test_images)



# Instantiate model
if model_name == "simple_cnn":
    model = create_model(input_shape=input_shape, optimizer=optimizer, num_classes=num_classes, show_summary=show_summary)

# Get feature_maps before training
data_input = train_images[19]
data_input = np.expand_dims(data_input, axis=0) # Expand to have (None, 32, 32, 3) dimension as an example
bef_train_feature_maps_dict, bef_train_feature_maps_df = get_feature_maps(model, data_input, show_maps=True)
# Get expressivity_score before training
bef_train_exp_score_dict, bef_train_pca_var_dict, bef_train_model_exp_score_sum = calculate_exp_score(model_name=model_name,
                                                                                           state='bef_train',
                                                                                           feature_maps_dict=bef_train_feature_maps_dict,
                                                                                           constant=small_constant,
                                                                                           show_exp_score=True)

# Train model
model, result = train_model(model, train_images, train_labels,
                      test_images, test_labels, dataset_name, epochs, batch_size)

# Get feature_maps after training
aft_train_feature_maps_dict, aft_train_feature_maps_df = get_feature_maps(model, data_input, show_maps=True)
# Get expressivity_score after training
aft_train_exp_score_dict, aft_train_pca_var_dict, aft_train_model_exp_score_sum = calculate_exp_score(model_name=model_name,
                                                                                           state='aft_train',
                                                                                           feature_maps_dict=aft_train_feature_maps_dict,
                                                                                           constant=small_constant,
                                                                                           show_exp_score=True)

if __name__ == "__main__":
    pass
