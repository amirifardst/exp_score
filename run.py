import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
log_path = os.path.join("logs", "app.log")
if os.path.exists(log_path):
    os.remove(log_path)

from src.utils import load_yaml
from data import dataloading
import importlib
from src.utils.utils import get_feature_maps
from src.utils.exp_score import calculate_exp_score
import numpy as np
from src.engine.trainer import train_model
from src.utils.utils import save_model,save_accuracy,save_exp_score
from tensorflow.keras.models import load_model
from src.logging.logger import get_logger
make_logger = get_logger(__name__)

config_dict = load_yaml.main(config_file_dir="yamls/config.yml")

model_name = config_dict["model"]["name"]
show_summary = config_dict["model"]["show_summary"]
num_classes = config_dict["model"]["num_classes"]
pretrained = config_dict["model"]["pretrained"]

dataset_name = config_dict["dataset"]["name"]
view_sample = config_dict["dataset"]["view_sample"]
input_shape = config_dict["dataset"]["input_shape"]

optimizer = config_dict["training"]["optimizer"]
epochs = config_dict["training"]["epochs"]
batch_size = config_dict["training"]["batch_size"]

small_constant = config_dict["expressivity"]["small_constant"]

# Dynamically import the model module based on the model name from the config
model_module = f"src.models.{model_name}"
create_model = importlib.import_module(model_module).create_model

print('model_name:', model_name)
# Load dataset
(train_images, train_labels), (test_images, test_labels) = dataloading.load_data(dataset_name, view_sample)

# Preprocess images
train_images, test_images = dataloading.normalize_images(train_images, test_images)



# Instantiate model
model = create_model(input_shape=input_shape, optimizer=optimizer, num_classes=num_classes, show_summary=show_summary)
# Get feature_maps before training
data_input = train_images[19]
data_input = np.expand_dims(data_input, axis=0) # Expand to have (None, 32, 32, 3) dimension as an example
bef_train_feature_maps_dict, bef_train_feature_maps_df = get_feature_maps(model, data_input, show_maps=True)
# Get expressivity_score before training
bef_train_show_exp_score_df, bef_train_exp_score_dict, bef_train_pca_var_dict, bef_train_model_exp_score_sum = calculate_exp_score(model_name=model_name,
                                                                                           state='bef_train',
                                                                                           feature_maps_dict=bef_train_feature_maps_dict,
                                                                                           constant=small_constant,
                                                                                           show_exp_score=True)

# Save expressivity scores before training
save_exp_score(bef_train_show_exp_score_df, model_name+'bef_train', dataset_name)

# Train model
if pretrained:
    model = load_model(f"results\models\complex_cnn_cifar10_20250723_165942.h5")
    make_logger.info(f"Loading pretrained model completed: {model_name}")

else:
    # Train the model
    make_logger.info(f"Training model: {model_name} on dataset: {dataset_name} started.")
    model, result = train_model(model, train_images, train_labels,
                      test_images, test_labels, epochs, batch_size)
    make_logger.info(f"Training completed for model: {model_name}")
    
    # Save the trained model
    save_model(model, model_name, dataset_name)
    # Save accuracy
    save_accuracy(result.history['val_accuracy'][-1], model_name, dataset_name)





# Get feature_maps after training
aft_train_feature_maps_dict, aft_train_feature_maps_df = get_feature_maps(model, data_input, show_maps=True)
# Get expressivity_score after training
aft_train_show_exp_score_df, aft_train_exp_score_dict, aft_train_pca_var_dict, aft_train_model_exp_score_sum = calculate_exp_score(model_name=model_name,
                                                                                           state='aft_train',
                                                                                           feature_maps_dict=aft_train_feature_maps_dict,
                                                                                           constant=small_constant,
                                                                                           show_exp_score=True)

# Save expressivity scores after training
save_exp_score(aft_train_show_exp_score_df, model_name+'aft_train', dataset_name)

if __name__ == "__main__":
    pass
