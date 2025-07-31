import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
log_path = os.path.join("logs", "app.log")
if os.path.exists(log_path):
    os.remove(log_path)

from src.utils import load_yaml
from data import dataloading
import importlib
import numpy as np
from src.engine.trainer import train_model
from src.utils.utils import save_model,save_accuracy
from tensorflow.keras.models import load_model
from src.logging.logger import get_logger
import glob
import random
import tensorflow as tf

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

optimizer = config_dict["training"]["optimizer"]
epochs = config_dict["training"]["epochs"]
batch_size = config_dict["training"]["batch_size"]


# Dynamically import the model module based on the model name from the config
model_module = f"src.models.{model_name}"
create_model = importlib.import_module(model_module).create_model

print('model_name:', model_name)
# Load dataset
(train_images, train_labels), (test_images, test_labels) = dataloading.load_data(dataset_name, view_sample)

# Preprocess images
train_images, test_images = dataloading.normalize_images(train_images, test_images)



# Instantiate model
model = create_model(database_name=dataset_name, model_name=model_name, model_input_shape=input_shape, num_classes=num_classes, optimizer=optimizer, show_summary=show_summary)

# Train model
if pretrained:
    model_files = glob.glob(f"results/models/{model_name}_*.h5")
    if not model_files:
        raise FileNotFoundError(f"No pretrained model found for {model_name}")
    model_path = model_files[0]  # You may want to sort or select the latest if multiple exist
    model = load_model(model_path)
    make_logger.info(f"Loading pretrained model completed: {model_name}")

else:
    # Train the model
    model, result = train_model(model, train_images, train_labels,
                      test_images, test_labels, epochs, batch_size)
    
    
    # Save the trained model
    save_model(model, model_name, dataset_name)
    # Save accuracy
    best_epoch = np.argmax(result.history['val_accuracy'])
    accuracy = result.history['accuracy'][best_epoch]
    val_accuracy = result.history['val_accuracy'][best_epoch]
    save_accuracy(model_name, dataset_name, accuracy, val_accuracy)



if __name__ == "__main__":
    pass
