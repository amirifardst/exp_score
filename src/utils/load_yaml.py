import yaml
from src.logging.logger import get_logger
make_logger = get_logger(__name__)

def load_config(path):
    """
    Load configuration from a YAML file.
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main(config_file_dir):
    """
    Main function to load and access configuration parameters.
    """
    config = load_config(path=config_file_dir)

    # Access model config
    model_name = config["model"]["name"]
    pretrained = config["model"]["pretrained"]
    num_classes = config["model"]["num_classes"]
    show_summary = config["model"]["show_summary"]

    # Access training config
    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    learning_rate = config["training"]["learning_rate"]

    # Access dataset config
    dataset_name = config["dataset"]["dataset_name"]
    train_download = config["dataset"]["download"]
    train_path = config["dataset"]["train_path"]
    val_path = config["dataset"]["val_path"]
    input_shape = tuple(config["dataset"]["input_shape"])
    view_sample = config["dataset"]["view_sample"]

    # Access expressivity config
    small_constant = config["expressivity"]["small_constant"]

    config_dict = {
        "model": {
            "name": model_name,
            "pretrained": pretrained,
            "num_classes": num_classes,
            "show_summary": show_summary
        },
        "training": {
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
        },
        "dataset": {
            "name": dataset_name,
            "download": train_download,
            "train_path": train_path,
            "val_path": val_path,
            "input_shape": input_shape,
            "view_sample": view_sample
        },
        "expressivity": {
            "small_constant": float(small_constant)
        }
        
    }


    make_logger.info("The following configuration parameters were loaded from yaml file:")
    make_logger.info(f"Model: {model_name}, show_summary: {show_summary}, pretrained: {pretrained}, classes: {num_classes}")
    make_logger.info(f"Training for {epochs} epochs with batch size {batch_size}")
    make_logger.info(f"Dataset_name: {dataset_name}, Downloading dataset: {train_download}, Training data path: {train_path}, Validation data path: {val_path}, view_sample: {view_sample}")
    make_logger.info(f"Input image shape: {input_shape}")
    make_logger.info(f"Expressivity small constant: {small_constant}")
    return config_dict