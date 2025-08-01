
import datetime
import os
import pandas as pd
from src.logging.logger import get_logger
from IPython.display import display
from datetime import datetime
from tensorflow.keras.models import Model
import numpy as np


make_logger = get_logger(__name__)


def get_filters(model,show_filters=False):

    """
    Take the model and give filters for layers
    Args: 
        model

    Returns:
        dict : A dictionary which includes filters for different layers
        pd.DataFrame : A DataFrame which includes filters for different layers

    """
    filters_dict = {}
    filters_df = pd.DataFrame(columns=["Layer Name", "Filter Weights Shape", "Biases Shape"])
    for index, layer in enumerate(model.layers):
        try :
            filter_wights, biases = layer.get_weights()[0], layer.get_weights()[1]
            # print(f"Layer Number: {index+1}")
            # print(f"Layer_Name: {layer.name}")
            # print(f"filter_wights: {filter_wights.shape}")
            # print(f"biases: {biases.shape}")
            filters_dict[layer.name] = [filter_wights,biases]
            filters_df = filters_df.append({"Layer Name": layer.name,
                                              "Filter Weights Shape": filter_wights.shape,
                                              "Biases Shape": biases.shape}, ignore_index=True)
        except:
            # print(f"Layer Number: {index+1}")
            # print(f"Layer_Name: {layer.name}")
            # print("Had No weights")
            filters_df = filters_df.append({"Layer Name": layer.name,
                                              "Filter Weights Shape": None,
                                              "Biases Shape": None}, ignore_index=True)
            pass

    if show_filters:
        print(filters_df)
    return filters_dict, filters_df

def get_all_feature_maps(model, input_tensor, show_maps=True, remove_unnecessary=True):
    """
    Extract feature maps from all layers in a Keras model using one forward pass.
   
    Args:
        model: tf.keras.Model
        input_tensor: input data (e.g., Gaussian or real image batch)

    Returns:
        feature_dict: {layer_name: output_array}
    """
    # Filter out InputLayer (no meaningful output)
    layers_to_use = [layer for layer in model.layers if hasattr(layer, 'output')]

    if remove_unnecessary:
        # Remove unnecessary layers (e.g., Dropout, Flatten)
        layers_to_use = [layer for layer in layers_to_use if not layer.__class__.__name__ in ['Dropout',
                                                                                              'Flatten','BatchNormalization',
                                                                                              "MaxPooling2D"]]

    
    # Build a model with multiple outputs
    feature_model = Model(inputs=model.inputs,
                          outputs=[layer.output for layer in layers_to_use])
   
    
    # Run one forward pass
    outputs = feature_model.predict(input_tensor, verbose=0)
    # Map layer names to outputs
    feature_dict = {layer.name: output for layer, output in zip(layers_to_use, outputs)}
    
    if show_maps:
        print('feature maps:')
        for layer_name, feature_map in feature_dict.items():
            print(f"Layer: {layer_name}, Feature Map Shape: {feature_map.shape}")
    return feature_dict



def save_accuracy(model_name, database_name, accuracy,val_accuracy,):
    """
    Save the model accuracy to a text file.
    Args:
        accuracy (float): The accuracy of the model.
        model_name (str): The name of the model.
    """
    df = pd.DataFrame({"Model Name": [model_name], "Accuracy": [accuracy], "Validation Accuracy": [val_accuracy]})
    save_dir = f"results/{database_name}/{model_name}"
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(f"{save_dir}/{model_name}_accuracy.csv", mode="a", header=True, index=True)
    make_logger.info(f"Accuracy of model {model_name} saved successfully.")

def save_model(model, model_name, database_name):
    # Create the directory if it doesn't exist
    save_dir = f"results/{database_name}/{model_name}"
    os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, f'{model_name}.h5'))
    make_logger.info(f"Model {model_name} saved to {save_dir}")

def save_exp_score(exp_score_df, model_name, database_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"results/{database_name}/{model_name}"
    os.makedirs(save_dir, exist_ok=True)
    exp_score_df.to_csv(f"{save_dir}/{model_name}_{timestamp}_score.csv", mode="a", header=True, index=True)
    make_logger.info(f"Expressivity scores of model {model_name} saved successfully.")
