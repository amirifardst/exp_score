
import datetime
import os
import pandas as pd
from tensorflow.keras.models import Model
from src.logging.logger import get_logger
from IPython.display import display
from datetime import datetime

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

def get_feature_maps(model,test_input,show_maps=False):
    """
    Get feature maps from the model for a given input.
    Args :
        model: The Keras model from which to extract feature maps.
        test_input: The input data for which to obtain feature maps.

    Returns:
        dict: A dictionary containing the feature maps for each layer.
        pd.DataFrame: A DataFrame containing the feature maps for each layer.
    """
    feature_maps_dict = {}
    feature_maps_df = pd.DataFrame(columns=["Layer Name", "Feature Map Shape"])
    for index in  range(len(model.layers)):
        name = model.layers[index].name
        new_model = Model(inputs=model.inputs, outputs=model.layers[index].output)
        feature_maps = new_model.predict(test_input)
        feature_maps_dict[name] = feature_maps

        row = [name, feature_maps.shape]
        feature_maps_df = pd.concat([feature_maps_df, pd.DataFrame([row], columns=feature_maps_df.columns)], ignore_index=True)
    if show_maps:
        display(feature_maps_df)

    make_logger.info("Feature maps extracted successfully.")
    return feature_maps_dict, feature_maps_df


def save_accuracy(accuracy, model_name,database_name):
    """
    Save the model accuracy to a text file.
    Args:
        accuracy (float): The accuracy of the model.
        model_name (str): The name of the model.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df = pd.DataFrame({"Model Name": [model_name], "Accuracy": [accuracy]})
    os.makedirs("results/model_accuracies/Val_accuracy/", exist_ok=True)
    df.to_csv(f"results/model_accuracies/Val_accuracy/{model_name}_{database_name}_{timestamp}_.csv", mode="a", header=False, index=False)
    make_logger.info("Model accuracy saved successfully.")

def save_model(model, model_name, database_name):
    # Create the directory if it doesn't exist
    save_dir = "results/models"
    os.makedirs(save_dir, exist_ok=True)

    # Add time format to the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model.save(os.path.join(save_dir, f'{model_name}_{database_name}_{timestamp}.h5'))
    make_logger.info(f"Model {model_name} saved to {save_dir}")

def save_exp_score(exp_score_df, model_name, database_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = "results/exp_scores"
    os.makedirs(save_dir, exist_ok=True)
    exp_score_df.to_csv(f"{save_dir}/{model_name}_{database_name}_{timestamp}.csv", mode="a", header=True, index=True)
    make_logger.info("Expressivity scores saved successfully.")
