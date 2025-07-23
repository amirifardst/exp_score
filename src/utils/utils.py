
import pandas as pd
from tensorflow.keras.models import Model
from src.logging.logger import get_logger
from IPython.display import display
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