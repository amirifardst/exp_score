
import pandas as pd
import numpy as np
from src.logging.logger import get_logger
make_logger = get_logger(__name__)

def get_statistics(exp_score_dict, show_exp_score=True):
    """
    Function to calculate statistics from the expressivity score dictionary.
    :param exp_score_dict: Dictionary containing expressivity scores for each layer.
    :param show_exp_score: Boolean to control whether to print the DataFrame.
    :return: DataFrame with statistics.
    """
    
    make_logger.info('Calculating statistics from expressivity score dictionary...')
    
    exp_score_df = pd.DataFrame({
        'Layer Name': list(exp_score_dict.keys()),
        'Spatial Size': [int(v['spatial_size']) for v in exp_score_dict.values()],
        'Number of Channels': [int(v['num_channels']) for v in exp_score_dict.values()],
        "Log (c)": [float(v['log_c']) for v in exp_score_dict.values()],
        'Expressivity Score': [float(v['expressivity_score']) for v in exp_score_dict.values()],
        'Normalized Expressivity Score': [float(v['normalized_expressivity_score']) for v in exp_score_dict.values()],
    })

    
    # Add min value
    min_row = pd.DataFrame({
        'Layer Name': ['min'],
        'Spatial Size': np.min([int(v['spatial_size']) for v in exp_score_dict.values()]),
        'Number of Channels': np.min([int(v['num_channels']) for v in exp_score_dict.values()]),
        'Log (c)': np.min([float(v['log_c']) for v in exp_score_dict.values()]),
        'Expressivity Score': np.min([float(v['expressivity_score']) for v in exp_score_dict.values()]),
        "Normalized Expressivity Score": np.min([float(v['normalized_expressivity_score']) for v in exp_score_dict.values()]),
    })

    #add max value
    max_row = pd.DataFrame({
        'Layer Name': ['max'],
        'Spatial Size': np.max([int(v['spatial_size']) for v in exp_score_dict.values()]),
        'Number of Channels': np.max([int(v['num_channels']) for v in exp_score_dict.values()]),
        'Log (c)': np.max([float(v['log_c']) for v in exp_score_dict.values()]),
        'Expressivity Score': np.max([float(v['expressivity_score']) for v in exp_score_dict.values()]),
        "Normalized Expressivity Score": np.max([float(v['normalized_expressivity_score']) for v in exp_score_dict.values()]),
    })
    # Add mean
    avg_row = pd.DataFrame({
        'Layer Name': ['mean'],
        'Spatial Size': np.mean([int(v['spatial_size']) for v in exp_score_dict.values()]),
        'Number of Channels': np.mean([int(v['num_channels']) for v in exp_score_dict.values()]),
        'Log (c)': np.mean([float(v['log_c']) for v in exp_score_dict.values()]),
        'Expressivity Score': np.mean([float(v['expressivity_score']) for v in exp_score_dict.values()]),
        "Normalized Expressivity Score": np.mean([float(v['normalized_expressivity_score']) for v in exp_score_dict.values()]),
    })

    # add median
    median_row = pd.DataFrame({
        'Layer Name': ['median'],
        'Spatial Size': np.median([int(v['spatial_size']) for v in exp_score_dict.values()]),
        'Number of Channels': np.median([int(v['num_channels']) for v in exp_score_dict.values()]),
        'Log (c)': np.median([float(v['log_c']) for v in exp_score_dict.values()]),
        'Expressivity Score': np.median([float(v['expressivity_score']) for v in exp_score_dict.values()]),
        "Normalized Expressivity Score": np.median([float(v['normalized_expressivity_score']) for v in exp_score_dict.values()]),
    })
    # Add std
    std_row = pd.DataFrame({
        'Layer Name': ['std'],
        'Spatial Size': np.std([int(v['spatial_size']) for v in exp_score_dict.values()]),
        'Number of Channels': np.std([int(v['num_channels']) for v in exp_score_dict.values()]),
        'Log (c)': np.std([float(v['log_c']) for v in exp_score_dict.values()]),
        'Expressivity Score': np.std([float(v['expressivity_score']) for v in exp_score_dict.values()]),
        "Normalized Expressivity Score": np.std([float(v['normalized_expressivity_score']) for v in exp_score_dict.values()]),
    })
    exp_score_df = pd.concat([exp_score_df, min_row, max_row, avg_row, median_row, std_row], ignore_index=True)

    if show_exp_score:
        print(exp_score_df)

    return exp_score_df