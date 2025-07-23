
import numpy as np
from src.logging.logger import get_logger
make_logger = get_logger()

def normalize_images(train_data, test_data):
    """
    Normalize the input data to a range of [0, 1].
    
    args:
    train_data (ndarray): The training data to be normalized.
    test_data (ndarray): The test data to be normalized.

    returns:
    (ndarray, ndarray): The normalized training and test data.
    """
    normalized_train_data, normalized_test_data = train_data / 255.0, test_data / 255.0
    assert np.max(normalized_train_data) <= 1.0 and np.min(normalized_train_data) >= 0.0, "Training data normalization failed"
    assert np.max(normalized_test_data) <= 1.0 and np.min(normalized_test_data) >= 0.0, "Test data normalization failed"

    make_logger.info(f"Data was normalized successfully: train_data shape {normalized_train_data.shape}, test_data shape {normalized_test_data.shape}")
    return normalized_train_data, normalized_test_data