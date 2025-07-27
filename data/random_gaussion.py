import tensorflow as tf
from src.logging.logger import get_logger
make_logger = get_logger(__name__)

seed = 42
tf.random.set_seed(seed)

def random_gaussian_input(batch_size, input_shape):
    """
    Generate random Gaussian input with the specified shape.
    
    Args:
        input_shape (tuple): Shape of the input (b, w, h, c).
    """
    return tf.random.normal((batch_size, *input_shape), dtype=tf.float32)

# Example usage:
# input_shape = (32, 32, 3)  # Example input shape
# batch_size = 64 
# random_input = random_gaussian_input(batch_size, input_shape)
# print(random_input.shape)  # Should print (64, 32, 32, 3)