from tensorflow.keras import layers, models
import tensorflow as tf
from src.logging.logger import get_logger
make_logger = get_logger()


def create_model(input_shape, num_classes,optimizer, show_summary=False):
    """
    Create a simple CNN model for image classification.

    Args:
        input_shape (tuple): The shape of the input images.
        num_classes (int): The number of output classes.

    Returns:
        model: A compiled CNN model.
    """
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes))  # Use num_classes parameter

    model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
    
    if show_summary:
        model.summary()

    make_logger.info("Model with input shape {} and output shape {} created successfully".format(input_shape, num_classes))

    return model

