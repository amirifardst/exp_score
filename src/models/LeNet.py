import tensorflow as tf
from tensorflow.keras import layers, models
from src import models
from src.logging.logger import get_logger
make_logger = get_logger()

 # Set initial seed for reproducibility
def conv2d_with_init(filters, kernel_size, **kwargs):
    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)
    return tf.keras.layers.Conv2D(filters, kernel_size, kernel_initializer=initializer, **kwargs)

def dense_with_init(units, **kwargs):
    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)
    return tf.keras.layers.Dense(units, kernel_initializer=initializer, **kwargs)

def create_model(input_shape, num_classes, optimizer, show_summary=False):
    """
    Create a simple CNN model for image classification.

    Args:
        input_shape (tuple): The shape of the input images.
        num_classes (int): The number of output classes.

    Returns:
        model: A compiled CNN model.
    """
   

    # Define the model architecture

    model = tf.keras.Sequential([
        conv2d_with_init(6, (5, 5), padding='valid', activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2), strides=(2, 2)),
        conv2d_with_init(16, (5, 5), padding='valid', activation='relu'),
        layers.MaxPooling2D((2, 2), strides=(2, 2)),
        layers.Flatten(),
        dense_with_init(120, activation='relu'),
        dense_with_init(84, activation='relu'),
        dense_with_init(num_classes)
    ])

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    if show_summary:
        model.summary()

    make_logger.info(f"Model with input shape {input_shape} and output shape {num_classes} created successfully")

    return model


