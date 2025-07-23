import tensorflow as tf
from src.logging.logger import get_logger
make_logger = get_logger()


def create_model(input_shape, num_classes, optimizer, show_summary=False):
    """
    Create a simple CNN model for image classification.

    Args:
        input_shape (tuple): The shape of the input images.
        num_classes (int): The number of output classes.

    Returns:
        model: A compiled CNN model.
    """
        
    model = tf.keras.Sequential([
                                tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='same',input_shape=input_shape),
                                tf.keras.layers.BatchNormalization(),
                                tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='same'),
                                tf.keras.layers.BatchNormalization(),
                                tf.keras.layers.MaxPooling2D((2,2)),
                                tf.keras.layers.Dropout(0.2),
                                tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same'),
                                tf.keras.layers.BatchNormalization(),
                                tf.keras.layers.MaxPooling2D((2,2)),
                                tf.keras.layers.Dropout(0.3),
                                tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same'),
                                tf.keras.layers.BatchNormalization(),
                                tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same'),
                                tf.keras.layers.BatchNormalization(),
                                tf.keras.layers.MaxPooling2D((2,2)),
                                tf.keras.layers.Dropout(0.4),
                                tf.keras.layers.Flatten(),
                                tf.keras.layers.Dense(128,activation='relu'),
                                tf.keras.layers.BatchNormalization(),
                                tf.keras.layers.Dropout(0.4),
                                tf.keras.layers.Dense(num_classes)])

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    if show_summary:
        model.summary()

    make_logger.info("Model with input shape {input_shape} and output shape {num_classes} created successfully".format(input_shape, num_classes))

    return model


