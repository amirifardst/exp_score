import tensorflow as tf
import os
from src.logging.logger import get_logger
make_logger = get_logger()

 # Set initial seed for reproducibility
def conv2d_with_init(filters, kernel_size, **kwargs):
    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)
    return tf.keras.layers.Conv2D(filters, kernel_size, kernel_initializer=initializer, **kwargs)

def dense_with_init(units, **kwargs):
    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)
    return tf.keras.layers.Dense(units, kernel_initializer=initializer, **kwargs)

def create_model(database_name, model_name, model_input_shape, num_classes, optimizer, show_summary=False):
    """
    Create model_4 for image classification.
    Model_4 is a complex CNN model.

    Args:
        input_shape (tuple): The shape of the input images.
        num_classes (int): The number of output classes.

    Returns:
        model: A compiled CNN model.
    """
   

    # Define the model architecture
    model = tf.keras.Sequential([
                                conv2d_with_init(32,(3,3),activation='relu',padding='same',input_shape=model_input_shape),
                                tf.keras.layers.BatchNormalization(),
                                conv2d_with_init(32,(3,3),activation='relu',padding='same'),
                                tf.keras.layers.BatchNormalization(),
                                tf.keras.layers.MaxPooling2D((2,2)),
                                tf.keras.layers.Dropout(0.2),
                                conv2d_with_init(64,(3,3),activation='relu',padding='same'),
                                tf.keras.layers.BatchNormalization(),
                                tf.keras.layers.MaxPooling2D((2,2)),
                                tf.keras.layers.Dropout(0.3),
                                conv2d_with_init(128,(3,3),activation='relu',padding='same'),
                                tf.keras.layers.BatchNormalization(),
                                conv2d_with_init(128,(3,3),activation='relu',padding='same'),
                                tf.keras.layers.BatchNormalization(),
                                tf.keras.layers.MaxPooling2D((2,2)),
                                tf.keras.layers.Dropout(0.4),
                                tf.keras.layers.Flatten(),
                                dense_with_init(128, activation='relu'),
                                tf.keras.layers.BatchNormalization(),
                                tf.keras.layers.Dropout(0.4),
                                dense_with_init(num_classes)])

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    if show_summary:
            save_dir = f"results/{database_name}/{model_name}"
            os.makedirs(save_dir, exist_ok=True)
            tf.keras.utils.plot_model(model, to_file=f"{save_dir}/{model_name}.png", show_shapes=True)
            print('In the following you can see the model summary:')
            model.summary()

    make_logger.info(f"Model with input shape {model_input_shape} and output shape {num_classes} created successfully")

    return model


