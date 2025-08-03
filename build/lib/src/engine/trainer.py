from src.logging.logger import get_logger
make_logger = get_logger(__name__)
import tensorflow as tf
import keras

import os

def train_model(model, train_images, train_labels, test_images, test_labels, epochs,batch_size):

    early_stopping = keras.callbacks.EarlyStopping(
                                                    monitor="val_accuracy",
                                                  patience=5,
                                                    verbose=1,
                                                    mode="auto",
                                                    baseline=None,
                                                    restore_best_weights=True,
                                                    start_from_epoch=0,
                                                    )
    callbacks = [early_stopping]
    # Add any callbacks you want to use, e.g., ModelCheckpoint, EarlyStopping,
    # TensorBoard, etc.
    make_logger.info("Starting model training...")
    # Train the model 
    if batch_size is not None:
        history = model.fit(train_images, train_labels, epochs=epochs,
                            validation_data=(test_images, test_labels), batch_size=batch_size, callbacks=callbacks)
    else:
        make_logger.warning("Batch size is None, using default batch size.")
        # If batch_size is None, use the default behavior of Keras

    make_logger.info("Model training completed.")

    return model, history