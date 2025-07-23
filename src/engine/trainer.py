from src.logging.logger import get_logger
make_logger = get_logger(__name__)
import tensorflow as tf

import os

def train_model(model, train_images, train_labels, test_images, test_labels, epochs,batch_size):

    history = model.fit(train_images, train_labels, epochs=epochs,
                        validation_data=(test_images, test_labels)) #batch_size=batch_size)
    make_logger.info("Model training completed.")

    return model, history