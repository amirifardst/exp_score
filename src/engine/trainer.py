from src.logging.logger import get_logger
make_logger = get_logger(__name__)
import tensorflow as tf
import os
from datetime import datetime

def train_model(model, train_images, train_labels, test_images, test_labels,
                database_name, epochs,batch_size):

    history = model.fit(train_images, train_labels, epochs=epochs,
                        validation_data=(test_images, test_labels)) #batch_size=batch_size)
    make_logger.info("Model training completed.")

    # Create the directory if it doesn't exist
    save_dir = "saved models"
    os.makedirs(save_dir, exist_ok=True)

    # Add time format to the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model.save(os.path.join(save_dir, f'simple_cnn_{database_name}_{timestamp}.h5'))
    make_logger.info(f"Model saved to {save_dir}")

    return model, history