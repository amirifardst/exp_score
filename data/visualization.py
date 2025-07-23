import matplotlib.pyplot as plt
from src.logging.logger import get_logger
import math

make_logger = get_logger()

def plot_sample_images(images, labels, class_names):

    """
    Plot a grid of sample images from the dataset.

    Args:
        images (ndarray): The images.
        labels (ndarray): The labels corresponding to the images.
        class_names (list): List of class names for labeling the images.
    """
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
        plt.xlabel(class_names[labels[i][0]])
    plt.show()
    make_logger.info("Sample images plotted successfully")


def plot_filter_maps(feature_maps_dict, layer_name):

    num_images = feature_maps_dict[layer_name].shape[-1]
    # Calculate number of rows needed to fit all images
    cols = 8
    rows = math.ceil(num_images / cols)
    
    ix = 1
    feature_maps = feature_maps_dict[layer_name]
    plt.figure(figsize=(15, 15))
    
    for _ in range(rows):
        for _ in range(cols):
            if ix > num_images:
                break  # Stop if we reached the number of images
    
            # Specify subplot and turn off axis
            ax = plt.subplot(rows, cols, ix)
            ax.set_xticks([])
            ax.set_yticks([])
    
            # Plot filter channel in grayscale
            plt.imshow(feature_maps[0, :, :, ix - 1], cmap='gray')
    
            ix += 1
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Show the figure
    plt.show()
    
