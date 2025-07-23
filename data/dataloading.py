from src.logging.logger import get_logger
from tensorflow.keras import datasets
from data.preprocessing import normalize_images
from data.visualization import plot_sample_images
#Logger setup
make_logger = get_logger()


def load_data(dataset_name,view_sample=False):
    """
    Load the specified dataset.

    Args:
        dataset_name (str): The name of the dataset to load.

    Returns:
        (train_images, train_labels), (test_images, test_labels): The training and test data.
    """
    if 'cifar10' in dataset_name: 
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
        (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
        make_logger.info(f"{dataset_name} dataset loaded with {len(train_images)} training and {len(test_images)} test samples.")
        make_logger.info(f"Training data shape: {train_images.shape}, Labels shape: {train_labels.shape}")
        # train_images, test_images = normalize_images(train_images, test_images)
        if view_sample:
            plot_sample_images(train_images, train_labels, class_names)
        return (train_images, train_labels), (test_images, test_labels)
    else:
        make_logger.error(f"Unknown dataset: {dataset_name}")



