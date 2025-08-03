
import os
from src.logging.logger import get_logger
make_logger = get_logger()
import tensorflow as tf
import keras


def bn_relu(x):
    """
Performs BN and ReLU activation sequentially.
"""
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    return x


initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)
def conv_block(x, n_filters1, n_filters2, n_blocks, layer, initializer=initializer):
    """
Basic building blocks for the ResNet-50 architecture.

Parameters
----------
x : tensor
  input tensor to conv block.
n_filters1 : int
  channel size during the first convolution part.
n_filters2 : int
  output channel size.
n_blocks : int
  defines how many times we should apply the "triple convolution".
layer : int
  specifies which layer we are on according to the original article.
initializer: keras initializer object
  specifies the kernel initialization method.

Returns
----------
x : tensor
  tensor after the convolution process
"""

    # store the original input
    # store the original input
    identity = x
    for n in range(n_blocks):
        # after layer 2, at the beginning of each layer, we want to downsample the input with 1x1 convolutions,
        # and strides of 2
        start_strides = 2 if (layer > 2 and n == 0) else 1
        x = keras.layers.Conv2D(filters=n_filters1, kernel_size=(1, 1), strides=start_strides, padding='same',
                                kernel_initializer=initializer)(x)
        x = bn_relu(x)
        # in every other case, strides are 1
        x = keras.layers.Conv2D(filters=n_filters1, kernel_size=(3, 3), padding='same', kernel_initializer=initializer)(
            x)
        x = bn_relu(x)
        x = keras.layers.Conv2D(filters=n_filters2, kernel_size=(1, 1), padding='same', kernel_initializer=initializer)(
            x)
        x = keras.layers.BatchNormalization()(x)

        # if we are at the beginning of the block, the channel dimension of the original input should match the dim.
        # of our current x
        if n == 0:
            identity = keras.layers.Conv2D(filters=n_filters2, kernel_size=(1, 1), strides=start_strides,
                                           padding='same', kernel_initializer=initializer)(identity)
            identity = keras.layers.BatchNormalization()(identity)

        # add x and identity (skip connection) and apply ReLU
        x = keras.layers.Add()([x, identity])
        x = keras.layers.Activation('relu')(x)

    return x


def create_model(database_name, model_name, model_input_shape, num_classes, optimizer, show_summary=True):
    """
    Create model_5 for image classification.
    Model_5 is a Resnet CNN model.

    Args:
        model_input_shape (tuple): The shape of the input images.
        num_classes (int): The number of output classes.

    Returns:
        model: A compiled CNN model.
    """
 
  
    # specify input shapes
    inputs = keras.layers.Input(model_input_shape)

    # conv1_x
    x = keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding='same', kernel_initializer=initializer)(
        inputs)
    x = bn_relu(x)
    x = keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)(x)

    # conv2_x
    x = conv_block(x, 64, 256, n_blocks=3, layer=2)
    # conv3_x
    x = conv_block(x, 128, 512, n_blocks=4, layer=3)
    # conv4_x
    x = conv_block(x, 256, 1024, n_blocks=6, layer=4)
    # conv5_x
    x = conv_block(x, 512, 2048, n_blocks=3, layer=5)

    # this will flatten x
    x = keras.layers.GlobalAveragePooling2D()(x)

    # classification layer
    x = keras.layers.Dense(units=num_classes, kernel_initializer=initializer)(x)

    model = keras.Model(inputs=inputs, outputs=x)

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