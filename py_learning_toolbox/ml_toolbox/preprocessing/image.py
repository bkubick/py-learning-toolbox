from __future__ import annotations

import typing

import tensorflow as tf

if typing.TYPE_CHECKING:
    ImagePreprocessingFunction = typing.Callable[[tf.Tensor, tf.Tensor], typing.Tuple[tf.Tensor, tf.Tensor]]


__all__ = ['generate_preprocess_image_function', 'load_and_resize_image']


def load_and_resize_image(filename: str, image_size: int = 224, scale: bool = True) -> tf.Tensor:
    """ Loads and resizes an image for the model.
    
        Args:
            filename (str): The filename of the image.
            image_size (int): The size of the image.
            scale (bool): Whether to scale the image between 0 and 1 or not.

        Returns:
            (tf.Tensor) The prepped image.
    """
    # Read in the image
    image = tf.io.read_file(filename)
    image = tf.io.decode_image(image, channels=3)
    image = tf.image.resize(image, [image_size, image_size])

    # Scale Image to get all between 0 & 1 (not always required)
    if scale:
        image = image / 255.

    return image


def generate_preprocess_image_function(img_shape: int = 224, scale: bool = False) -> ImagePreprocessingFunction:
    """ Generates the preprocessing function for the given image shape.

        - Convert image dtype from `uint8` to `float32`
        - Scale image values between 0 and 1
        - Reshapes image to (img_shape, img_shape, color_channels)

        Args:
            img_shape (int): The size the img should be shaped to
            scale (bool): Whether to scale the image values between 0 and 1
        
        Returns:
            (ImagePreprocessingFunction) The preprocessing function.
    """
    def preprocess_image(image: tf.Tensor, label: tf.Tensor) -> typing.Tuple[tf.Tensor, tf.Tensor]:
        """ Preprocess the given image performing the following:

            - Convert image dtype from `uint8` to `float32`
            - Scale image values between 0 and 1
            - Reshapes image to (img_shape, img_shape, color_channels)

            Args:
                image (tf.Tensor): Image to preprocess
                label (tf.Tensor): Label for the corresponding image
        """
        image = tf.image.resize(image, [img_shape, img_shape])

        if scale:
            image /= 255.

        return tf.cast(image, tf.float32), label

    return preprocess_image
