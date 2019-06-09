import tensorflow as tf
import numpy as np
from PIL import Image

from tensorflow.examples.tutorials.mnist import input_data

def parse_image(filename, label, image_size):

    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpge(image_string, channels=3)
    image_tf = tf.image.convert_image_dtype(image_decoded, tf.float32)
    image_resized = tf.image.resize_image(image_tf, [image_size,image_size])
    
    return image_resized


def input_fn(filenames, labels, params, is_training, num_samples):
    """ Use tf.data.Dataset
    Given any filenames and labels, create the iterable tf dataset """

    parse_fn = lambda f, l: parse_image(f, l, params.image_size)

    if is_training:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
            .shuffle(num_samples)  # whole dataset into the buffer ensures good shuffling
            .map(parse_fn, num_parallel_calls=params.num_parallel_calls)
            .batch(params.batch_size)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )
    else:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
            .map(parse_fn)
            .batch(params.batch_size)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )
    iterator = dataset.make_initializable_iterator()
    images, labels = iterator.get_next()
    iterator_init_op = iterator.initializer()

    inputs = {'images':images, 'labels':labels, 'iterator_init_op':iterator_init_op}

    return inputs
    