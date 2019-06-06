import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

class ResNet(object):
    def __init__(self, images, labels, num_classes, global_step, name, is_training):
        # Create a useful list of private variables for training the network 
        self._images = images
        self._labels = labels
        self._global_step = global_step
        self._name = name
        self._is_training = is_training
        self._flops = 0
        self._weights = 0

    def model_fn(self):
        filters = [64, 64, 128, 256, 512]
        kernels = [7, 3, 3, 3, 3]
        strides = [2, 1, 1, 1, 1]

        # The first conv block
        with tf.variable_scope("conv1"):
            out = tf.layers.conv2d(self._images, filters[0], strides[0])
            out = tf.layers.batch_normalization(out)
            out = tf.nn.relu(out)
            out = tf.nn.max_pool(out, 3, strides[0], padding="valid")
        

    
    def __identity_block_3(self, input, channels, filter_size, strides, name="res_identity_block", momentum = 0.99, is_training=true):
        with tf.variable_scope(name):
            shortcut = input
            out = tf.layers.conv2d(input, channels, filter_size, strides=1, padding="same")
            out = tf.layers.batch_normalization(out, momentum=momentum, training=is_training)
            out = tf.nn.relu(out)
            out = tf.layers.conv2d(out, channels, filter_size, strides=1, padding="same")
            out = tf.layers.batch_normalization(out, momentum=momentum, training=is_training)
            out = tf.add(out,shortcut)
            out = tf.nn.relu(out)

            return out

    def __conv_block(self, input, channels, filter_size, strides, name="res_conv_block", momentum = 0.99, is_training=true):
        with tf.variable_scope(name):
            shortcut = tf.layers.conv2d(input, channels, filter_size, strides, padding="same")
            shortcut = tf.layers.batch_normalization(shortcut, momentum=momentum, training=is_training)

            out = tf.layers.conv2d(input, channels, filter_size, strides=2, padding="same")
            out = tf.layers.batch_normalization(out, momentum=momentum, training=is_training)
            out = tf.nn.relu(out)
            out = tf.layers.conv2d(out, channels, filter_size, strides=1, padding="same")
            out = tf.layers.batch_normalization(out, momentum=momentum, training=is_training)
            out = tf.nn.relu(out)
            out = tf.layers.conv2d(out, channels, filter_size, strides=1, padding="same")
            out = tf.layers.batch_normalization(out, momentum=momentum, training=is_training)
            out = tf.add(out,shortcut)
            out = tf.nn.relu(out)

            return out
