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
    def __init__(self, images, labels, num_classes):
        self._images = images
        self._labels = labels

    def model_fn(self):
        self.
    
    def __resnet_block_3(self, input, channels, filter_size, momentum = 0.99, is_training=true):
        out = tf.layers.conv2d(input, channels, filter_size)
        out = tf.layers.batch_normalization(out, momentum=momentum, training=is_training)
        out = tf.nn.relu(out)
        out = tf.layers.conv2d(out, channels, filter_size)
        out = tf.layers.batch_normalization(out, momentum=momentum, training=is_training)
        out = tf.add(out,input)
        out = tf.nn.relu(out)

