import tensorflow as tf

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
    def __init__(self, name):
        # Create a useful list of private variables for training the network
        self._name = name
        self._flops = 0
        self._weights = 0
        self._learning_rate = 0.0001

    def __build_model(self, inputs):
        channels = [64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 512, 512, 512]
        kernels = [7, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
        strides = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        assert len(kernels) == len(channels)

        # DESIGN OF THE ARCHITECTURE

        # The first conv block
        with tf.variable_scope("conv1"):
            out = tf.layers.conv2d(inputs, channels[0], strides[0])
            out = tf.layers.batch_normalization(out)
            out = tf.nn.relu(out)
            out = tf.nn.max_pool(out, 3, strides[0], padding="valid")

        out = self.__identity_block(out, channels[1], kernels[1], 1, name="identity1")
        out = self.__identity_block(out, channels[2], kernels[2], 1, name="identity2")
        out = self.__identity_block(out, channels[3], kernels[3], 1, name="identity3")

        out = self.__conv_block(out, channels[4], kernels[4], 1, name="conv_res_block1")
        out = self.__identity_block(out, channels[5], kernels[5], 1, name="identity4")
        out = self.__identity_block(out, channels[6], kernels[6], 1, name="identity5")
        out = self.__identity_block(out, channels[7], kernels[7], 1, name="identity6")

        out = self.__conv_block(out, channels[8], kernels[8], 1, name="conv_res_block2")
        out = self.__identity_block(out, channels[9], kernels[9], 1, name="identity7")
        out = self.__identity_block(out, channels[10], kernels[10], 1, name="identity8")
        out = self.__identity_block(out, channels[11], kernels[11], 1, name="identity9")
        out = self.__identity_block(out, channels[12], kernels[12], 1, name="identity10")
        out = self.__identity_block(out, channels[13], kernels[13], 1, name="identity11")

        out = tf.layers.average_pooling2d(out, pool_size=7, strides=1, padding="valid", name="pool")
        out = tf.reshape(-1,512)
        logits = tf.layers.dense(out, 10)

        return logits

    def model_fn(self, inputs, labels, is_training):
        with tf.variable_scope("model"):
            logits = self.__build_model(inputs)
            predictions = tf.argmax(logits, 1)
        
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))
        optimizer = tf.train.AdamOptimizer(self._learning_rate)
        train_op = optimizer.minimize(loss, global_step=self._global_step)

        # -----------------------------------------------------------
        # METRICS AND SUMMARIES
        # Metrics for evaluation using tf.metrics (average over whole dataset)
        with tf.variable_scope("metrics"):
            metrics = {
                'accuracy': tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, 1)),
                'loss': tf.metrics.mean(loss)
            }

        # Group the update ops for the tf.metrics
        update_metrics_op = tf.group(*[op for _, op in metrics.values()])

        # Get the op to reset the local variables used in tf.metrics
        metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
        metrics_init_op = tf.variables_initializer(metric_variables)

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        
        model_spec = inputs
        model_spec = inputs
        model_spec['variable_init_op'] = tf.global_variables_initializer()
        model_spec["predictions"] = predictions
        model_spec['loss'] = loss
        model_spec['accuracy'] = accuracy
        model_spec['metrics_init_op'] = metrics_init_op
        model_spec['metrics'] = metrics
        model_spec['update_metrics'] = update_metrics_op
        model_spec['summary_op'] = tf.summary.merge_all()

        if is_training:
            model_spec['train_op'] = train_op

        return model_spec

    
    def __identity_block(self, input, channels, filter_size, strides, name, momentum = 0.99, is_training=true):
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

    def __conv_block(self, input, channels, filter_depths, strides, name, momentum = 0.99, is_training=true):
        with tf.variable_scope(name):
            shortcut = tf.layers.conv2d(input, channels, filter_size, strides, padding="same")
            shortcut = tf.layers.batch_normalization(shortcut, momentum=momentum, training=is_training)

            out = tf.layers.conv2d(input, channels, filter_size, strides=2, padding="same")
            out = tf.layers.batch_normalization(out, momentum=momentum, training=is_training)
            out = tf.nn.relu(out)
            out = tf.layers.conv2d(out, channels, filter_size, strides=1, padding="same")
            out = tf.layers.batch_normalization(out, momentum=momentum, training=is_training)
            out = tf.add(out,shortcut)
            out = tf.nn.relu(out)

            return out
