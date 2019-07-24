import tensorflow as tf


class Generator(object):
    """the definition of Generator"""

    def __init__(self, FLAGS):
        self.channel = FLAGS.channel
        self.n_filter = 64
        self.inc_filter = 32
        self.num_repeat_RRDB = FLAGS.num_repeat_RRDB
        self.residual_scaling = FLAGS.residual_scaling
        self.init_kernel = tf.initializers.he_normal(seed=FLAGS.initialization_random_seed)

    def _conv_RRDB(self, x, out_channel, num=None, activate=True):
        with tf.variable_scope('block_{0}'.format(num)):
            x = tf.layers.conv2d(x, out_channel, 3, 1, padding='same', kernel_initializer=self.init_kernel, name='conv')
            if activate:
                x = tf.nn.leaky_relu(x, alpha=0.2, name='leakyReLU')

        return x

    def _denseBlock(self, x, num=None):
        with tf.variable_scope('DenseBlock_sub{0}'.format(num)):
            x1 = self._conv_RRDB(x, self.inc_filter, 0)
            x2 = self._conv_RRDB(tf.concat([x, x1], axis=3), self.inc_filter, 1)
            x3 = self._conv_RRDB(tf.concat([x, x1, x2], axis=3), self.inc_filter, 2)
            x4 = self._conv_RRDB(tf.concat([x, x1, x2, x3], axis=3), self.inc_filter, 3)
            x5 = self._conv_RRDB(tf.concat([x, x1, x2, x3, x4], axis=3), self.n_filter, 4, activate=False)

        return x5 * self.residual_scaling

    def _RRDB(self, x, num=None):
        """Residual in Residual Dense Block"""
        with tf.variable_scope('RRDB_sub{0}'.format(num)):
            x_branch = tf.identity(x)

            x_branch += self._denseBlock(x_branch, 0)
            x_branch += self._denseBlock(x_branch, 1)
            x_branch += self._denseBlock(x_branch, 2)

        return x + x_branch * self.residual_scaling

    def _upsampling_layer(self, x, num=None):
        x = tf.layers.conv2d_transpose(x, self.n_filter, 3, 2, padding='same', name='upsample_{0}'.format(num))
        x = tf.nn.leaky_relu(x, alpha=0.2, name='leakyReLU')

        return x

    def build(self, x):
        with tf.variable_scope('first_conv'):
            x = tf.layers.conv2d(x, self.n_filter, 3, 1, padding='same', kernel_initializer=self.init_kernel,
                                 name='conv')

        with tf.variable_scope('RRDB'):
            x_branch = tf.identity(x)

            for i in range(self.num_repeat_RRDB):
                x_branch = self._RRDB(x_branch, i)

            x_branch = tf.layers.conv2d(x_branch, self.n_filter, 3, 1, padding='same',
                                        kernel_initializer=self.init_kernel, name='trunk_conv')

        x += x_branch

        with tf.variable_scope('Upsampling'):
            x = self._upsampling_layer(x, 1)
            x = self._upsampling_layer(x, 2)

        with tf.variable_scope('last_conv'):
            x = tf.layers.conv2d(x, self.n_filter, 3, 1, padding='same', kernel_initializer=self.init_kernel,
                                 name='conv_1')
            x = tf.nn.leaky_relu(x, alpha=0.2, name='leakyReLU')
            x = tf.layers.conv2d(x, self.channel, 3, 1, padding='same', kernel_initializer=self.init_kernel,
                                 name='conv_2')

        return x


class Discriminator(object):
    """the definition of Discriminator"""

    def __init__(self, FLAGS):
        self.channel = FLAGS.channel
        self.n_filter = 64
        self.inc_filter = 32
        self.init_kernel = tf.initializers.he_normal(seed=FLAGS.initialization_random_seed)

    def _conv_block(self, x, out_channel, num=None):
        with tf.variable_scope('block_{0}'.format(num)):
            x = tf.layers.conv2d(x, out_channel, 3, 1, padding='same', use_bias=False,
                                 kernel_initializer=self.init_kernel, name='conv_1')
            x = tf.layers.BatchNormalization(name='batch_norm_1')(x)
            x = tf.nn.leaky_relu(x, alpha=0.2, name='leakyReLU_1')

            x = tf.layers.conv2d(x, out_channel, 4, 2, padding='same', use_bias=False,
                                 kernel_initializer=self.init_kernel, name='conv_2')
            x = tf.layers.BatchNormalization(name='batch_norm_2')(x)
            x = tf.nn.leaky_relu(x, alpha=0.2, name='leakyReLU_2')

            return x

    def build(self, x):
        with tf.variable_scope('first_conv'):
            x = tf.layers.conv2d(x, self.n_filter, 3, 1, padding='same', use_bias=False,
                                 kernel_initializer=self.init_kernel, name='conv_1')
            x = tf.nn.leaky_relu(x, alpha=0.2, name='leakyReLU_1')
            x = tf.layers.conv2d(x, self.n_filter, 4, 2, padding='same', use_bias=False,
                                 kernel_initializer=self.init_kernel, name='conv_2')
            x = tf.layers.BatchNormalization(name='batch_norm_1')(x)
            x = tf.nn.leaky_relu(x, alpha=0.2, name='leakyReLU_2')

        with tf.variable_scope('conv_block'):
            x = self._conv_block(x, self.n_filter * 2, 0)
            x = self._conv_block(x, self.n_filter * 4, 1)
            x = self._conv_block(x, self.n_filter * 8, 2)
            x = self._conv_block(x, self.n_filter * 8, 3)

        with tf.variable_scope('full_connected'):
            x = tf.layers.flatten(x)
            x = tf.layers.dense(x, 100, name='fully_connected_1')
            x = tf.nn.leaky_relu(x, alpha=0.2, name='leakyReLU_1')
            x = tf.layers.dense(x, 1, name='fully_connected_2')

        return x


class Perceptual_VGG19(object):
    """the definition of VGG19. This network is used for constructing perceptual loss"""
    @staticmethod
    def build(x):
        # Block 1
        x = tf.layers.conv2d(x, 64, (3, 3), activation='relu', padding='same', name='block1_conv1')
        x = tf.layers.conv2d(x, 64, (3, 3), activation='relu', padding='same', name='block1_conv2')
        x = tf.layers.max_pooling2d(x, (2, 2), strides=(2, 2), name='block1_pool')

        # Block 2
        x = tf.layers.conv2d(x, 128, (3, 3), activation='relu', padding='same', name='block2_conv1')
        x = tf.layers.conv2d(x, 128, (3, 3), activation='relu', padding='same', name='block2_conv2')
        x = tf.layers.max_pooling2d(x, (2, 2), strides=(2, 2), name='block2_pool')

        # Block 3
        x = tf.layers.conv2d(x, 256, (3, 3), activation='relu', padding='same', name='block3_conv1')
        x = tf.layers.conv2d(x, 256, (3, 3), activation='relu', padding='same', name='block3_conv2')
        x = tf.layers.conv2d(x, 256, (3, 3), activation='relu', padding='same', name='block3_conv3')
        x = tf.layers.conv2d(x, 256, (3, 3), activation='relu', padding='same', name='block3_conv4')
        x = tf.layers.max_pooling2d(x, (2, 2), strides=(2, 2), name='block3_pool')

        # Block 4
        x = tf.layers.conv2d(x, 512, (3, 3), activation='relu', padding='same', name='block4_conv1')
        x = tf.layers.conv2d(x, 512, (3, 3), activation='relu', padding='same', name='block4_conv2')
        x = tf.layers.conv2d(x, 512, (3, 3), activation='relu', padding='same', name='block4_conv3')
        x = tf.layers.conv2d(x, 512, (3, 3), activation='relu', padding='same', name='block4_conv4')
        x = tf.layers.max_pooling2d(x, (2, 2), strides=(2, 2), name='block4_pool')

        # Block 5
        x = tf.layers.conv2d(x, 512, (3, 3), activation='relu', padding='same', name='block5_conv1')
        x = tf.layers.conv2d(x, 512, (3, 3), activation='relu', padding='same', name='block5_conv2')
        x = tf.layers.conv2d(x, 512, (3, 3), activation='relu', padding='same', name='block5_conv3')
        x = tf.layers.conv2d(x, 512, (3, 3), activation=None, padding='same', name='block5_conv4')

        return x
