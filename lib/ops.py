import tensorflow as tf


def scale_initialization(weights, FLAGS):
    return [tf.assign(weight, weight * FLAGS.weight_initialize_scale) for weight in weights]
