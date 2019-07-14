from collections import OrderedDict

import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19


def scale_initialization(weights, FLAGS):
    return [tf.assign(weight, weight * FLAGS.weight_initialize_scale) for weight in weights]


def _transfer_vgg19_weight(FLAGS, weight_dict):
    from_model = VGG19(include_top=False, weights='imagenet', input_tensor=None,
                       input_shape=(FLAGS.HR_image_size, FLAGS.HR_image_size, FLAGS.channel))

    fetch_weight = []

    for layer in from_model.layers:
        if 'conv' in layer.name:
            W, b = layer.get_weights()

            fetch_weight.append(
                tf.assign(weight_dict['loss_generator/perceptual_vgg19/{}/kernel'.format(layer.name)], W)
            )
            fetch_weight.append(
                tf.assign(weight_dict['loss_generator/perceptual_vgg19/{}/bias'.format(layer.name)], b)
            )

    return fetch_weight


def load_vgg19_weight(FLAGS):
    vgg_weight = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='loss_generator/perceptual_vgg19')

    assert len(vgg_weight) > 0, 'No VGG19 weight was collected. The target scope might be wrong.'

    weight_dict = {}
    for weight in vgg_weight:
        weight_dict[weight.name.rsplit(':', 1)[0]] = weight

    return _transfer_vgg19_weight(FLAGS, weight_dict)


def extract_weight(network_vars):
    weight_dict = OrderedDict()

    for weight in network_vars:
        weight_dict[weight.name] = weight.eval()

    return weight_dict


def interpolate_weight(FLAGS, pretrain_weight):
    fetch_weight = []
    alpha = FLAGS.interpolation_param

    for name, pre_weight in pretrain_weight.items():
        esrgan_weight = tf.get_default_graph().get_tensor_by_name(name)

        assert pre_weight.shape == esrgan_weight.shape, 'The shape of weights does not match'

        fetch_weight.append(tf.assign(esrgan_weight, (1 - alpha) * pre_weight + alpha * esrgan_weight))

    return fetch_weight
