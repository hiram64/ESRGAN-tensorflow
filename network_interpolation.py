import os

import cv2
import numpy as np
import tensorflow as tf

from lib.network import Generator
from lib.ops import extract_weight, interpolate_weight
from lib.utils import create_dirs, de_normalize_image, load_inference_data


def set_flags():
    Flags = tf.app.flags

    Flags.DEFINE_string('data_dir', './data/inference', 'inference data directory')
    Flags.DEFINE_string('checkpoint_dir', './checkpoint', 'checkpoint directory')
    Flags.DEFINE_string('interpolation_result_dir', './interpolation_result', 'output directory during inference')
    Flags.DEFINE_integer('channel', 3, 'Number of input/output image channel')
    Flags.DEFINE_integer('num_repeat_RRDB', 10, 'The number of repeats of RRDB blocks')
    Flags.DEFINE_float('residual_scaling', 0.2, 'residual scaling parameter')
    Flags.DEFINE_string('pre_train_checkpoint_dir', './pre_train_checkpoint', 'checkpoint directory')
    Flags.DEFINE_integer('initialization_random_seed', 111, 'random seed of networks initialization')
    Flags.DEFINE_float('interpolation_param', 0.8, 'random seed of networks initialization')

    return Flags.FLAGS


def main():
    # set flag
    FLAGS = set_flags()

    # make dirs
    target_dirs = [FLAGS.interpolation_result_dir]
    create_dirs(target_dirs)

    # load test data
    LR_inference, LR_filenames = load_inference_data(FLAGS)
    LR_data = tf.placeholder(tf.float32, shape=[1, None, None, FLAGS.channel], name='LR_input')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    ### Load Pretrain model weights
    # build Generator
    with tf.name_scope('generator'):
        with tf.variable_scope('generator'):
            gen_out = Generator(FLAGS).build(LR_data)

    with tf.Session(config=config) as sess:
        pre_saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'))
        pre_saver.restore(sess, tf.train.latest_checkpoint(FLAGS.pre_train_checkpoint_dir))

        pretrain_weight = extract_weight(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'))

    tf.reset_default_graph()

    ### Run network interpolation and inference
    LR_data = tf.placeholder(tf.float32, shape=[1, None, None, FLAGS.channel], name='LR_input')

    with tf.name_scope('generator'):
        with tf.variable_scope('generator'):
            gen_out = Generator(FLAGS).build(LR_data)

    with tf.Session(config=config) as sess:
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator'))
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))

        sess.run(interpolate_weight(FLAGS, pretrain_weight))

        for i, test_img in enumerate(LR_inference):
            fetches = {'gen_HR': gen_out}

            feed_dict = {
                LR_data: test_img
            }

            result = sess.run(fetches=fetches, feed_dict=feed_dict)

            cv2.imwrite(os.path.join(FLAGS.inference_result_dir, LR_filenames[i]),
                        de_normalize_image(np.squeeze(result['gen_HR'])))


if __name__ == '__main__':
    main()
