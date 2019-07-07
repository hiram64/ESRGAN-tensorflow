import os
import math

from sklearn.utils import shuffle
import tensorflow as tf

from lib.network import Generator
from lib.ops import scale_initialization
from lib.utils import normalize_images, save_image


def train_pretrain_generator(FLAGS, LR_train, HR_train):
    """pre-train deep network as initialization weights of ESRGAN Generator"""
    LR_data = tf.placeholder(tf.float32, shape=[None, FLAGS.LR_image_size, FLAGS.LR_image_size, FLAGS.channel],
                             name='LR_input')
    HR_data = tf.placeholder(tf.float32, shape=[None, FLAGS.HR_image_size, FLAGS.HR_image_size, FLAGS.channel],
                             name='HR_input')

    with tf.name_scope('generator'):
        with tf.variable_scope('generator'):
            pretrain_generator = Generator(FLAGS)
            pre_gen_out = pretrain_generator.build(LR_data)

    with tf.name_scope('loss_function'):
        with tf.variable_scope('pixel-wise_loss'):
            pre_gen_loss = tf.reduce_mean(tf.reduce_mean(tf.square(pre_gen_out - HR_data), axis=3))

    with tf.name_scope('optimizer'):
        with tf.variable_scope('generator_optimizer'):
            pre_gen_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            pre_gen_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss=pre_gen_loss,
                                                                                                   var_list=pre_gen_var)
    # summary writer
    pre_summary = tf.summary.merge([tf.summary.scalar('pre-train : pixel-wise_loss', pre_gen_loss)])

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver(max_to_keep=10)

    num_train_data = len(HR_train)
    num_batch_in_train = int(math.floor(num_train_data / FLAGS.batch_size))
    num_epoch = int(math.ceil(FLAGS.num_iter / num_batch_in_train))

    HR_train, LR_train = normalize_images(HR_train, LR_train)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(scale_initialization(pre_gen_var, FLAGS))

        writer = tf.summary.FileWriter(FLAGS.log_dir, graph=sess.graph, filename_suffix='pre-train')
        global_iter = 0

        for epoch in range(num_epoch):
            print('epoch: ', epoch, flush=True)
            HR_train, LR_train = shuffle(HR_train, LR_train, random_state=222)

            for iteration in range(num_batch_in_train):
                if global_iter > FLAGS.num_iter:
                    break

                fetches = {'pre_gen_loss': pre_gen_loss, 'pre_gen_optimizer': pre_gen_optimizer, 'gen_HR': pre_gen_out,
                           'summary': pre_summary}

                feed_dict = {
                    HR_data: HR_train[iteration * FLAGS.batch_size:iteration * FLAGS.batch_size + FLAGS.batch_size],
                    LR_data: LR_train[iteration * FLAGS.batch_size:iteration * FLAGS.batch_size + FLAGS.batch_size]
                }

                # update weights
                result = sess.run(fetches=fetches, feed_dict=feed_dict)

                # save summary every n iter
                if global_iter % FLAGS.train_summary_save_freq == 0:
                    writer.add_summary(result['summary'], global_step=global_iter)

                # save samples every n iter
                if global_iter % FLAGS.train_sample_save_freq == 0:
                    print('iteration : ', global_iter, ' pixel_loss', result['pre_gen_loss'])
                    save_image(FLAGS, result['gen_HR'], 'pre-train', global_iter, save_max_num=5)

                # save checkpoint
                if global_iter % FLAGS.train_ckpt_save_freq == 0:
                    saver.save(sess, os.path.join(FLAGS.pre_train_checkpoint_dir, 'pre_gen'), global_step=global_iter)

                global_iter += 1

        writer.close()
