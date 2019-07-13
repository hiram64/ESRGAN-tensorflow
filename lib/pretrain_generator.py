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

    global_iter = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.pretrain_learning_rate, global_iter,
                                               FLAGS.pretrain_lr_decay_step, 0.5, staircase=True)
    with tf.name_scope('optimizer'):
        with tf.variable_scope('generator_optimizer'):
            pre_gen_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            pre_gen_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=pre_gen_loss,
                                                                                             global_step=global_iter,
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

    fetches = {'pre_gen_loss': pre_gen_loss, 'pre_gen_optimizer': pre_gen_optimizer, 'gen_HR': pre_gen_out,
               'summary': pre_summary}

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(global_iter.initializer)
        sess.run(scale_initialization(pre_gen_var, FLAGS))

        writer = tf.summary.FileWriter(FLAGS.logdir, graph=sess.graph, filename_suffix='pre-train')

        for epoch in range(num_epoch):
            print('epoch: ', epoch, flush=True)
            HR_train, LR_train = shuffle(HR_train, LR_train, random_state=222)

            for iteration in range(num_batch_in_train):
                current_iter = tf.train.global_step(sess, global_iter)

                if current_iter > FLAGS.num_iter:
                    break

                feed_dict = {
                    HR_data: HR_train[iteration * FLAGS.batch_size:iteration * FLAGS.batch_size + FLAGS.batch_size],
                    LR_data: LR_train[iteration * FLAGS.batch_size:iteration * FLAGS.batch_size + FLAGS.batch_size]
                }

                # update weights
                result = sess.run(fetches=fetches, feed_dict=feed_dict)

                # save summary every n iter
                if current_iter % FLAGS.train_summary_save_freq == 0:
                    writer.add_summary(result['summary'], global_step=current_iter)

                # save samples every n iter
                if current_iter % FLAGS.train_sample_save_freq == 0:
                    print('iteration : ', current_iter, ' pixel_loss', result['pre_gen_loss'])
                    save_image(FLAGS, result['gen_HR'], 'pre-train', current_iter, save_max_num=5)

                # save checkpoint
                if current_iter % FLAGS.train_ckpt_save_freq == 0:
                    saver.save(sess, os.path.join(FLAGS.pre_train_checkpoint_dir, 'pre_gen'), global_step=current_iter)

        writer.close()
