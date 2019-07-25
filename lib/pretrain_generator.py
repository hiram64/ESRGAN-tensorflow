import gc
import os
import math

from sklearn.utils import shuffle
import tensorflow as tf

from lib.ops import scale_initialization
from lib.train_module import Network, Loss, Optimizer
from lib.utils import log, normalize_images, save_image


def train_pretrain_generator(FLAGS, LR_train, HR_train, logflag):
    """pre-train deep network as initialization weights of ESRGAN Generator"""
    log(logflag, 'Pre-train : Process start', 'info')

    LR_data = tf.placeholder(tf.float32, shape=[None, FLAGS.LR_image_size, FLAGS.LR_image_size, FLAGS.channel],
                             name='LR_input')
    HR_data = tf.placeholder(tf.float32, shape=[None, FLAGS.HR_image_size, FLAGS.HR_image_size, FLAGS.channel],
                             name='HR_input')

    # build Generator
    network = Network(FLAGS, LR_data)
    pre_gen_out = network.generator()

    # build loss function
    loss = Loss()
    pre_gen_loss = loss.pretrain_loss(pre_gen_out, HR_data)

    # build optimizer
    global_iter = tf.Variable(0, trainable=False)
    pre_gen_var, pre_gen_optimizer = Optimizer().pretrain_optimizer(FLAGS, global_iter, pre_gen_loss)

    # build summary writer
    pre_summary = tf.summary.merge(loss.add_summary_writer())

    num_train_data = len(HR_train)
    num_batch_in_train = int(math.floor(num_train_data / FLAGS.batch_size))
    num_epoch = int(math.ceil(FLAGS.num_iter / num_batch_in_train))

    HR_train, LR_train = normalize_images(HR_train, LR_train)

    fetches = {'pre_gen_loss': pre_gen_loss, 'pre_gen_optimizer': pre_gen_optimizer, 'gen_HR': pre_gen_out,
               'summary': pre_summary}

    gc.collect()

    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            allow_growth=True,
            visible_device_list=FLAGS.gpu_dev_num
        )
    )

    saver = tf.train.Saver(max_to_keep=10)

    # Start session
    with tf.Session(config=config) as sess:
        log(logflag, 'Pre-train : Training starts', 'info')

        sess.run(tf.global_variables_initializer())
        sess.run(global_iter.initializer)
        sess.run(scale_initialization(pre_gen_var, FLAGS))

        writer = tf.summary.FileWriter(FLAGS.logdir, graph=sess.graph, filename_suffix='pre-train')

        for epoch in range(num_epoch):
            log(logflag, 'Pre-train Epoch: {0}'.format(epoch), 'info')

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
                    log(logflag,
                        'Pre-train iteration : {0}, pixel-wise_loss : {1}'.format(current_iter, result['pre_gen_loss']),
                        'info')
                    save_image(FLAGS, result['gen_HR'], 'pre-train', current_iter, save_max_num=5)

                # save checkpoint
                if current_iter % FLAGS.train_ckpt_save_freq == 0:
                    saver.save(sess, os.path.join(FLAGS.pre_train_checkpoint_dir, 'pre_gen'), global_step=current_iter)

        writer.close()
        log(logflag, 'Pre-train : Process end', 'info')
