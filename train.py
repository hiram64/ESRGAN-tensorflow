import os
import math

from sklearn.utils import shuffle
import tensorflow as tf

from lib.network import Generator, Discriminator
from lib.ops import scale_initialization
from lib.pretrain_generator import train_pretrain_generator
from lib.utils import create_dirs, normalize_images, save_image, load_npz_data, load_and_save_data


def set_flags():
    Flags = tf.app.flags

    # About data
    Flags.DEFINE_string('data_dir', './data/LSUN', 'data directory')
    Flags.DEFINE_string('HR_data_dir', './data/HR_data', 'HR data directory')
    Flags.DEFINE_string('LR_data_dir', './data/LR_data', 'LR data directory')
    Flags.DEFINE_string('npz_data_dir', './data/npz', 'The npz data dir')
    Flags.DEFINE_string('HR_npz_filename', 'HR_image.npz', 'the filename of HR image npz file')
    Flags.DEFINE_string('LR_npz_filename', 'LR_image.npz', 'the filename of LR image npz file')
    Flags.DEFINE_boolean('save_data', False, 'Whether to load and save data as npz file')
    Flags.DEFINE_string('train_result_dir', './train_result', 'output directory during training')
    Flags.DEFINE_boolean('crop', True, 'Whether image cropping is enabled')
    Flags.DEFINE_integer('crop_size', 128, 'the size of crop of training HR images')

    # About Network
    Flags.DEFINE_integer('scale_SR', 4, 'the scale of super-resolution')
    Flags.DEFINE_integer('num_repeat_RRDB', 10, 'The number of repeats of RRDB blocks')
    Flags.DEFINE_float('residual_scaling', 0.2, 'residual scaling parameter')
    Flags.DEFINE_integer('initialization_random_seed', 111, 'random seed of networks initialization')
    Flags.DEFINE_string('perceptual_loss', 'pixel-wise', 'the part of loss function')
    Flags.DEFINE_string('gan_loss_type', 'RaGAN', 'the type of GAN loss functions')

    # About training
    Flags.DEFINE_boolean('pretrain_generator', True, 'Whether to pretrain generator')
    Flags.DEFINE_integer('num_iter', 20000, 'The number of iterations')
    Flags.DEFINE_integer('batch_size', 32, 'Mini-batch size')
    Flags.DEFINE_integer('channel', 3, 'Number of input/output image channel')
    Flags.DEFINE_float('learning_rate', 2e-4, 'learning rate')
    Flags.DEFINE_float('weight_initialize_scale', 0.1, 'scale to multiply after MSRA initialization')
    Flags.DEFINE_integer('HR_image_size', 128, 'Image width and height of HR image')
    Flags.DEFINE_integer('LR_image_size', 32,
                         'Image width and height of LR image. This size should be 1/4 of HR_image_size exactly.')
    Flags.DEFINE_integer('train_sample_save_freq', 1000, 'save samples during training every n iteration')
    Flags.DEFINE_integer('train_ckpt_save_freq', 2000, 'save checkpoint during training every n iteration')
    Flags.DEFINE_integer('train_summary_save_freq', 100, 'save summary during training every n iteration')
    Flags.DEFINE_float('epsilon', 1e-12, 'used in loss function')
    Flags.DEFINE_float('gan_loss_coeff', 0.005, 'used in perceptual loss')
    Flags.DEFINE_float('content_loss_coeff', 0.01, 'used in perceptual loss')
    Flags.DEFINE_string('pre_train_checkpoint_dir', './pre_train_checkpoint', 'checkpoint directory')
    Flags.DEFINE_string('checkpoint_dir', './checkpoint', 'checkpoint directory')
    Flags.DEFINE_string('log_dir', './log', 'log directory')

    return Flags.FLAGS


def main():
    # set flag
    FLAGS = set_flags()

    # make dirs
    target_dirs = [FLAGS.HR_data_dir, FLAGS.LR_data_dir, FLAGS.npz_data_dir, FLAGS.train_result_dir,
                   FLAGS.pre_train_checkpoint_dir, FLAGS.checkpoint_dir, FLAGS.log_dir]
    create_dirs(target_dirs)

    # load data
    if FLAGS.save_data:
        HR_train, LR_train = load_and_save_data(FLAGS)
    else:
        HR_train, LR_train = load_npz_data(FLAGS)

    # pre-train generator with pixel-wise loss and save the trained model
    if FLAGS.pretrain_generator:
        train_pretrain_generator(FLAGS, LR_train, HR_train)
        tf.reset_default_graph()

    LR_data = tf.placeholder(tf.float32, shape=[None, FLAGS.LR_image_size, FLAGS.LR_image_size, FLAGS.channel],
                             name='LR_input')
    HR_data = tf.placeholder(tf.float32, shape=[None, FLAGS.HR_image_size, FLAGS.HR_image_size, FLAGS.channel],
                             name='HR_input')

    # build Generator and Discriminator
    with tf.name_scope('generator'):
        with tf.variable_scope('generator'):
            generator = Generator(FLAGS)
            gen_out = generator.build(LR_data)

    discriminator = Discriminator(FLAGS)
    with tf.name_scope('real_discriminator'):
        with tf.variable_scope('discriminator', reuse=False):
            dis_out_real = discriminator.build(HR_data)

    with tf.name_scope('fake_discriminator'):
        with tf.variable_scope('discriminator', reuse=True):
            dis_out_fake = discriminator.build(gen_out)

    # define loss functions
    with tf.name_scope('loss_function'):
        with tf.variable_scope('generator_loss'):
            if FLAGS.gan_loss_type == 'RaGAN':
                g_loss_p1 = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_out_real - tf.reduce_mean(dis_out_fake),
                                                            labels=tf.zeros_like(dis_out_real)))

                g_loss_p2 = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_out_fake - tf.reduce_mean(dis_out_real),
                                                            labels=tf.ones_like(dis_out_fake)))

                gen_loss = FLAGS.gan_loss_coeff * (g_loss_p1 + g_loss_p2) / 2
            else:
                gen_loss = FLAGS.gan_loss_coeff * tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_out_fake, labels=tf.ones_like(dis_out_fake)))

            # content loss : L1 distance
            content_loss = FLAGS.content_loss_coeff * tf.reduce_mean(
                tf.reduce_sum(tf.abs(gen_out - HR_data), axis=[1, 2, 3]))

            gen_loss += content_loss

            # perceptual loss
            if FLAGS.perceptual_loss == 'pixel-wise':
                perc_loss = tf.reduce_mean(tf.reduce_mean(tf.square(gen_out - HR_data), axis=3))
                gen_loss += perc_loss

        with tf.variable_scope('discriminator_loss'):
            if FLAGS.gan_loss_type == 'RaGAN':
                d_loss_real = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_out_real - tf.reduce_mean(dis_out_fake),
                                                            labels=tf.ones_like(dis_out_real))) / 2

                d_loss_fake = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_out_fake - tf.reduce_mean(dis_out_real),
                                                            labels=tf.zeros_like(dis_out_fake))) / 2

                dis_loss = d_loss_real + d_loss_fake
            else:
                d_loss_real = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_out_real, labels=tf.ones_like(dis_out_real)))
                d_loss_fake = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_out_fake, labels=tf.zeros_like(dis_out_fake)))

                dis_loss = d_loss_real + d_loss_fake

    # define optimizers
    with tf.name_scope('optimizer'):
        with tf.variable_scope('discriminator_optimizer'):
            dis_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
            dis_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss=dis_loss,
                                                                                               var_list=dis_var)

        with tf.variable_scope('generator_optimizer'):
            with tf.control_dependencies([dis_optimizer]):
                gen_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
                gen_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss=gen_loss,
                                                                                                   var_list=gen_var)
    # summary writer
    tr_summary = tf.summary.merge([
        tf.summary.scalar('generator_loss', gen_loss),
        tf.summary.scalar('content_loss', content_loss),
        tf.summary.scalar('perceptual_loss', perc_loss),
        tf.summary.scalar('discriminator_loss', dis_loss),
        tf.summary.scalar('discriminator_fake_loss', d_loss_fake),
        tf.summary.scalar('discriminator_real_loss', d_loss_real)
    ])

    # Start Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    num_train_data = len(HR_train)
    num_batch_in_train = int(math.floor(num_train_data / FLAGS.batch_size))
    num_epoch = int(math.ceil(FLAGS.num_iter / num_batch_in_train))

    HR_train, LR_train = normalize_images(HR_train, LR_train)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(scale_initialization(dis_var, FLAGS))
        writer = tf.summary.FileWriter(FLAGS.log_dir, graph=sess.graph)
        global_iter = 0

        pre_saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'))
        pre_saver.restore(sess, tf.train.latest_checkpoint(FLAGS.pre_train_checkpoint_dir))

        saver = tf.train.Saver(max_to_keep=10)

        for epoch in range(num_epoch):
            print('epoch: ', epoch, flush=True)
            HR_train, LR_train = shuffle(HR_train, LR_train, random_state=222)

            for iteration in range(num_batch_in_train):
                if global_iter > FLAGS.num_iter:
                    break

                fetches = {'dis_optimizer': dis_optimizer, 'gen_optimizer': gen_optimizer,
                           'dis_loss': dis_loss, 'gen_loss': gen_loss,
                           'gen_HR': gen_out,
                           'summary': tr_summary
                           }

                feed_dict = {
                    HR_data: HR_train[iteration * FLAGS.batch_size:iteration * FLAGS.batch_size + FLAGS.batch_size],
                    LR_data: LR_train[iteration * FLAGS.batch_size:iteration * FLAGS.batch_size + FLAGS.batch_size]
                }

                # update weights of G/D
                result = sess.run(fetches=fetches, feed_dict=feed_dict)

                # save summary every n iter
                if global_iter % FLAGS.train_summary_save_freq == 0:
                    writer.add_summary(result['summary'], global_step=global_iter)

                # save samples every n iter
                if global_iter % FLAGS.train_sample_save_freq == 0:
                    print('iteration : ', global_iter, ' gen_loss', result['gen_loss'])
                    print('iteration : ', global_iter, ' dis_loss', result['dis_loss'])

                    save_image(FLAGS, result['gen_HR'], 'train', global_iter, save_max_num=5)

                if global_iter % FLAGS.train_ckpt_save_freq == 0:
                    saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'gen'), global_step=global_iter)

                global_iter += 1

        writer.close()


if __name__ == '__main__':
    main()
