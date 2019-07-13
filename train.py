import gc
import os
import math

from sklearn.utils import shuffle
import tensorflow as tf

from lib.ops import load_vgg19_weight
from lib.pretrain_generator import train_pretrain_generator
from lib.train_module import Network, Loss, Optimizer
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
    Flags.DEFINE_boolean('save_data', True, 'Whether to load and save data as npz file')
    Flags.DEFINE_string('train_result_dir', './train_result', 'output directory during training')
    Flags.DEFINE_boolean('crop', True, 'Whether image cropping is enabled')
    Flags.DEFINE_integer('crop_size', 128, 'the size of crop of training HR images')

    # About Network
    Flags.DEFINE_integer('scale_SR', 4, 'the scale of super-resolution')
    Flags.DEFINE_integer('num_repeat_RRDB', 10, 'The number of repeats of RRDB blocks')
    Flags.DEFINE_float('residual_scaling', 0.2, 'residual scaling parameter')
    Flags.DEFINE_integer('initialization_random_seed', 111, 'random seed of networks initialization')
    Flags.DEFINE_string('perceptual_loss', 'VGG19', 'the part of loss function. "VGG19" or "pixel-wise"')
    Flags.DEFINE_string('gan_loss_type', 'RaGAN', 'the type of GAN loss functions')

    # About training
    Flags.DEFINE_integer('num_iter', 20000, 'The number of iterations')
    Flags.DEFINE_integer('batch_size', 32, 'Mini-batch size')
    Flags.DEFINE_integer('channel', 3, 'Number of input/output image channel')
    Flags.DEFINE_boolean('pretrain_generator', True, 'Whether to pretrain generator')
    Flags.DEFINE_float('pretrain_learning_rate', 2e-4, 'learning rate for pretrain')
    Flags.DEFINE_float('pretrain_lr_decay_step', 20000, 'decay by every n iteration')
    Flags.DEFINE_float('learning_rate', 1e-4, 'learning rate')
    Flags.DEFINE_float('weight_initialize_scale', 0.1, 'scale to multiply after MSRA initialization')
    Flags.DEFINE_integer('HR_image_size', 128,
                         'Image width and height of HR image. This flag is valid when crop flag is set to false.')
    Flags.DEFINE_integer('LR_image_size', 32,
                         'Image width and height of LR image. This size should be 1/4 of HR_image_size exactly. '
                         'This flag is valid when crop flag is set to false.')
    Flags.DEFINE_integer('train_sample_save_freq', 1000, 'save samples during training every n iteration')
    Flags.DEFINE_integer('train_ckpt_save_freq', 2000, 'save checkpoint during training every n iteration')
    Flags.DEFINE_integer('train_summary_save_freq', 100, 'save summary during training every n iteration')
    Flags.DEFINE_float('epsilon', 1e-12, 'used in loss function')
    Flags.DEFINE_float('gan_loss_coeff', 0.005, 'used in perceptual loss')
    Flags.DEFINE_float('content_loss_coeff', 0.01, 'used in content loss')
    Flags.DEFINE_string('pre_train_checkpoint_dir', './pre_train_checkpoint', 'pre-train checkpoint directory')
    Flags.DEFINE_string('checkpoint_dir', './checkpoint', 'checkpoint directory')
    Flags.DEFINE_string('logdir', './log', 'log directory')

    return Flags.FLAGS


def main():
    # set flag
    FLAGS = set_flags()

    # make dirs
    target_dirs = [FLAGS.HR_data_dir, FLAGS.LR_data_dir, FLAGS.npz_data_dir, FLAGS.train_result_dir,
                   FLAGS.pre_train_checkpoint_dir, FLAGS.checkpoint_dir, FLAGS.logdir]
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
        gc.collect()

    LR_data = tf.placeholder(tf.float32, shape=[None, FLAGS.LR_image_size, FLAGS.LR_image_size, FLAGS.channel],
                             name='LR_input')
    HR_data = tf.placeholder(tf.float32, shape=[None, FLAGS.HR_image_size, FLAGS.HR_image_size, FLAGS.channel],
                             name='HR_input')

    # build Generator and Discriminator
    network = Network(FLAGS, LR_data, HR_data)
    gen_out = network.generator()
    dis_out_real, dis_out_fake = network.discriminator(gen_out)

    # build loss function
    loss = Loss()
    gen_loss, dis_loss = loss.gan_loss(FLAGS, HR_data, gen_out, dis_out_real, dis_out_fake)

    # define optimizers
    global_iter = tf.Variable(0, trainable=False)
    dis_var, dis_optimizer, gen_var, gen_optimizer = Optimizer().gan_optimizer(FLAGS, global_iter, dis_loss, gen_loss)

    # build summary writer
    tr_summary = tf.summary.merge(loss.add_summary_writer())

    num_train_data = len(HR_train)
    num_batch_in_train = int(math.floor(num_train_data / FLAGS.batch_size))
    num_epoch = int(math.ceil(FLAGS.num_iter / num_batch_in_train))

    HR_train, LR_train = normalize_images(HR_train, LR_train)

    fetches = {'dis_optimizer': dis_optimizer, 'gen_optimizer': gen_optimizer,
               'dis_loss': dis_loss, 'gen_loss': gen_loss,
               'gen_HR': gen_out,
               'summary': tr_summary
               }

    gc.collect()

    # Start Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(global_iter.initializer)

        writer = tf.summary.FileWriter(FLAGS.logdir, graph=sess.graph)

        pre_saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'))
        pre_saver.restore(sess, tf.train.latest_checkpoint(FLAGS.pre_train_checkpoint_dir))

        if FLAGS.perceptual_loss == 'VGG19':
            sess.run(load_vgg19_weight(FLAGS))

        saver = tf.train.Saver(max_to_keep=10)

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

                # update weights of G/D
                result = sess.run(fetches=fetches, feed_dict=feed_dict)

                # save summary every n iter
                if current_iter % FLAGS.train_summary_save_freq == 0:
                    writer.add_summary(result['summary'], global_step=current_iter)

                # save samples every n iter
                if current_iter % FLAGS.train_sample_save_freq == 0:
                    print('iteration : ', current_iter, ' gen_loss', result['gen_loss'])
                    print('iteration : ', current_iter, ' dis_loss', result['dis_loss'])

                    save_image(FLAGS, result['gen_HR'], 'train', current_iter, save_max_num=5)

                if current_iter % FLAGS.train_ckpt_save_freq == 0:
                    saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'gen'), global_step=current_iter)

        writer.close()


if __name__ == '__main__':
    main()
