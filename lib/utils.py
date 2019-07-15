import logging
import os
import glob

import cv2
import numpy as np


def log(logflag, message, level='info'):
    """logging to stdout and logfile if flag is true"""
    print(message, flush=True)

    if logflag:
        if level == 'info':
            logging.info(message)
        elif level == 'warning':
            logging.warning(message)
        elif level == 'error':
            logging.error(message)
        elif level == 'critical':
            logging.critical(message)


def create_dirs(target_dirs):
    """create necessary directories to save output files"""
    for dir_path in target_dirs:
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)


def normalize_images(*arrays):
    """normalize input image arrays"""
    return [arr / 127.5 - 1 for arr in arrays]


def de_normalize_image(image):
    """de-normalize input image array"""
    return (image + 1) * 127.5


def save_image(FLAGS, images, phase, global_iter, save_max_num=5):
    """save images in specified directory"""
    if phase == 'train' or phase == 'pre-train':
        save_dir = FLAGS.train_result_dir
    elif phase == 'inference':
        save_dir = FLAGS.inference_result_dir
        save_max_num = len(images)
    else:
        print('specified phase is invalid')

    for i, img in enumerate(images):
        if i >= save_max_num:
            break

        cv2.imwrite(save_dir + '/{0}_HR_{1}_{2}.jpg'.format(phase, global_iter, i), de_normalize_image(img))


def crop(img, FLAGS):
    """crop patch from an image with specified size"""
    img_h, img_w, _ = img.shape

    rand_h = np.random.randint(img_h - FLAGS.crop_size)
    rand_w = np.random.randint(img_w - FLAGS.crop_size)

    return img[rand_h:rand_h + FLAGS.crop_size, rand_w:rand_w + FLAGS.crop_size, :]


def data_augmentation(LR_images, HR_images, aug_type='horizontal_flip'):
    """data augmentation. input arrays should be [N, H, W, C]"""

    if aug_type == 'horizontal_flip':
        return LR_images[:, :, ::-1, :], HR_images[:, :, ::-1, :]
    elif aug_type == 'rotation_90':
        return np.rot90(LR_images, k=1, axes=(1, 2)), np.rot90(HR_images, k=1, axes=(1, 2))


def load_and_save_data(FLAGS, logflag):
    """make HR and LR data. And save them as npz files"""
    assert os.path.isdir(FLAGS.data_dir) is True, 'Directory specified by data_dir does not exist or is not a directory'

    all_file_path = glob.glob(FLAGS.data_dir + '/*')
    assert len(all_file_path) > 0, 'No file in the directory'

    ret_HR_image = []
    ret_LR_image = []

    for file in all_file_path:
        img = cv2.imread(file)
        filename = file.rsplit('/', 1)[-1]

        # crop patches if flag is true. Otherwise just resize HR and LR images
        if FLAGS.crop:
            for _ in range(FLAGS.num_crop_per_image):
                img_h, img_w, _ = img.shape

                if (img_h < FLAGS.crop_size) or (img_w < FLAGS.crop_size):
                    print('Skip crop target image because of insufficient size')
                    continue

                HR_image = crop(img, FLAGS)
                LR_crop_size = np.int(np.floor(FLAGS.crop_size / FLAGS.scale_SR))
                LR_image = cv2.resize(HR_image, (LR_crop_size, LR_crop_size), interpolation=cv2.INTER_LANCZOS4)

                cv2.imwrite(FLAGS.HR_data_dir + '/' + filename, HR_image)
                cv2.imwrite(FLAGS.LR_data_dir + '/' + filename, LR_image)

                ret_HR_image.append(HR_image)
                ret_LR_image.append(LR_image)
        else:
            HR_image = cv2.resize(img, (FLAGS.HR_image_size, FLAGS.HR_image_size), interpolation=cv2.INTER_LANCZOS4)
            LR_image = cv2.resize(img, (FLAGS.LR_image_size, FLAGS.LR_image_size), interpolation=cv2.INTER_LANCZOS4)

            cv2.imwrite(FLAGS.HR_data_dir + '/' + filename, HR_image)
            cv2.imwrite(FLAGS.LR_data_dir + '/' + filename, LR_image)

            ret_HR_image.append(HR_image)
            ret_LR_image.append(LR_image)

    assert len(ret_HR_image) > 0 and len(ret_LR_image) > 0, 'No availale image is found in the directory'
    log(logflag, 'Data process : {} images are processed'.format(len(ret_HR_image)), 'info')

    ret_HR_image = np.array(ret_HR_image)
    ret_LR_image = np.array(ret_LR_image)

    if FLAGS.data_augmentation:
        LR_flip, HR_flip = data_augmentation(ret_LR_image, ret_HR_image, aug_type='horizontal_flip')
        LR_rot, HR_rot = data_augmentation(ret_LR_image, ret_HR_image, aug_type='rotation_90')

        ret_LR_image = np.append(ret_LR_image, LR_flip, axis=0)
        ret_HR_image = np.append(ret_HR_image, HR_flip, axis=0)
        ret_LR_image = np.append(ret_LR_image, LR_rot, axis=0)
        ret_HR_image = np.append(ret_HR_image, HR_rot, axis=0)

        del LR_flip, HR_flip, LR_rot, HR_rot

    np.savez(FLAGS.npz_data_dir + '/' + FLAGS.HR_npz_filename, images=ret_HR_image)
    np.savez(FLAGS.npz_data_dir + '/' + FLAGS.LR_npz_filename, images=ret_LR_image)

    return ret_HR_image, ret_LR_image


def load_npz_data(FLAGS):
    """load array data from data_path"""
    return np.load(FLAGS.npz_data_dir + '/' + FLAGS.HR_npz_filename)['images'], \
           np.load(FLAGS.npz_data_dir + '/' + FLAGS.LR_npz_filename)['images']


def load_inference_data(FLAGS):
    """load data from directory for inference"""
    assert os.path.isdir(FLAGS.data_dir) is True, 'Directory specified by data_dir does not exist or is not a directory'

    all_file_path = glob.glob(FLAGS.data_dir + '/*')
    assert len(all_file_path) > 0, 'No file in the directory'

    ret_LR_image = []
    ret_filename = []

    for file in all_file_path:
        img = cv2.imread(file)
        img = normalize_images(img)
        ret_LR_image.append(img[0][np.newaxis, ...])

        ret_filename.append(file.rsplit('/', 1)[-1])

    assert len(ret_LR_image) > 0, 'No available image is found in the directory'

    return ret_LR_image, ret_filename
