import argparse
import glob
import math
import os

import cv2
import numpy as np


def visualize(args):
    """visualize images. Bicubic interpolation(generated in this script), ESRGAN, ESRGAN with network interpolation
    and HR are tiled into an image.
    The images of ESRGAN, ESRGAN with network interpolation have the same filename as HR.
    """
    HR_files = glob.glob(args.HR_data_dir + '/*')

    for file in HR_files:
        # HR(GT)
        hr_img = cv2.imread(file)
        h, w, _ = hr_img.shape
        filename = file.rsplit('/', 1)[-1].rsplit('.', 1)[0]

        # LR -> bicubic
        r_h, r_w = math.floor(h / 4), math.floor(w / 4)
        lr_img = cv2.resize(hr_img, (r_w, r_h), cv2.INTER_CUBIC)
        bic_img = cv2.resize(lr_img, (w, h), cv2.INTER_CUBIC)

        # inference
        inf_path_jpg = os.path.join(args.inference_result, filename + '.jpg')
        inf_path_png = os.path.join(args.inference_result, filename + '.png')

        if os.path.isfile(inf_path_jpg):
            inf_path = inf_path_jpg
        elif os.path.isfile(inf_path_png):
            inf_path = inf_path_png
        else:
            raise FileNotFoundError('Images should have the same filename as HR image and be the formats of jpg or png')

        inf_img = cv2.imread(inf_path)

        # network interpolation inference
        ni_path_jpg = os.path.join(args.network_interpolation_result, filename + '.jpg')
        ni_path_png = os.path.join(args.network_interpolation_result, filename + '.png')
        if os.path.isfile(ni_path_jpg):
            ni_path = ni_path_jpg
        elif os.path.isfile(inf_path_png):
            ni_path = ni_path_png
        else:
            raise FileNotFoundError('Images should have the same filename as HR image and be the formats of jpg or png')

        ni_img = cv2.imread(ni_path)

        h_upper = int(math.floor(h / 2) + args.path_size / 2)
        h_lower = int(math.floor(h / 2) - args.path_size / 2)

        w_right = int(math.floor(w / 2) + args.path_size / 2)
        w_left = int(math.floor(w / 2) - args.path_size / 2)

        h_size = h_upper - h_lower
        w_size = w_right - w_left

        out_arr = np.empty((h_size, w_size * 4, 3))

        # tile images from left to right : bicubic -> ESRGAN-inference -> Network interpolation -> HR(GT)
        out_arr[:, :w_size, :] = bic_img[h_lower:h_upper, w_left:w_right, :]
        out_arr[:, w_size:w_size * 2, :] = inf_img[h_lower:h_upper, w_left:w_right, :]
        out_arr[:, w_size * 2:w_size * 3, :] = ni_img[h_lower:h_upper, w_left:w_right, :]
        out_arr[:, w_size * 3:w_size * 4, :] = hr_img[h_lower:h_upper, w_left:w_right, :]

        cv2.imwrite(args.output_dir + '/' + '{}.png'.format(filename), out_arr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--HR_data_dir', default='./data/div2_inf_HR', type=str)
    parser.add_argument('--inference_result', default='./inference_result_div2', type=str)
    parser.add_argument('--network_interpolation_result', default='./interpolation_result_div2', type=str)
    parser.add_argument('--path_size', default=512, type=str)
    parser.add_argument('--output_dir', default='./', type=str)

    args = parser.parse_args()

    visualize(args)
