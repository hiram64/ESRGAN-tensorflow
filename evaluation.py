import argparse
import glob
import os

import cv2
from skimage.measure import compare_psnr, compare_ssim


def calc_measures(hr_path, calc_psnr=True, calc_ssim=True):
    """calculate PSNR and SSIM for all HR images and their mean.
    These paired images should have the same filename.
    """

    HR_files = glob.glob(hr_path + '/*')
    mean_psnr = 0
    mean_ssim = 0

    for file in HR_files:
        hr_img = cv2.imread(file)
        filename = file.rsplit('/', 1)[-1]
        path = os.path.join(args.inference_result, filename)

        if not os.path.isfile(path):
            raise FileNotFoundError('')

        inf_img = cv2.imread(path)

        # compare HR image and inferenced image with measures
        print('-' * 10)
        if calc_psnr:
            psnr = compare_psnr(hr_img, inf_img)
            print('{0} : PSNR {1:.3f} dB'.format(filename, psnr))
            mean_psnr += psnr
        if calc_ssim:
            ssim = compare_ssim(hr_img, inf_img, multichannel=True)
            print('{0} : SSIM {1:.3f}'.format(filename, ssim))
            mean_ssim += ssim

    print('-' * 10)
    if calc_psnr:
        print('mean-PSNR {:.3f} dB'.format(mean_psnr / len(HR_files)))
    if calc_ssim:
        print('mean-SSIM {:.3f}'.format(mean_ssim / len(HR_files)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--HR_data_dir', default='./data/div2_inf_HR', type=str)
    parser.add_argument('--inference_result', default='./inference_result_div2', type=str)

    args = parser.parse_args()

    calc_measures(args.HR_data_dir, calc_psnr=True, calc_ssim=True)
