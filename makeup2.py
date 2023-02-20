import cv2
import os
import numpy as np
from skimage.filters import gaussian
from test import evaluate
import argparse
from pdb import set_trace

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--img-path', default='imgs/116.jpg')
    return parse.parse_args()


def sharpen(img):
    img = img * 1.0
    gauss_out = gaussian(img, sigma=5, multichannel=True)

    alpha = 1.5
    img_out = (img - gauss_out) * alpha + img

    img_out = img_out / 255.0

    mask_1 = img_out < 0
    mask_2 = img_out > 1

    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2
    img_out = np.clip(img_out, 0, 1)
    img_out = img_out * 255
    return np.array(img_out, dtype=np.uint8)


def hair(image, parsing, part=17, color=[230, 50, 20]):
    


    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # tar_hsv = cv2.cvtColor(tar_color, cv2.COLOR_BGR2HSV)
    changed_hsv = image_hsv.copy()

    image_hsv_gaussian = image_hsv.copy()
    image_hsv_gaussian = gaussian(image_hsv_gaussian, sigma=1, channel_axis = 0)

    for part, color in zip(parts, colors):
        # face
        if part == 1:
            image_hsv[parsing == part] = image_hsv_gaussian[parsing == part]
        # lip
        if part == 12 or part == 13:
            b, g, r = color      #[10, 50, 250]       # [10, 250, 10]
            tar_color = np.array(color)[np.newaxis, np.newaxis, :]
            set_trace()
            tar_hsv = cv2.cvtColor(tar_color, cv2.COLOR_BGR2HSV)
            image_hsv[:, :, 0:2] = tar_hsv[:,:, 0:2] #* 0.8 + image_hsv[:, :, 0:2] * 0.2
        # else:
        #     image_hsv[:, :, 0:1] = tar_hsv[:, :, 0:1] #* 0.8 + tar_hsv[:, :, 0:1] * 0.2

    changed = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    for part, color in zip(parts, colors):
        if part == 12 or part == 13:
            tmp = image.astype(float)[parsing == part] * 0.5 + changed.astype(float)[parsing == part] * 0.5
            changed[parsing == part] = tmp.astype(np.uint8)
    # changed[parsing != part] = image[parsing != part]

    return changed


if __name__ == '__main__':
    # 1  face
    # 11 teeth
    # 12 upper lip
    # 13 lower lip
    # 17 hair

    args = parse_args()

    table = {
        'face' : 1,
        'hair': 17,
        'upper_lip': 12,
        'lower_lip': 13
    }

    image_path = args.img_path
    cp = 'cp/79999_iter.pth'

    image = cv2.imread(image_path)
    ori = image.copy()
    parsing = evaluate(image_path, cp)
    parsing = cv2.resize(parsing, image.shape[0:2], interpolation=cv2.INTER_NEAREST)

    parts = [table['hair'], table['upper_lip'], table['lower_lip']]

    colors = [[230, 50, 20], [20, 70, 180], [20, 70, 180]]

    #for part, color in zip(parts, colors):
    image = hair(image, parsing, parts, colors)

    cv2.imshow('image', cv2.resize(ori, (512, 512)))
    cv2.imshow('color', cv2.resize(image, (512, 512)))

    cv2.waitKey(0)
    cv2.destroyAllWindows()















