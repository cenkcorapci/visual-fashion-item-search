import glob
import logging
import os

import numpy as np
from PIL import Image
from colorama import Fore
from keras.preprocessing import image
from tqdm import tqdm

from commons.config import DEFAULT_IMAGE_SIZE
from commons.config import MVC_IMAGES_FOLDER


def scale_image(image, max_size, method=Image.ANTIALIAS):
    """
    resize 'image' to 'max_size' keeping the aspect ratio
    and place it in center of white 'max_size' image
    """
    image.thumbnail(max_size, method)
    offset = (int((max_size[0] - image.size[0]) / 2), int((max_size[1] - image.size[1]) / 2))
    back = Image.new("RGB", max_size, "white")
    back.paste(image, offset)

    return back


def is_image_intact(img_path):
    try:
        f = open(img_path, 'rb')
        f = Image.open(f)
        f = scale_image(f, [DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE])
        f = np.asarray(f)
        img_data = image.img_to_array(f)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = img_data.astype('float32') / 255.
        img_data = np.clip(img_data, 0., 1.)
        _ = img_data[0]
        return True
    except Exception as exp:
        logging.error(exp)
        return False


def delete_corrupt_images():
    '''
    Tool for omitting unreadable image files from generated triplets
    :return:
    '''
    deleted = 0
    available_image_list = glob.glob("{0}*.jpg".format(MVC_IMAGES_FOLDER))
    for img in tqdm(available_image_list, desc='Checking images',
                    bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.MAGENTA, Fore.RESET), ):
        if not is_image_intact(img):
            try:
                os.remove(img)
                deleted += 1
            except Exception as exp:
                logging.error('Can not delete {0}'.format(img), exp)
    logging.info('Deleted {0} images.'.format(deleted))
