import concurrent.futures
import logging

import numpy as np
import pandas as pd
from PIL import Image
from colorama import Fore
from keras.preprocessing import image
from tqdm import tqdm

from commons.config import DEFAULT_IMAGE_SIZE
from commons.config import MVC_GENERATED_TRIPLETS_CSV, MVC_IMAGES_FOLDER


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


def get_checked_images():
    '''
    Tool for omitting unreadable image files from generated triplets
    :return:
    '''
    data_set = pd.read_csv(MVC_GENERATED_TRIPLETS_CSV)
    data_set = [[row['anchor'], row['positive'], row['negative']] for _, row in
                tqdm(data_set.iterrows(), desc='Loading triplets',
                     bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.MAGENTA, Fore.RESET),
                     total=len(data_set))]
    new_set = []
    for sample in tqdm(data_set, desc='Checking images',
                       bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.YELLOW, Fore.RESET), ):
        with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
            s = list(executor.map(is_image_intact, [MVC_IMAGES_FOLDER + img for img in sample]))
            if len([x for x in s if x]) == 3:
                new_set.append(sample)
            else:
                logging.warn('{0} is not readable'.format(sample))

    pd.DataFrame(new_set, columns=['anchor', 'positive', 'negative']).to_csv(MVC_GENERATED_TRIPLETS_CSV, index=False)
    return new_set
