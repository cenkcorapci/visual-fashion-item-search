import concurrent.futures
import logging

import pandas as pd

from commons.atomic_counters import AtomicProgressBar, AtomicCounter
from commons.config import MVC_INFO_PATH, MVC_IMAGES_FOLDER
from data.utils import download_image

# Get data
image_links = pd.read_json(MVC_INFO_PATH)
image_links = image_links.image_url_2x.tolist()
p_bar = AtomicProgressBar(total=len(image_links), desc='Downloading images')

counter = AtomicCounter()


def download_with_log(img_url):
    try:
        download_image(img_url, MVC_IMAGES_FOLDER)
        counter.increment()
    except Exception as exp:
        logging.error('Can not download, will try again', exp)
        try:
            download_image(img_url, MVC_IMAGES_FOLDER)
            counter.increment()
        except Exception as exp:
            logging.error('Yep, can not download...', exp)
    finally:
        p_bar.increment()


with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
    _ = executor.map(download_with_log, set(image_links))

logging.info('Downloaded {0} images successfully.'.format(counter.value))
