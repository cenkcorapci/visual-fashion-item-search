# -*- coding: utf-8 -*-
"""Model configs.
"""

import logging
import pathlib

# Logs
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
LOGS_PATH = '/tmp/tb_logs/'

# Experiments
RANDOM_STATE = 41
DEFAULT_IMAGE_SIZE = 512

# Local files
TEMP_PATH = '/tmp'
IMAGE_DOWNLOAD_PATH = '/run/media/twoaday/data-storag/data-sets/cimri/images/'
CIMRI_CSV = '/run/media/twoaday/data-storag/data-sets/cimri/match_core_data_set_with_images.csv'

PRODUCT_IMAGES_PATH = '/run/media/twoaday/data-storag/data-sets/fashion-product-images-small/images/'
PRODUCT_IMAGES_CSV_PATH = '/run/media/twoaday/data-storag/data-sets/fashion-product-images-small/styles.csv'

MVC_BASE_PATH = '/run/media/twoaday/data-storag/data-sets/multi-view-clothing/'
MVC_ATTRIBUTES_PATH = '/run/media/twoaday/data-storag/data-sets/multi-view-clothing/attribute_labels.json'
MVC_IMAGE_LINKS_PATH = '/run/media/twoaday/data-storag/data-sets/multi-view-clothing/image_links.json'
MVC_INFO_PATH = '/run/media/twoaday/data-storag/data-sets/multi-view-clothing/mvc_info.json'
MVC_IMAGES_FOLDER = '/run/media/twoaday/data-storag/data-sets/multi-view-clothing/images/'

DIM_RED_MODEL_PATH = '/run/media/twoaday/data-storag/data-sets/cimri/pca_2.pkl'

CATEGORY_COUNTS_DICT = {'masterCategory': 7,
                        'subCategory': 45,
                        'articleType': 142,
                        'gender': 5,
                        'season': 4,
                        'baseColour': 46,
                        'usage': 8}

# create directories
logging.info("Checking directories...")
pathlib.Path(IMAGE_DOWNLOAD_PATH).mkdir(parents=True, exist_ok=True)
logging.info("Directories are set.")
