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
DEFAULT_IMAGE_SIZE = 256
DEFAULT_VECTOR_SIZE = 512

# Local files
COMPOSITE_MODEL_PATH = '/home/twoaday/ai/models/fashion-vectorizer-converted.pth.tar'
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
MVC_GENERATED_EASY_TRIPLETS_CSV = '/run/media/twoaday/data-storag/data-sets/multi-view-clothing/triplets-easy.csv'
MVC_GENERATED_MEDIUM_TRIPLETS_CSV = '/run/media/twoaday/data-storag/data-sets/multi-view-clothing/triplets-medium.csv'
MVC_GENERATED_HARD_TRIPLETS_CSV = '/run/media/twoaday/data-storag/data-sets/multi-view-clothing/triplets-hard.csv'

WHERE2BUYIT_IMAGES_FOLDER = '/run/media/twoaday/data-storag/data-sets/where2buyit/photos/'
WHERE2BUYIT_IMAGES_LIST_FILE = '/run/media/twoaday/data-storag/data-sets/where2buyit/photos.txt'

STANFORD_PRODUCT_IMAGES_FOLDER = '/run/media/twoaday/data-storag/data-sets/stanford-products-img-similarity/Stanford_Online_Products/'

DIM_RED_MODEL_PATH = '/run/media/twoaday/data-storag/data-sets/cimri/pca_2.pkl'

CATEGORY_COUNTS_DICT = {'masterCategory': 7,
                        'subCategory': 45,
                        'articleType': 142,
                        'gender': 5,
                        'season': 4,
                        'baseColour': 46,
                        'usage': 8}

# Composite Model
GPU_ID = 0
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
TRIPLET_BATCH_SIZE = 32
EXTRACT_BATCH_SIZE = 128
TEST_BATCH_COUNT = 30
NUM_WORKERS = 4
LR = 0.001
MOMENTUM = 0.5
EPOCH = 10
DUMPED_MODEL = "model_10_final.pth.tar"

LOG_INTERVAL = 10
DUMP_INTERVAL = 500
TEST_INTERVAL = 100

DATASET_BASE = r'/DATACETNER/1/ch/deepfashion_data'
ENABLE_INSHOP_DATASET = True
INSHOP_DATASET_PRECENT = 0.8
IMG_SIZE = 256
CROP_SIZE = 224
INTER_DIM = 512
CATEGORIES = 20
N_CLUSTERS = 50
COLOR_TOP_N = 10
TRIPLET_WEIGHT = 2.0
ENABLE_TRIPLET_WITH_COSINE = False  # Buggy when backward...
COLOR_WEIGHT = 0.1
DISTANCE_METRIC = ('euclidean', 'euclidean')
FREEZE_PARAM = False

# create directories
logging.info("Checking directories...")
pathlib.Path(IMAGE_DOWNLOAD_PATH).mkdir(parents=True, exist_ok=True)
logging.info("Directories are set.")
