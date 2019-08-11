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

# Local files
TEMP_PATH = '/tmp'

PRODUCT_IMAGES_PATH = '/run/media/twoaday/data-storag/data-sets/fashion-product-images-small/images/'
PRODUCT_IMAGES_CSV_PATH = '/run/media/twoaday/data-storag/data-sets/fashion-product-images-small/styles.csv'

MVC_BASE_PATH = '/run/media/twoaday/data-storag/data-sets/multi-view-clothing/'
MVC_ATTRIBUTES_PATH = '/run/media/twoaday/data-storag/data-sets/multi-view-clothing/attribute_labels.json'
MVC_IMAGE_LINKS_PATH = '/run/media/twoaday/data-storag/data-sets/multi-view-clothing/image_links.json'
MVC_INFO_PATH = '/run/media/twoaday/data-storag/data-sets/multi-view-clothing/mvc_info.json'
MVC_IMAGES_FOLDER = '/run/media/twoaday/data-storag/data-sets/multi-view-clothing/images/'

DUMPED_MODEL = "/home/twoaday/ai/models/fashion-vectorizer.pth.tar"

# Deep Learning
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

LOG_INTERVAL = 10
DUMP_INTERVAL = 500
TEST_INTERVAL = 100

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
