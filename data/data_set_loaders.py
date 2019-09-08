import glob
import logging
import os

import pandas as pd
from tqdm import tqdm

from commons.config import WHERE2BUYIT_IMAGES_FOLDER, STANFORD_PRODUCT_IMAGES_FOLDER, MVC_IMAGES_FOLDER, MVC_INFO_PATH


def load_where2buy_it_data_set():
    logging.info('Loading where2buy it data set')
    result = [y for x in os.walk(WHERE2BUYIT_IMAGES_FOLDER) for y in glob.glob(os.path.join(x[0], '*.jpg'))]
    df_data_set = []
    for file in tqdm(result, desc='Parsing files'):
        s = file.split('/')
        name, product, category = s[-1].replace('.jpg', ''), 'w2bt_' + s[-2], s[-3]
        df_data_set.append([name, product, category, file])
    df_data_set = pd.DataFrame(df_data_set)
    df_data_set.columns = ['name', 'product', 'category', 'file']
    return df_data_set


def load_stanford_product_images_data_set():
    logging.info('Loading stanford product images data set')
    result = [y for x in os.walk(STANFORD_PRODUCT_IMAGES_FOLDER) for y in glob.glob(os.path.join(x[0], '*.JPG'))]

    df_data_set = []
    for file in tqdm(result, desc='Parsing files'):
        s = file.split('/')
        name, product = s[-1].replace('.jpg', ''), 'spd_' + s[-1].split('_')[0]
        category = s[-2].replace('_final', '')
        df_data_set.append([name, product, category, file])
    df_data_set = pd.DataFrame(df_data_set)
    df_data_set.columns = ['name', 'product', 'category', 'file']
    return df_data_set


def load_mvc_data_set():
    logging.info('Loading mvc data set')
    df_data_set = []
    df_mvc_info = pd.read_json(MVC_INFO_PATH)
    product_ids = df_mvc_info.productId.unique().tolist()
    logging.info('Found {0} products'.format(len(product_ids)))
    available_image_list = glob.glob("{0}*.jpg".format(MVC_IMAGES_FOLDER))
    available_image_list = [x.split('/')[-1] for x in tqdm(available_image_list,
                                                           desc='Parsing available image files')]
    available_image_list = set(available_image_list)
    for _, row in tqdm(df_mvc_info.iterrows(), desc='Parsing files', total=len(df_mvc_info)):
        file_name = row['image_url_2x'].split('/')[-1]
        if file_name in available_image_list:
            category = row['category'].replace('"', '').lower()
            category = 'underwear' if category == 'underwear & intimates' else category
            category = 'outerwear' if category == 'coats & outerwear' else category
            df_data_set.append([file_name,
                                'mvc_' + row['productName'],
                                category,
                                MVC_IMAGES_FOLDER + file_name])

    df_data_set = pd.DataFrame(df_data_set)
    df_data_set.columns = ['name', 'product', 'category', 'file']
    return df_data_set
