import glob
import os

import pandas as pd
import requests
from tqdm import tqdm

from commons.config import DEFAULT_IMAGE_SIZE, IMAGE_DOWNLOAD_PATH
from commons.config import WHERE2BUYIT_IMAGES_FOLDER


def download_cimri_image(image_id, size=DEFAULT_IMAGE_SIZE):
    file_url = "https://cdn.cimri.io/image/{0}x{0}/asdf_{1}.jpg".format(size, image_id)
    r = requests.get(file_url)
    save_path = IMAGE_DOWNLOAD_PATH + "{0}.jpg".format(image_id)
    with open(save_path, 'wb') as f:
        f.write(r.content)


def download_image(image_url, download_path, image_name=None):
    name = image_url.split('/')[-1] if image_name is None else image_name
    r = requests.get(image_url)
    save_path = download_path + name
    with open(save_path, 'wb') as f:
        f.write(r.content)


def delete_image(image_id):
    os.remove(IMAGE_DOWNLOAD_PATH + "{0}.jpg".format(image_id))


def load_data_set():
    result = [y for x in os.walk(WHERE2BUYIT_IMAGES_FOLDER) for y in glob.glob(os.path.join(x[0], '*.jpg'))]
    df_data_set = []
    for file in tqdm(result, desc='Parsing files'):
        s = file.split('/')
        name, product, category = s[-1].replace('.jpg', ''), s[-2], s[-3]
        df_data_set.append([name, product, category, file])
    df_data_set = pd.DataFrame(df_data_set)
    df_data_set.columns = ['name', 'product', 'category', 'file']
    return df_data_set
