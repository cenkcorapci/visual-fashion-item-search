import concurrent.futures
import glob
import logging
import pathlib
import shutil
from io import BytesIO

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

from commons.atomic_counters import AtomicProgressBar
from commons.config import WHERE2BUYIT_IMAGES_FOLDER, WHERE2BUYIT_IMAGES_LIST_FILE

# Get data
photos_df = WHERE2BUYIT_IMAGES_LIST_FILE
photos_df = pd.read_csv(photos_df, error_bad_lines=False, header=None)
photos_df.columns = ['photo_id', 'url']

pair_file_list = glob.glob("/run/media/twoaday/data-storag/data-sets/where2buyit/meta/json/*.json")
pair_file_list = [p for p in pair_file_list if '_pairs_' in p]
pair_file_list = [p for p in pair_file_list if 'train_' in p]

retrieval_file_list = glob.glob("/run/media/twoaday/data-storag/data-sets/where2buyit/meta/json/*.json")
retrieval_file_list = [p for p in retrieval_file_list if 'retrieval_' in p]

for pair_file in tqdm(pair_file_list[1:], desc='Getting pairs for each category'):
    category_name = pair_file.split('pairs_')[1].replace('.json', '')
    df_pairs = pd.read_json(pair_file).shuffle()
    df_retrieval = [f for f in retrieval_file_list if category_name in f][0]
    df_retrieval = pd.read_json(df_retrieval)
    p_bar = AtomicProgressBar(total=len(df_pairs), desc='Downloading images of {0}'.format(category_name))


    def download_with_log(row):
        path = None
        try:
            # Download query image
            bbox = row['bbox']
            if bbox['width'] < 224 or bbox['height'] < 224:
                logging.warn('Crop area is too small, passing the sample')
                return

            path = '{0}/{1}/{2}/'.format(WHERE2BUYIT_IMAGES_FOLDER, category_name, row['product'])
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
            img = requests.get(photos_df.loc[photos_df.photo_id == row['photo']].url.values[0]).content
            img = Image.open(BytesIO(img))

            bbox = (bbox['left'], bbox['top'], bbox['left'] + bbox['width'], bbox['top'] + bbox['height'])
            crop = img.crop(bbox)
            crop.save(path + 'query.jpg', 'JPEG')
            # Download result images
            result_images = df_retrieval.loc[df_retrieval['product'] == row['product']].photo.values
            for img_id in result_images[:min(3, len(result_images))]:
                img_name = '{0}.jpg'.format(img_id)
                img = requests.get(photos_df.loc[photos_df.photo_id == img_id].url.values[0]).content
                img = Image.open(BytesIO(img))
                img.save(path + img_name, 'JPEG')

        except Exception as exp:
            logging.error('Can not download, cleaning sample directory...', exp)
            if path is not None:
                try:
                    shutil.rmtree(path)
                except Exception as exp:
                    logging.error('Can not clean up', exp)
        finally:
            p_bar.increment()


    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        _ = executor.map(download_with_log,
                         [row for _, row in df_pairs.iterrows()])
