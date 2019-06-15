import concurrent.futures
import logging

import pandas as pd
from tqdm import tqdm

from commons.atomic_counters import AtomicProgressBar
from commons.config import CIMRI_CSV
from data.utils import download_image

# Get data
df = pd.read_csv(CIMRI_CSV, error_bad_lines=False)
df = df.drop(columns=['Unnamed: 0'])
df = df.loc[df.isSame == 1]
df = df.drop(columns=['offerId', 'isSame', 'secondProduct'])
df = df.dropna()
fashion_categories = df.loc[df.firstProduct.str.contains(
    '(gozluk|takim|kolye|kupe|bilezik|bere|atki|sapka|pijama|corap|cizme|kemer|kulaklik|eldiven|sal|sort|mont|bandana|polar|fular|esarp|yelek|gomlek|sweatshirt|ceket|pardesu|palto|saat|ayakkabi|t-shirt|elbise|canta|bluz|pantalon|jean|etek|kolye)',
    regex=True)]
fashion_categories = fashion_categories.categoryIdOfFirst.unique()
df = df.loc[df.categoryIdOfFirst.isin(fashion_categories)]

image_ids = []
for _, row in tqdm(df.iterrows(), total=len(df), desc='Gathering image ids'):
    try:
        s = []
        [s.append(img_id) for img_id in row['productImages'].split('-')]
        [s.append(img_id) for img_id in row['offerImages'].split('-')]
        s = set(s)
        [image_ids.append(img_id) for img_id in s]
    except Exception as exp:
        logging.error(exp)

p_bar = AtomicProgressBar(total=len(image_ids), desc='Downloading images')


def download_with_log(img_id):
    try:
        download_image(img_id)
    except Exception as exp:
        logging.error(exp)
    finally:
        p_bar.increment()


with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
    _ = executor.map(download_with_log, set(image_ids))
