import glob
import itertools
import logging

import pandas as pd
from colorama import Fore
from tqdm import tqdm

from commons.config import MVC_INFO_PATH, MVC_IMAGES_FOLDER


def generate_triplets(easy=False):
    data_set = []
    df_mvc_info = pd.read_json(MVC_INFO_PATH)
    product_ids = df_mvc_info.productId.unique().tolist()
    logging.info('Found {0} products'.format(len(product_ids)))
    available_image_list = glob.glob("{0}*.jpg".format(MVC_IMAGES_FOLDER))
    available_image_list = [x.split('/')[-1] for x in tqdm(available_image_list,
                                                           desc='Parsing available image files',
                                                           bar_format="{l_bar}%s{bar}%s{r_bar}" % (
                                                               Fore.YELLOW, Fore.RESET))]
    query_cache = {}
    df_mvc_sub_set = df_mvc_info.sample(5000)
    if not easy:
        for sub_cat in tqdm(df_mvc_info.subCategory2.unique().tolist()):
            for gender in tqdm(df_mvc_info.productGender.unique().tolist()):
                query_cache[(sub_cat, gender)] = df_mvc_info.loc[df_mvc_info.subCategory2 == sub_cat].loc[
                    df_mvc_info.productGender == gender][['productId', 'colourId', 'image_url_2x']]

    for product_id in tqdm(product_ids, desc='Generating {0} image triplets'.format('easy' if easy else 'hard'),
                           bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.MAGENTA, Fore.RESET)):
        try:
            # Every item can have multiple different colour variations, i consider them as different products
            df_product = df_mvc_info.loc[df_mvc_info.productId == product_id]
            colour_variations = df_product.colourId.unique().tolist()
            for colour in colour_variations:
                # Find products images
                positives = df_product.loc[df_product.colourId == colour].image_url_2x.tolist()

                # Get similar but different images
                sub_cat = df_product.subCategory2.tolist()[0]
                df_neg = None
                if easy:
                    df_neg = df_mvc_sub_set.loc[df_mvc_sub_set.productId != product_id]
                else:
                    gender = df_product.productGender.tolist()[0]
                    df_neg = query_cache[(sub_cat, gender)]
                    df_neg = df_neg.loc[df_neg.colourId == colour]
                    df_neg = df_neg.loc[df_neg.productId != product_id]

                # Generate pairs from products images
                pairs = list(itertools.combinations(positives, 2))

                if not easy:
                    # Add similar but different images to pairs as negative examples
                    df_neg = df_neg.sample(min(len(pairs), len(df_neg))).image_url_2x.tolist()
                else:
                    add = max(len(pairs) - len(df_neg), int(len(pairs) / 2))
                    add_df = df_mvc_sub_set.loc[df_mvc_info.productId != product_id].sample(add)
                    df_neg = df_neg.sample(min(len(df_neg), int(len(pairs) / 2))).image_url_2x.tolist()
                    df_neg += add_df.image_url_2x.tolist()

                if len(pairs) > len(df_neg):
                    add_df = df_mvc_info.loc[df_mvc_info.productId != product_id].loc[
                        df_mvc_info.subCategory2 == sub_cat]
                    add_df = add_df.sample(len(pairs) - len(df_neg)).image_url_2x.tolist()
                    df_neg += add_df

                # Generate triplets
                for pair, negative in zip(pairs, df_neg):
                    anchor, positive = pair
                    sample = [anchor.split('/')[-1], positive.split('/')[-1], negative.split('/')[-1]]
                    sample = [img for img in sample if img in available_image_list]
                    if len(sample) == 3:
                        data_set.append(sample)
        except Exception as exp:
            logging.error('Can not generate triplet', exp)
    return data_set
