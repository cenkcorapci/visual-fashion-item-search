import argparse
import glob
import itertools
import logging
import os
import pathlib
import random

import pandas as pd
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ai.callbacks import step_decay_schedule
from ai.models.similarity_model import ImageSimilarityNetwork
from commons.config import MVC_INFO_PATH, MVC_BASE_PATH, MVC_IMAGES_FOLDER, MVC_GENERATED_TRIPLETS_CSV
from commons.image_utils import get_checked_images
from data.triples_data_set import TriplesDataSet

MODEL_NAME = 'xception_similarity'

TB_LOG_DIRECTORY = MVC_BASE_PATH + 'tb_logs/' + MODEL_NAME + '/'
MODEL_PATH = MVC_BASE_PATH + 'models/'

pathlib.Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)
pathlib.Path(TB_LOG_DIRECTORY).mkdir(parents=True, exist_ok=True)

# Get Parameters ---------------------------------------------
usage_docs = """
--epochs <integer> Number of epochs
--val_split <float> Set validation split(between 0 and 1)
--batch_size <int> Batch size for training
--sample_size <int> How many products will be used in training
"""

parser = argparse.ArgumentParser(usage=usage_docs)

parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--val_split', type=float, default=.1)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--sample_size', type=int, default=None)

args = parser.parse_args()

# Generate data set ------------------------------------------
data_set = []
if os.path.exists(MVC_GENERATED_TRIPLETS_CSV):
    data_set = pd.read_csv(MVC_GENERATED_TRIPLETS_CSV)
    if args.sample_size is not None:
        data_set = random.sample(data_set, args.sample_size)

else:
    df_mvc_info = pd.read_json(MVC_INFO_PATH)
    product_ids = df_mvc_info.productId.unique().tolist()
    if args.sample_size is not None:
        product_ids = random.sample(product_ids, args.sample_size)
    logging.info('Found {0} products'.format(len(product_ids)))
    available_image_list = glob.glob("{0}*.jpg".format(MVC_IMAGES_FOLDER))
    available_image_list = [x.split('/')[-1] for x in tqdm(available_image_list, desc='Parsing available image files')]

    query_cache = {}
    for sub_cat in tqdm(df_mvc_info.subCategory2.unique().tolist()):
        for gender in tqdm(df_mvc_info.productGender.unique().tolist()):
            query_cache[(sub_cat, gender)] = df_mvc_info.loc[df_mvc_info.subCategory2 == sub_cat].loc[
                df_mvc_info.productGender == gender][['productId', 'colourId', 'image_url_2x']]

    for product_id in tqdm(product_ids, desc='Generating image triplets'):
        try:
            # Every item can have multiple different colour variations, we consider them different
            df_product = df_mvc_info.loc[df_mvc_info.productId == product_id]
            colour_variations = df_product.colourId.unique().tolist()
            for colour in colour_variations:
                # Find products images
                positives = df_product.loc[df_product.colourId == colour].image_url_2x.tolist()

                # Get similar but different images
                sub_cat = df_product.subCategory2.tolist()[0]
                gender = df_product.productGender.tolist()[0]
                df_neg = query_cache[(sub_cat, gender)]
                df_neg = df_neg.loc[df_neg.colourId == colour]
                df_neg = df_neg.loc[df_neg.productId != product_id]

                # Generate pairs from products images
                pairs = list(itertools.combinations(positives, 2))

                # Add similar but different images to pairs as negative examples
                df_neg = df_neg.sample(min(len(pairs), len(df_neg))).image_url_2x.tolist()
                if len(pairs) > len(df_neg):
                    cat = df_mvc_info.loc[df_mvc_info.productId == product_id].subCategory2.tolist()[0]
                    add_df = df_mvc_info.loc[df_mvc_info.productId != product_id].loc[df_mvc_info.subCategory2 == cat]
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

    pd.DataFrame(data_set, columns=['anchor', 'positive', 'negative']).to_csv(MVC_GENERATED_TRIPLETS_CSV, index=False)
logging.info('Usinga a data set with {0} triplets, example triplet: {1}'.format(len(data_set), data_set[0]))

train_data_set, validation_data_set = train_test_split(data_set, test_size=args.val_split)
train_data_set = TriplesDataSet(train_data_set, args.batch_size)
validation_data_set = TriplesDataSet(validation_data_set,
                                     batch_size=args.batch_size,
                                     shuffle_on_end=False,
                                     do_augmentations=False)

# Prepare model -------------------------------------------

tb_callback = TensorBoard(log_dir=TB_LOG_DIRECTORY, histogram_freq=0, write_graph=True,
                          write_images=False,
                          embeddings_freq=0,
                          embeddings_metadata=None)
es_callback = EarlyStopping(patience=3, monitor='val_loss')
checkpoint_callback = ModelCheckpoint(MODEL_PATH + MODEL_NAME + '.{epoch:02d}-{val_loss:.2f}.hdf5',
                                      monitor='val_loss',
                                      verbose=1,
                                      save_best_only=False,
                                      save_weights_only=False)
lr_scheduler = step_decay_schedule(initial_lr=.0002, decay_factor=.1, step_size=2)
callbacks = [tb_callback, es_callback, checkpoint_callback, lr_scheduler]

model = ImageSimilarityNetwork().generate_network()

# Train ----------------------------------------------------
model.fit_generator(
    generator=train_data_set,
    callbacks=callbacks,
    validation_data=validation_data_set,
    epochs=args.epochs,
    use_multiprocessing=True,
    workers=8,
    max_queue_size=8,
    verbose=1)
