import argparse
import logging
import pathlib
import random

import pandas as pd
from colorama import Fore
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ai.callbacks import step_decay_schedule
from ai.models.similarity_model import ImageSimilarityNetwork
from commons.config import MVC_BASE_PATH, MVC_GENERATED_EASY_TRIPLETS_CSV
from data.triples_data_set import TriplesDataSet

MODEL_NAME = 'inception_similarity'

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
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--sample_size', type=int, default=None)

args = parser.parse_args()

# Generate data set ------------------------------------------
data_set = pd.read_csv(MVC_GENERATED_EASY_TRIPLETS_CSV)

data_set = [[row['anchor'], row['positive'], row['negative']] for _, row in
            tqdm(data_set.iterrows(), desc='Loading triplets',
                 bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.MAGENTA, Fore.RESET),
                 total=len(data_set))]

if args.sample_size is not None:
    data_set = random.sample(data_set, args.sample_size)

logging.info('Using a data set with {0} triplets, example triplet: {1}'.format(len(data_set), data_set[0]))

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
    workers=4,
    max_queue_size=32,
    verbose=1)
