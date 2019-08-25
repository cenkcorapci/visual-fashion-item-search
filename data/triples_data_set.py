import logging
import random as pyrandom

import imgaug.augmenters as iaa
import keras
import numpy as np
from PIL import Image
from keras.preprocessing import image
from sklearn.utils import shuffle

from commons.config import DEFAULT_IMAGE_SIZE
from commons.image_utils import scale_image


class TriplesDataSet(keras.utils.Sequence):
    def __init__(self, data_set, batch_size=32, shuffle_on_end=True, do_augmentations=True):
        self._batch_size = batch_size
        self._shuffle = shuffle_on_end
        self._query_images = data_set
        self._queries = self._query_images.loc[self._query_images.name != 'query'].values
        self._query_images = self._query_images.loc[self._query_images.name == 'query']
        queries_dict = {}
        for s in self._query_images[['product', 'file']].values:
            queries_dict[s[0]] = s[1]
        self._query_images = queries_dict
        self._do_augmentations = do_augmentations

        self._aug = iaa.Sequential([
            iaa.GaussianBlur(sigma=(0.0, 1.5)),
            iaa.Fliplr(0.5),  # horizontally flip 50% of the images
            iaa.Flipud(0.5)  # vertically flip 50% of the images
        ])

        self.on_epoch_end()

    def __len__(self):
        return int(len(self._queries) / self._batch_size)

    def __getitem__(self, index, random=False):
        try:
            i = index
            if random:
                i = pyrandom.randint(0, int(self.__len__() / self._batch_size)) if random else index
            samples = self._queries[i * self._batch_size:(i + 1) * self._batch_size]
            X, y = self.__data_generation(samples)
            return X, y
        except Exception as exp:
            logging.error("Can't fetch batch #{0}, fetching a random batch.".format(index), exp)
            if not random:
                return self.__getitem__(index, random=True)  # Get a random batch if an error occurs
            else:
                raise exp

    def on_epoch_end(self):
        if self._shuffle:
            logging.info("Shuffling data set")
            self._queries = shuffle(self._queries)

    def __data_generation(self, samples):
        y = np.zeros((self._batch_size, 1))
        products = set([row['product'] for _, row in samples])
        negs = self._queries.loc[~self._queries['product'].isin(products)].sample(len(samples))[['files']]
        negs = negs.values
        X = [np.empty((self._batch_size, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3))] * 3
        for i, (_, row) in enumerate(samples.iterrows()):
            product = row['product']
            pos = self._query_images[product]
            X[0][i] = self._load_image(row['file'])
            X[1][i] = self._load_image(pos)
            X[2][i] = self._load_image(negs[i])
        return X, y

    def _load_image(self, img_path):
        f = open(img_path, 'rb')
        f = Image.open(f)
        f = scale_image(f, [DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE])
        f = np.asarray(f)
        img_data = image.img_to_array(f)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = img_data.astype('float32') / 255.
        img_data = np.clip(img_data, 0., 1.)
        img_data = img_data[0]
        if self._do_augmentations:
            img_data = self._aug.augment_image(img_data)
        return img_data
