import logging
import random as pyrandom

import imgaug.augmenters as iaa
import keras
import numpy as np
from PIL import Image
from keras.preprocessing import image
from sklearn.utils import shuffle

from commons.config import MVC_IMAGES_FOLDER
from commons.image_utils import scale_image


class TriplesDataSet(keras.utils.Sequence):
    def __init__(self, data, batch_size=32, shuffle_on_end=True, do_augmentations=True, image_size=224):
        self._batch_size = batch_size
        self._shuffle = shuffle_on_end
        self._data_set = data
        self._image_size = image_size
        self._do_augmentations = do_augmentations

        self._aug = iaa.Sequential([
            iaa.Fliplr(0.5),  # horizontally flip 50% of the images
            iaa.Flipud(0.5)  # vertically flip 50% of the images
        ])

        self.on_epoch_end()

    def __len__(self):
        return int(len(self._data_set) / self._batch_size)

    def __getitem__(self, index, random=False):
        try:
            i = pyrandom.randint(0, int(self.__len__() / self._batch_size)) if random else index
            samples = self._data_set[i * self._batch_size:(i + 1) * self._batch_size]
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
            shuffle(self._data_set)

    def __data_generation(self, samples):
        X = [np.empty((self._batch_size, self._image_size, self._image_size, 3))] * 3
        y = np.zeros((self._batch_size, 1))

        for i, sample in enumerate(samples):
            for j, file_name in enumerate(sample):
                X[j][i] = self._load_image(MVC_IMAGES_FOLDER + file_name)
        return X, y

    def _load_image(self, img_path):
        f = open(img_path, 'rb')
        f = Image.open(f)
        f = scale_image(f, [self._image_size, self._image_size])
        f = np.asarray(f)
        img_data = image.img_to_array(f)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = img_data.astype('float32') / 255.
        img_data = np.clip(img_data, 0., 1.)
        img_data = img_data[0]
        if self._do_augmentations:
            img_data = self._aug.augment_image(img_data)
        return img_data
