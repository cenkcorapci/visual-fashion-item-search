from __future__ import print_function

import logging
import os

import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image

from commons.config import DEFAULT_IMAGE_SIZE, RANDOM_STATE, IMAGE_DOWNLOAD_PATH
from data.utils import download_image, delete_image


class ImageClassificationModel:
    vgg16 = 'VGG16'
    vgg19 = 'VGG19'


class ImageVectorizationModel:
    def __init__(self, model_name):
        self._model_name = model_name
        if model_name == ImageClassificationModel.vgg16:
            self._model = VGG16(weights='imagenet', include_top=False,
                                input_shape=(DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3))
        elif model_name == ImageClassificationModel.vgg19:
            self._model = VGG19(weights='imagenet', include_top=False,
                                input_shape=(DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3))
        else:
            raise Exception('Unknown model selection')
        np.random.seed(RANDOM_STATE)
        self._graph = tf.get_default_graph()

    def _get_image(self, image_id):
        file_path = IMAGE_DOWNLOAD_PATH + "{0}.jpg".format(image_id)
        if not os.path.exists(file_path):
            download_image(image_id, DEFAULT_IMAGE_SIZE)
        img = mpimg.imread(file_path)
        return img

    def get_feature_vectors(self, img_id):
        def vectorize():
            with self._graph.as_default():
                img = self._get_image(img_id)
                img_data = image.img_to_array(img)
                img_data = np.expand_dims(img_data, axis=0)
                img_data = preprocess_input(img_data)
                features = self._model.predict(img_data)

                return features.flatten()

        try:
            if self._model_name == ImageClassificationModel.vgg16:
                from keras.applications.vgg16 import preprocess_input
                return vectorize()
            elif self._model_name == ImageClassificationModel.vgg19:
                from keras.applications.vgg19 import preprocess_input
                return vectorize()
            else:
                raise Exception('Unknown model selection')

        except Exception as exp:
            logging.error("Can't get feature vector for image {0}".format(exp))
            raise exp
