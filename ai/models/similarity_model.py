from keras.applications.xception import Xception
from keras.layers import Dense, Dropout, GlobalMaxPooling2D
from keras.layers import Input, concatenate
from keras.models import Model
from keras.optimizers import Adam

from ai.loss_functions import triplet_loss

from commons.config import DEFAULT_IMAGE_SIZE


class ImageSimilarityNetwork:
    def __init__(self, dimensions=[DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3]):
        self._dimensions = dimensions
        self._optimization = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)

    def _create_base_network(self):
        """
        Base network to be shared.
        """
        base_model = Xception(input_shape=(self._dimensions[0], self._dimensions[1], self._dimensions[2]),
                              weights="imagenet", include_top=False)
        for layer in base_model.layers:
            layer.trainable = True
        x = base_model.output

        # added layers
        x = GlobalMaxPooling2D()(x)
        x = Dropout(0.5)(x)
        vector = Dense(2048, activation='selu', kernel_initializer='lecun_normal')(x)

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=vector)

        return model

    def generate_network(self):
        h, w, c = self._dimensions[0], self._dimensions[1], self._dimensions[2]
        anchor_input = Input((h, w, c,), name='anchor_input')
        positive_input = Input((h, w, c,), name='positive_input')
        negative_input = Input((h, w, c,), name='negative_input')

        # Shared embedding layer for positive and negative items
        shared_network = self._create_base_network()

        encoded_anchor = shared_network(anchor_input)
        encoded_positive = shared_network(positive_input)
        encoded_negative = shared_network(negative_input)

        merged_vector = concatenate([encoded_anchor, encoded_positive, encoded_negative], axis=-1, name='merged_layer')

        model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=merged_vector)
        model.compile(loss=triplet_loss, optimizer=self._optimization)
        return model