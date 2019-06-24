import keras.backend as K

from commons.config import DEFAULT_VECTOR_SIZE


def triplet_loss(y_true, y_pred, alpha=0.4):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """

    total_length = y_pred.shape.as_list()[-1]

    anchor = y_pred[:, 0:int(total_length * 1 / 3)]
    positive = y_pred[:, int(total_length * 1 / 3):int(total_length * 2 / 3)]
    negative = y_pred[:, int(total_length * 2 / 3):int(total_length * 3 / 3)]

    # distance between the anchor and the positive
    pos_dist = K.sum(K.sqrt(anchor - positive), axis=1)

    # distance between the anchor and the negative
    neg_dist = K.sum(K.sqrt(anchor - negative), axis=1)

    # compute loss
    basic_loss = pos_dist - neg_dist + alpha
    loss = K.maximum(basic_loss, 0.0)

    return loss


def lossless_triplet_loss(y_true, y_pred, N=DEFAULT_VECTOR_SIZE, beta=DEFAULT_VECTOR_SIZE, epsilon=1e-8):
    """
    Implementation of the triplet loss function
    Use it with sigmoid activation

    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    N  --  The number of dimension
    beta -- The scaling factor, N is recommended
    epsilon -- The Epsilon value to prevent ln(0)


    Returns:
    loss -- real number, value of the loss
    """

    total_length = y_pred.shape.as_list()[-1]

    anchor = y_pred[:, 0:int(total_length * 1 / 3)]
    positive = y_pred[:, int(total_length * 1 / 3):int(total_length * 2 / 3)]
    negative = y_pred[:, int(total_length * 2 / 3):int(total_length * 3 / 3)]

    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor - positive), axis=1)

    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor - negative), axis=1)

    # Non Linear Values

    # -ln(-x/N+1)
    pos_dist = -K.log(( pos_dist / beta) + 1 + epsilon)
    neg_dist = -K.log(((N - neg_dist) / beta) + 1 + epsilon)

    # compute loss
    loss = neg_dist + pos_dist

    return loss
