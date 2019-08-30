import numpy as np
import torch
import torch.utils.data as data
from scipy.spatial.distance import cdist
from torch.autograd import Variable

from data.single_sample_data_set import SingleSampleDataSet


def vectorize(extractor, img):
    """
    Extracts deep and color features from given image

    :param extractor: Feature extractor model
    :param img: Image path
    :return: a numpy lsit of deep features and a numpy list of color features
    """
    single_loader = torch.utils.data.DataLoader(SingleSampleDataSet(img), batch_size=1)
    data = list(single_loader)[0]
    data = Variable(data).cuda()
    deep_feat, color_feat = extractor(data)
    deep_feat = deep_feat[0].squeeze()
    color_feat = color_feat[0]
    return deep_feat, color_feat


def get_top_n(dist, labels, retrieval_top_n):
    ind = np.argpartition(dist, -retrieval_top_n)[-retrieval_top_n:][::-1]
    ret = list(zip([labels[i] for i in ind], dist[ind]))
    ret = sorted(ret, key=lambda x: x[1], reverse=True)
    return ret


def get_similarity(feature, feats, metric='cosine'):
    dist = -cdist(np.expand_dims(feature, axis=0), feats, metric)[0]
    return dist


def get_deep_color_top_n(deep_features, color_features, deep_feats_set, color_feats_set, labels, retrieval_top_n=5):
    deep_scores = get_similarity(deep_features, deep_feats_set)
    color_scores = get_similarity(color_features, color_feats_set, 'euclidean')
    results = get_top_n(deep_scores + color_scores * .1, labels, retrieval_top_n)
    return results

