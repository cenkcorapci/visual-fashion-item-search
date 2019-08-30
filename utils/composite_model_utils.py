# -*- coding: utf-8 -*-

import torch

from ai.feature_extractor import FeatureExtractor
from ai.models.composite_model import f_model, c_model, p_model
from commons.config import COMPOSITE_MODEL_PATH


def load_extractor_model(model_path):
    main_model = f_model(model_path=model_path).cuda()
    color_model = c_model().cuda()
    pooling_model = p_model().cuda()
    extractor = FeatureExtractor(main_model, color_model, pooling_model)
    return extractor


def load_model(path=COMPOSITE_MODEL_PATH):
    return torch.load(path)
