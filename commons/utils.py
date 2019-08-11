# -*- coding: utf-8 -*-

import os

import torch

from commons.config import *


def load_model(path=None):
    if not path:
        return None
    full = os.path.join(path)
    for i in [path, full]:
        if os.path.isfile(i):
            return torch.load(i)
    return None

