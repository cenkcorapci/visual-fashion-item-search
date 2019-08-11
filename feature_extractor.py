import numpy as np
from torch import nn

from commons.config import COLOR_TOP_N


class FeatureExtractor(nn.Module):
    def __init__(self, deep_module, color_module, pooling_module):
        super(FeatureExtractor, self).__init__()
        self.deep_module = deep_module
        self.color_module = color_module
        self.pooling_module = pooling_module
        self.deep_module.eval()
        self.color_module.eval()
        self.pooling_module.eval()

    def forward(self, x):
        cls, feat, conv_out = self.deep_module(x)
        color = self.color_module(x).cpu().data.numpy()  # N * C * 7 * 7
        weight = self.pooling_module(conv_out).cpu().data.numpy()  # N * 1 * 7 * 7
        result = []
        for i in range(cls.size(0)):
            weight_n = weight[i].reshape(-1)
            idx = np.argpartition(weight_n, -COLOR_TOP_N)[-COLOR_TOP_N:][::-1]
            color_n = color[i].reshape(color.shape[1], -1)
            color_selected = color_n[:, idx].reshape(-1)
            result.append(color_selected)
        return feat.cpu().data.numpy(), result
