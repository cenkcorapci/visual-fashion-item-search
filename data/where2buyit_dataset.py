# -*- coding: utf-8 -*-

import numpy as np
import torch.utils.data as data
from PIL import Image
import pandas as pd


class Where2BuyItDataset(data.Dataset):
    def __init__(self, df, transforms=None):
        self._df = df[['product', 'file']]
        self._df['product'] = pd.factorize(df['product'])[0]
        self._transforms = transforms

    def __len__(self):
        return len(self._df)

    def __getitem__(self, index):
        sample = self._df.values[index]
        label, img = sample[0], sample[1]
        return self._process_img(img), np.array(label)

    def targets(self):
        return self._df['product'].values

    def _process_img(self, img_path):
        with open(img_path, 'rb') as f:
            with Image.open(f) as img:
                img = img.convert('RGB')
                if self._transforms is not None:
                    img = self._transforms(img)
        return img
