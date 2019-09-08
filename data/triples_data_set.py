# -*- coding: utf-8 -*-
import logging

import pandas as pd
import torch.utils.data as data
from PIL import Image
from sklearn.utils import shuffle
from torchvision import transforms
from tqdm import tqdm
import random
from ai.feature_extraction import vectorize, get_deep_color_top_n
from commons.config import BASE_MODEL_PATH
from commons.config import DEFAULT_IMAGE_SIZE
from utils.composite_model_utils import load_extractor_model


class TripletsDataSet(data.Dataset):
    def __init__(self, df, transform=True,
                 shuffle=True,
                 difficulty=.5,
                 image_size=DEFAULT_IMAGE_SIZE,
                 base_model_path=BASE_MODEL_PATH,
                 nagative_sample_count=2000):
        self._df = df
        self._shuffle = shuffle
        self._transform = transform
        self._difficulty = difficulty
        self._nagative_sample_count = nagative_sample_count
        self._data_transform = transforms.Compose([
            transforms.Resize([image_size, image_size]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-45, 45), resample=Image.BILINEAR),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
        ])
        self._extractor = load_extractor_model(base_model_path)
        self._current_vectors = []
        self.on_epoch_end()
        self._negative_df = []

    def on_epoch_end(self):
        if self._shuffle:
            self._df = shuffle(self._df)
        self._current_vectors = self._vectorize_samples(self._nagative_sample_count)

    def __len__(self):
        return len(self._df)

    def __getitem__(self, index):
        query_row = self._df.values[index]
        img = query_row[3]
        img_p = self._df.loc[self._df['product'] == query_row[1]].sample(1)['file'].values[0]
        img_n = self._get_negatives(query_row, 1)[0][0]

        img, img_p, img_n = self._process_img(img), self._process_img(img_p), self._process_img(img_n)
        return img, img_p, img_n

    def _vectorize_samples(self, num_samples=2000):
        rows = []
        for _, row in tqdm(self._df.sample(min(len(self._df), num_samples)).iterrows(),
                           desc='Vectorizing some of the samples',
                           total=min(len(self._df), num_samples)):
            try:
                deep_feat, color_feat = vectorize(self._extractor, row['file'])
                rows.append([row['name'],
                             row['product'],
                             row['category'],
                             row['file'],
                             deep_feat,
                             color_feat])
            except Exception as exp:
                logging.error('Can not vectorize {0}'.format(row['file']), exp)
        rows = pd.DataFrame(rows)
        rows.columns = ['name', 'product', 'category', 'file', 'deep_feat', 'color_feat']
        return rows

    def _get_negatives(self, query_row, n):
        rs = self._current_vectors
        rs = rs.loc[rs['category'] != query_row[2]]
        rs = rs.loc[rs['product'] != query_row[1]]
        if random.uniform(0, 1) <= self._difficulty:
            try:
                deep_feat, color_feat = vectorize(self._extractor, query_row[3])
                similar = get_deep_color_top_n(deep_feat,
                                               color_feat,
                                               rs.deep_feat.values.tolist(),
                                               rs.color_feat.values.tolist(),
                                               rs.file.values,
                                               n)
                return similar
            except Exception as exp:
                logging.error('Can not fetch close negatives, fetching random samples.. ', exp)
        return rs.sample(n)

    def _process_img(self, img_path):
        with open(img_path, 'rb') as f:
            with Image.open(f) as img:
                img = img.convert('RGB')
        if self._transform is not None:
            img = self._data_transform(img)
        return img


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from data.data_set_loaders import load_where2buy_it_data_set

    df = load_where2buy_it_data_set(500)
    generator = TripletsDataSet(df, nagative_sample_count=100)
    generator = DataLoader(generator, batch_size=1)
    for img, img_p, img_n in list(generator):
        print(img.shape)
        print(img_p.shape)
        print(img_n.shape)
