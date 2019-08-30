import argparse
from pymongo import MongoClient

from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from PIL import ImageOps
import torch.utils.data as data
from keras.preprocessing import image
from tqdm import tqdm
from io import BytesIO
from torchvision import transforms
from torch.autograd import Variable

from sklearn.cluster import KMeans
from sklearn.externals import joblib
import os
from sklearn.decomposition import PCA

from utils.image_utils import scale_image
import os
import glob
import requests
import numpy as np
import pandas as pd
import torch
from utils.composite_model_utils import load_extractor_model
from ai.feature_extraction import vectorize
import logging

parser = argparse.ArgumentParser(description='Create a searchable product image database')
parser.add_argument('--data_path',
                    type=str, default='/run/media/twoaday/data-storag/data-sets/where2buyit/photos',
                    help='Data set root path')
parser.add_argument('--model_path',
                    type=str, default='/home/twoaday/ai/models/fashion-vectorizer-converted.pth.tar',
                    help='Feature extraction model path')
parser.add_argument('--num_clusters',
                    type=int, default=50,
                    help='How many clusters will the data be divided to')
parser.add_argument('--knn_df_model_path',
                    type=str, default='/home/twoaday/ai/models/knn-deep-features-fashion.m',
                    help='Feature extraction model path')
args = parser.parse_args()

mongo_client = MongoClient()
products_collection = mongo_client["deep-fashion"]["products"]

extractor = load_extractor_model(args.model_path)

result = [y for x in tqdm(os.walk(args.data_path), 'Walking through folders')
          for y in tqdm(glob.glob(os.path.join(x[0], '*.jpg')), desc='Checking image files')]

df_data_set = []
for file in tqdm(result, desc='Parsing files'):
    s = file.split('/')
    name, product, category = s[-1].replace('.jpg', ''), s[-2], s[-3]
    df_data_set.append([name, product, category, file])
df_data_set = pd.DataFrame(df_data_set)
df_data_set.columns = ['name', 'product', 'category', 'file']

df_data_set = df_data_set.loc[df_data_set.name != 'query']
rows = []

for _, row in tqdm(df_data_set.iterrows(), desc='Vectorizing', total=len(df_data_set)):
    try:
        deep_feat, color_feat = vectorize(extractor, row['file'])
        rows.append((row['name'],
                     row['product'],
                     row['category'],
                     row['file'],
                     deep_feat,
                     color_feat))
    except Exception as exp:
        logging.error('Can not vectorize {0}'.format(row['file']), exp)

model = KMeans(n_clusters=args.num_clusters, n_jobs=8).fit(rows[:, 4])
joblib.dump(model, args.knn_df_model_path)
clusters = model.predict(row[:, 4])

rows = [r.append(c) for r, c in tqdm(zip(rows, clusters), desc='Appending clusters')]

