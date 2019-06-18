import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from tqdm import tqdm

from commons.config import PRODUCT_IMAGES_CSV_PATH, DIM_RED_MODEL_PATH
from feature_extraction.image_vectorization import ImageVectorizationModel, ImageClassificationModel
from scipy import spatial
import random

# Get data
df = pd.read_csv(PRODUCT_IMAGES_CSV_PATH, error_bad_lines=False)

df['image'] = df.apply(lambda row: str(row['id']) + ".jpg", axis=1)
df = df.dropna()

df = df[['image', 'productDisplayName']]

print('Targeting;\n')
for t in self._df[target_column].unique().tolist():
    print(t)

# Extract visual features
img_vectorizer = ImageVectorizationModel(ImageClassificationModel.vgg16)
image_vectors = []
image_list = []
for _, row in tqdm(df.sample(3).iterrows(), total=len(df), desc='Vectorizing images'):
    img_ids = []
    [img_ids.append(img_id) for img_id in row['productImages'].split('-')]
    [img_ids.append(img_id) for img_id in row['offerImages'].split('-')]
    img_ids = set(img_ids)
    for img_id in tqdm(img_ids, desc='Vectorizing {0}'.format(row['productId'])):
        try:
            image_vectors.append(img_vectorizer.get_feature_vectors(img_id))
            image_list.append([row['productId'], img_id])
        except Exception as exp:
            logging.error(exp)

# Reduce dimensionality
pca = PCA(n_components=2)
pca.fit(image_vectors)
X_embedded = pca.transform(image_vectors)

with open(DIM_RED_MODEL_PATH, 'wb') as pca_file:
    pickle.dump(pca, pca_file)

X_embedded = np.array(X_embedded)
image_list = np.array(image_list)
df_concatenated = np.concatenate((image_list, X_embedded), axis=1)
df_concatenated = pd.DataFrame(df_concatenated, columns=['productId', 'imageId', 'x', 'y'])

# Generate index
tree = spatial.KDTree(image_vectors)
distances, indexes = tree.query(random.sample(image_vectors, 1), k=5)
for index in indexes:
    print(image_list[index])
