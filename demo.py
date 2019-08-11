import pandas as pd
from IPython.core.display import display
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

from ai.dnn import FashionModel, ColorModel, PoolingModel
from commons.config import DUMPED_MODEL
from commons.image_utils import img_to_input
from feature_extractor import FeatureExtractor

IMAGE_BASE_PATH = '/run/media/twoaday/data-storag/data-sets/multi-view-clothing/images/'

fashion_model = FashionModel(freeze_param=True, model_path=DUMPED_MODEL).cuda()
color_model = ColorModel().cuda()
pooling_model = PoolingModel().cuda()

extractor = FeatureExtractor(fashion_model, color_model, pooling_model)

df_triplets_medium = pd.read_csv('/run/media/twoaday/data-storag/data-sets/multi-view-clothing/triplets-medium.csv')

sample = [[row['anchor'], row['positive'], row['negative']] for _, row in df_triplets_medium.sample(1).iterrows()][0]
anchor_vec, anchor_vec_color = extractor(img_to_input(IMAGE_BASE_PATH + sample[0]))

for name, img_id in zip(['anchor', 'positive', 'negative'], sample):
    path = IMAGE_BASE_PATH + img_id
    img = Image.open(path)
    img = img.resize((128, 128))
    vec, color_vec = extractor(img_to_input(path))
    display(img)
    print('{0} : {1}'.format(name, cosine_similarity([anchor_vec, vec])))
