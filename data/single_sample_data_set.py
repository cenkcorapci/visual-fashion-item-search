import torch.utils.data as data
from PIL import Image

from data.transformers import data_transform


class SingleSampleDataSet(data.Dataset):
    def __init__(self, img_path):
        self.img_path = img_path

    def __len__(self):
        return 1

    def __getitem__(self, index):
        with open(self.img_path, 'rb') as f:
            with Image.open(f) as img:
                img = img.convert('RGB')
                data = data_transform(img)
                return data
