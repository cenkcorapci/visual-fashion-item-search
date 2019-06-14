import os

import requests

from commons.config import DEFAULT_IMAGE_SIZE, IMAGE_DOWNLOAD_PATH


def download_image(image_id, size=DEFAULT_IMAGE_SIZE):
    file_url = "https://cdn.cimri.io/image/{0}x{0}/asdf_{1}.jpg".format(size, image_id)
    r = requests.get(file_url)
    save_path = IMAGE_DOWNLOAD_PATH + "{0}.jpg".format(image_id)
    with open(save_path, 'wb') as f:
        f.write(r.content)


def delete_image(image_id):
    os.remove(IMAGE_DOWNLOAD_PATH + "{0}.jpg".format(image_id))

