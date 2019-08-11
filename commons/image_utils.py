import requests
from PIL import Image
from torchvision import transforms

from commons.config import CROP_SIZE


def read_crop(img_path, bbox):
    """
    Extracts the part of the image from given bounding box coordinates
    :param img_path: full path for the image
    :param bbox: bounding box as tuple ; x1, y1, x2, y2
    :return: part of image cropped from the bbox coordinates
    """
    with open(img_path, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('RGB')
    x1, y1, x2, y2 = bbox
    if x1 < x2 <= img.size[0] and y1 < y2 <= img.size[1]:
        img = img.crop((x1, y1, x2, y2))
    return img


def scale_image(image, max_size, method=Image.ANTIALIAS):
    """
    resize 'image' to 'max_size' keeping the aspect ratio
    and place it in center of white 'max_size' image
    """
    image.thumbnail(max_size, method)
    offset = (int((max_size[0] - image.size[0]) / 2), int((max_size[1] - image.size[1]) / 2))
    back = Image.new("RGB", max_size, "white")
    back.paste(image, offset)

    return back


def download_image(image_url, download_path):
    """
    downloads the image to given image path
    :param image_url: url
    :param download_path: download path
    :return:
    """
    name = image_url.split('/')[-1]
    r = requests.get(image_url)
    save_path = download_path + name
    with open(save_path, 'wb') as f:
        f.write(r.content)


img_transform = transforms.Compose([
    transforms.Scale(CROP_SIZE),
    transforms.CenterCrop(CROP_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def img_to_input(img_path):
    with open(img_path, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('RGB')
            return img_transform(img)
