from torchvision import transforms

from commons.config import CROP_SIZE

data_transform = transforms.Compose([
    transforms.Scale(CROP_SIZE),
    transforms.CenterCrop(CROP_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
