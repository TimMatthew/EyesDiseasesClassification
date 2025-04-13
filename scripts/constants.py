from torch import float32
from torchvision.transforms import v2
from torchvision import transforms

TRAIN_PATH = r'../dataset/train'
VAL_PATH = r'../dataset/val'
TEST_PATH = r'../dataset/test'
DATASET_PATH = r'../dataset'
MY_MEAN = [0.42017278, 0.2799531, 0.17135893]
MY_STD = [0.29486006, 0.21830283, 0.16484465]
# mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225) -- IMAGE NET
EPOCHS = 30

AUGMENT_TRANSFORM = v2.Compose([
    transforms.Resize((254, 254)),
    transforms.RandomResizedCrop(254, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(MY_MEAN, MY_STD),
])

PRETRAINED_AUGMENT_TRANSFORM = v2.Compose([
    transforms.Resize((254, 254)),
    # transforms.RandomResizedCrop(254, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(MY_MEAN, MY_STD),
])


DEFAULT_TRANSFORM = v2.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MY_MEAN, MY_STD),  # BasicEDA Custom parameters
])
