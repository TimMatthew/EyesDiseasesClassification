from torch import float32
from torchvision.transforms import v2
from torchvision import transforms

TRAIN_PATH = r'dataset/train'
VAL_PATH = r'dataset/val'
TEST_PATH = r'dataset/test'
MY_MEAN = [0.42017278, 0.2799531, 0.17135893]
MY_STD = [0.29486006, 0.21830283, 0.16484465]
EPOCHS = 20

AUGMENT_TRANSFORM = v2.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MY_MEAN, MY_STD),  # BasicEDA Custom parameters
    transforms.RandomRotation(40),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomCrop(112)
])

DEFAULT_TRANSFORM = v2.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # ImageNet parameters
    transforms.Normalize(MY_MEAN, MY_STD),  # BasicEDA Custom parameters
])
