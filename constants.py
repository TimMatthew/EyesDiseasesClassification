from torch import float32
from torchvision.transforms import v2
from torchvision import transforms

TRAIN_PATH = r'dataset/train'
VAL_PATH = r'dataset/val'
TEST_PATH = r'dataset/test'
MY_MEAN = [0.4204, 0.2791, 0.1704]
MY_STD = [0.2972, 0.2190, 0.1630]

AUGMENT_TRANSFORM = v2.Compose([
    v2.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MY_MEAN, std=MY_STD),
    transforms.RandomRotation(40),
    transforms.RandomHorizontalFlip()
    # v2.ToImage,
    # v2.ToDtype(float32, scale=True)

])

DEFAULT_TRANSFORM = v2.Compose([
    v2.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # v2.ToImage,
    # v2.ToDtype(float32, scale=True)
])