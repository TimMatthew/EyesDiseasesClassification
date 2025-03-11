from torch import float32
from torchvision.transforms import v2
from torchvision import transforms

TRAIN_PATH = r'C:\Users\tymop\CharmRepos\DiabeticRetinopathyProject\dataset\train'
VAL_PATH = r'C:\Users\tymop\CharmRepos\DiabeticRetinopathyProject\dataset\val'
TEST_PATH = r'C:\Users\tymop\CharmRepos\DiabeticRetinopathyProject\dataset\test'

AUGMENT_TRANSFORM = v2.Compose([
    v2.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
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