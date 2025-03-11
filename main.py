import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from misc import (dataset_to_train, dataset_to_val,
                  dataset_to_test, numbers_of_files,
                  show_image_and_label, count_occurences_per_classes,
                  depict_class_images_share, calculate_normalization)
from constants import (TRAIN_PATH, VAL_PATH, TEST_PATH,
                       AUGMENT_TRANSFORM as aug, DEFAULT_TRANSFORM as no_aug)
import numpy as np
import matplotlib

matplotlib.use('TkAgg')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # 7-2-1
    train_set = ImageFolder(TRAIN_PATH, transform=aug)
    val_set = ImageFolder(VAL_PATH, transform=no_aug)
    test_set = ImageFolder(TEST_PATH, transform=no_aug)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=True, num_workers=4)

    print("Train part: ", len(train_set.samples))
    print("Validation part: ", len(val_set.samples))
    print("Test part: ", len(test_set.samples))

    classes = train_set.classes
    print("Classes of datasets: ", classes)

    # all_data = torch.cat([batch[0] for batch in train_loader], dim=0)
    # calculate_normalization(all_data)

    # images, labels = next(iter(train_loader))
    #
    # show_image_and_label(images[3], labels[3], train_set.classes)
    # show_image_and_label(images[5], labels[5], train_set.classes)
    # show_image_and_label(images[14], labels[14], train_set.classes)
    # show_image_and_label(images[8], labels[8], train_set.classes)
    # show_image_and_label(images[26], labels[26], train_set.classes)
    # show_image_and_label(images[16], labels[16], train_set.classes)
    # show_image_and_label(images[11], labels[11], train_set.classes)
    #
    train_labels_occurrences = count_occurences_per_classes(train_loader)
    val_labels_occurrences = count_occurences_per_classes(val_loader)
    test_labels_occurrences = count_occurences_per_classes(test_loader)

    depict_class_images_share(classes, train_labels_occurrences)
    depict_class_images_share(classes, val_labels_occurrences)
    depict_class_images_share(classes, test_labels_occurrences)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
