import torch
import torchsummary as summary
from torchvision.datasets import ImageFolder
from torchvision import transforms
import CNN
from torch.utils.data import DataLoader, ConcatDataset
from misc import (dataset_to_train, dataset_to_val,
                  dataset_to_test, numbers_of_files, deploy_model, train,
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
    val_loader = DataLoader(val_set, batch_size=16, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=True, num_workers=4)
    full_dataset = ConcatDataset([train_set, val_set, test_set])

    print("Train part: ", len(train_set.samples))
    print("Validation part: ", len(val_set.samples))
    print("Test part: ", len(test_set.samples))

    classes = train_set.classes
    print("Classes of datasets: ", classes)

    # matrix = torch.tensor(
    #
    #     [[[0.2035, 1.2959, 1.8101, -0.4644],
    #     [1.5027, -0.3270, 0.5905, 0.6538],
    #     [-1.5745, 1.3330, -0.5596, -0.6548],
    #     [0.1264, -0.5080, 1.6420, 0.1992]],
    #
    #     [[0.2035, 1.2959, 1.8101, -0.4644],
    #      [1.5027, -0.3270, 0.5905, 0.6538],
    #      [-1.5745, 1.3330, -0.5596, -0.6548],
    #      [0.1264, -0.5080, 1.6420, 0.1992]],
    #
    #     [[0.2035, 1.2959, 1.8101, -0.4644],
    #      [1.5027, -0.3270, 0.5905, 0.6538],
    #      [-1.5745, 1.3330, -0.5596, -0.6548],
    #      [0.1264, -0.5080, 1.6420, 0.1992]]]
    # )
    #
    # # Compute mean and standard deviation
    # mean = torch.mean(matrix, dim=(1, 2))
    # print(mean)

    # train_data = torch.cat([batch[0] for batch in train_loader], dim=0)
    # val_data = torch.cat([batch[0] for batch in val_loader], dim=0)
    # test_data = torch.cat([batch[0] for batch in test_loader], dim=0)
    # all_data = torch.cat([train_data, val_data, test_data], dim=0)
    # print(all_data.shape)
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
    # train_labels_occurrences = count_occurences_per_classes(train_loader)
    # val_labels_occurrences = count_occurences_per_classes(val_loader)
    # test_labels_occurrences = count_occurences_per_classes(test_loader)
    #
    # depict_class_images_share(classes, train_labels_occurrences)
    # depict_class_images_share(classes, val_labels_occurrences)
    # depict_class_images_share(classes, test_labels_occurrences)

    my_cnn = deploy_model()
    # calculate_normalization(torch.zeros((1, 3, 224, 224)), r'dataset')
    print(my_cnn)

    # summary.summary(my_cnn, (3, 224, 224), 32)

    train(my_cnn, train_loader, val_loader)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
