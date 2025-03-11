from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib
import os
import shutil
from torch import cuda, cat, bincount, norm
import torch

matplotlib.use('TkAgg')


def dataset_to_train(dataset_path):
    pass


def dataset_to_test(dataset_path):
    new_train_dir = os.path.join(dataset_path, 'test')

    if not os.path.isdir(new_train_dir):
        os.mkdir(new_train_dir)

    eye_diseases = os.listdir(dataset_path)

    for eye_dir in eye_diseases:

        disease_path = os.path.join(dataset_path, eye_dir)
        test_images_amount = int(len(os.listdir(disease_path)) * 0.1)

        files = os.listdir(disease_path)
        i = 1

        for file in files:

            if i <= test_images_amount:
                full_file = os.path.join(disease_path, file)
                shutil.move(full_file, new_train_dir)
                i += 1
            else:
                break


def dataset_to_val(dataset_path):
    new_train_dir = os.path.join(dataset_path, 'val')

    if not os.path.isdir(new_train_dir):
        os.mkdir(new_train_dir)

    eye_diseases = os.listdir(dataset_path)

    for eye_dir in eye_diseases:

        disease_path = os.path.join(dataset_path, eye_dir)
        val_images_amount = int(len(os.listdir(disease_path)) * 0.2)

        files = os.listdir(disease_path)
        i = 1

        for file in files:

            if i <= val_images_amount:
                full_file = os.path.join(disease_path, file)
                shutil.move(full_file, new_train_dir)
                i += 1
            else:
                break


def numbers_of_files(directory):
    subdirs = os.listdir(directory)

    for disease in subdirs:
        n = 0
        files = os.listdir(os.path.join(directory, disease))
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.webp')):
                n += 1
        print(f"{disease}: ", n)


def show_image_and_label(image, label, classes):
    to_pil = transforms.ToPILImage()
    pil_image = to_pil(image)
    plt.imshow(pil_image)
    plt.axis('off')
    plt.title(f"Label: {classes[label]}")
    plt.show()


def count_occurences_per_classes(loader):
    classes = []
    # Append labels from every batch
    for images, labels in loader:
        classes.append(labels)

    # Concatinate a labels tensor into a "vector" tensor
    labels_tensor = cat(classes)
    labels_occurences = bincount(labels_tensor)

    for label, count in enumerate(labels_occurences):
        print("Class", label, ": ", count.item(), "times")

    return labels_occurences


def depict_class_images_share(classes, image_set):
    plt.figure(figsize=(10, 5))
    plt.bar(classes, image_set.numpy(), color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Occurences')
    #plt.title('Share of images per class in training part')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def calculate_normalization(loader):

    dataset_mean = torch.mean(loader, dim=(0, 2, 3), keepdim=True)
    dataset_std = torch.std(loader, dim=(0, 2, 3), keepdim=True, unbiased=False)

    print("dataset_mean: ", dataset_mean.squeeze())
    print("dataset_std: ", dataset_std.squeeze())
