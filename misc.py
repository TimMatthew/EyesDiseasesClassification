import math

from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib
import os
import shutil
from constants import EPOCHS
from torch import cuda, cat, bincount, norm, nn, optim, device, zeros, cuda, no_grad
from basic_image_eda import BasicImageEDA
import cv2
import skimage.io
from CNN import MyCNN

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
    # plt.title('Share of images per class in training part')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def deploy_model():
    my_device = "cuda" if cuda.is_available() else "cpu"
    return MyCNN().to(my_device)


def train(model, train_loader, valid_loader):
    my_device = 'cuda' if cuda.is_available() else 'cpu'

    criterion = nn.CrossEntropyLoss()

    backbone_opt = optim.Adam(model.backbone.parameters(), lr=0.0001, weight_decay=0.00001)
    fc_opt = optim.Adam(model.fc.parameters(), lr=0.001, weight_decay=0.0001)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []

    results = []
    for epoch in range(EPOCHS):

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        process = 0  # custom val for tracking a process of training per epoch

        for inputs, labels in train_loader:
            process += 1
            print(f"{epoch + 1} epoch processing training... {process}")
            inputs, labels = inputs.to(my_device), labels.to(my_device)

            backbone_opt.zero_grad()
            fc_opt.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            backbone_opt.step()
            fc_opt.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_acc = 100 * correct / total
        epoch_loss = running_loss / len(train_loader)

        train_results = (f"Epoch #{epoch + 1}/{EPOCHS} ------- "
                         f"Training Accuracy: {epoch_acc:.4f}%, "
                         f"Loss: {epoch_loss:.4f}")

        print(train_results)

        train_accs.append(epoch_acc)
        train_losses.append(epoch_loss)

        # VALIDATION

        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0

        with no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(my_device), labels.to(my_device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                valid_loss += loss.item()
                _, predicted = outputs.max(1)
                valid_total += labels.size(0)
                valid_correct += predicted.eq(labels).sum().item()

            epoch_acc = 100 * valid_correct / valid_total
            epoch_loss = valid_loss / len(valid_loader)

            valid_results = (f"\nValidation Accuracy: {epoch_acc:.4f}%, "
                             f"Loss: {epoch_loss:.4f}")

            print(valid_results)

            results.append(train_results + valid_results)

            valid_accs.append(epoch_acc)
            valid_losses.append(epoch_loss)

    for elem in results:
        print(elem)

    show_stats(train_accs, valid_accs, train_losses, valid_losses)

    return model


def show_stats(train_accs, valid_accs, train_losses, valid_losses):
    epochs = range(1, EPOCHS + 1)

    plt.plot(epochs, train_accs, 'y', label='Train accuracy')
    plt.plot(epochs, valid_accs, 'r', label='Valid accuracy')
    plt.title('Training and validation accuracies')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.show()

    plt.plot(epochs, train_losses, 'y', label='Train loss')
    plt.plot(epochs, valid_losses, 'r', label='Valid loss')
    plt.title('Training and validation losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def calculate_normalization(tensor, path):
    # dataset_mean = torch.mean(loader, dim=(0, 2, 3), keepdim=True)
    # dataset_std = torch.std(loader, dim=(0, 2, 3), keepdim=True, unbiased=False)

    # dataset_mean = torch.zeros(1)
    # dataset_std = torch.zeros(1)
    # print('==> Computing mean and std..')
    # for inputs, _labels in tqdm(tensor):
    #     for i in range(tensor.shape[1]):
    #         dataset_mean[i] += inputs[:, i, :, :].mean()
    #         dataset_std[i] += inputs[:, i, :, :].std()
    # dataset_mean.div_(len(dataset))
    # dataset_std.div_(len(dataset))
    # print(dataset_mean, dataset_std)

    # print(tensor.shape)
    # print(tensor[0])

    images_amount = tensor.shape[0]
    dataset_mean = zeros(3)
    dataset_std = zeros(3)
    BasicImageEDA.explore(path)

    # # MEAN
    #
    # image_index = 0
    #
    # for image in tensor:
    #     print(f"Image processed {image_index}...")
    #     channel_idx = 0
    #     image_index += 1
    #     for channel in image:
    #         last_row = -1
    #         last_col = 0
    #         print(f"Channel {channel_idx}")
    #         # We need to get the last pixel in a channel feature map
    #         while last_row != channel.shape[0] and last_col != channel.shape[1]:
    #             for row in channel:
    #                 last_row += 1
    #                 last_col = 0
    #                 for pixel in row:  # column
    #                     dataset_mean[channel_idx] += pixel
    #                     last_col += 1
    #         channel_idx += 1
    #
    # dataset_mean /= images_amount
    # print(dataset_mean)
    #
    # image_index = 0
    #
    # # STD
    # for image in tensor:
    #     print(f"Image processed {image_index}...")
    #     channel_idx = 0
    #     image_index += 1
    #     for channel in image:
    #         last_row = -1
    #         last_col = 0
    #         print(f"Channel {channel_idx}")
    #         # We need to get the last pixel in a channel feature map
    #         while last_row != channel.shape[0] and last_col != channel.shape[1]:
    #             for row in channel:
    #                 last_row += 1
    #                 last_col = 0
    #                 for pixel in row:  # column
    #                     dataset_std[channel_idx] += math.pow((pixel - dataset_mean[channel_idx]), 2)
    #                     last_col += 1
    #         channel_idx += 1
    #
    # dataset_std /= images_amount
    #
    # print(dataset_mean)
    # print(dataset_std)

    # MEAN: (1) tensor([0.2316, 0.1802, 0.1308]) (2) tensor([0.2306, 0.1794, 0.1303]) (3) tensor([0.2315, 0.1801, 0.1308])
