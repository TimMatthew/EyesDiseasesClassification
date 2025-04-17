import math

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torchvision import transforms, models
import matplotlib.pyplot as plt
import matplotlib
import os
import shutil
from constants import EPOCHS
from torch import cuda, device, cat, bincount, norm, nn, optim, device, zeros, cuda, no_grad
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from CNN import MyCNN
from optuna import trial

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
    for images, labels in loader:
        classes.append(labels)

    labels_tensor = cat(classes)
    labels_occurences = bincount(labels_tensor)

    for label, count in enumerate(labels_occurences):
        print("Class", label, ": ", count.item(), "times")

    return labels_occurences


def depict_class_images_share(classes, image_set, subset_name):
    plt.figure(figsize=(10, 5))
    plt.bar(classes, image_set.numpy(), color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Occurrences')
    plt.title(fr'{subset_name} image share per class in training part')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def deploy_model():
    my_device = "cuda" if cuda.is_available() else "cpu"
    return MyCNN().to(my_device)


def deploy_pre_trained():
    pretrained_model = models.resnet50(models.ResNet50_Weights.IMAGENET1K_V2)
    print(pretrained_model)
    # pretrained_model.classifier[1] = nn.Linear(pretrained_model.classifier[1].in_features, 4)
    pretrained_model.fc = nn.Linear(pretrained_model.fc.in_features, 4)
    print(pretrained_model)

    # for params in pretrained_model.features[:-3].parameters():
    #     params.requires_grad = False
    #
    # for params in pretrained_model.features[-3:].parameters():
    #     params.requires_grad = True
    #
    # for params in pretrained_model.avgpool.parameters():
    #     params.requires_grad = True
    #
    # for params in pretrained_model.classifier.parameters():
    #     params.requires_grad = True



    # for params in pretrained_model.parameters():
    #     params.requires_grad = False
    #
    # for params in pretrained_model.layer4.parameters():
    #     params.requires_grad = True
    #
    # for params in pretrained_model.layer3.parameters():
    #     params.requires_grad = True
    #
    # for params in pretrained_model.fc.parameters():
    #     params.requires_grad = True

    pretrained_device = "cuda" if cuda.is_available() else "cpu"
    pretrained_model.to(pretrained_device)

    # summary.summary(pretrained_model, (3, 254, 254), 32)

    return pretrained_model


def train(model, train_loader, valid_loader, fc_lr, bone_lr, fc_decay, bone_decay, number=999, is_differ=True,
          to_save_stats=False, is_pretrained=False):
    pretrained_lr, pretrained_decay = 1e-4, 1e-5
    my_device = 'cuda' if cuda.is_available() else 'cpu'

    criterion = nn.CrossEntropyLoss()

    backbone_opt = None
    fc_opt = None
    optimizer = None
    scheduler, fc_scheduler, backbone_scheduler = None, None, None
    reduceLROnPlateauScheduler, FCReduceLROnPlateauScheduler, BoneReduceLROnPlateauScheduler = None, None, None

    if is_pretrained:
        if is_differ:
            backbone_opt = optim.Adam(model.parameters(), lr=bone_lr, weight_decay=bone_decay)
            fc_opt = optim.Adam(model.classifier.parameters(), lr=fc_lr, weight_decay=fc_decay)
            # fc_scheduler = StepLR(fc_opt, 10, 0.1)
            # backbone_scheduler = StepLR(backbone_opt, 10, 0.1)
            # FCReduceLROnPlateauScheduler = ReduceLROnPlateau(fc_opt, mode='max', factor=0.5, patience=3)
            # BoneReduceLROnPlateauScheduler = ReduceLROnPlateau(backbone_opt, mode='max', factor=0.5, patience=3)
        else:
            optimizer = optim.Adam(model.parameters(), lr=pretrained_lr, weight_decay=pretrained_decay)
            # scheduler = StepLR(optimizer, 10, 0.1)
            # reduceLROnPlateauScheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    else:
        if is_differ:
            backbone_opt = optim.Adam(model.backbone.parameters(), lr=bone_lr, weight_decay=bone_decay)
            fc_opt = optim.Adam(model.classifier.parameters(), lr=fc_lr, weight_decay=fc_decay)
            # fc_scheduler = StepLR(fc_opt, 10, 0.1)
            # backbone_scheduler = StepLR(backbone_opt, 10, 0.1)
            # FCReduceLROnPlateauScheduler = ReduceLROnPlateau(fc_opt, mode='max', factor=0.5, patience=3)
            # BoneReduceLROnPlateauScheduler = ReduceLROnPlateau(backbone_opt, mode='max', factor=0.5, patience=3)
        else:
            optimizer = optim.Adam(model.parameters(), lr=fc_lr, weight_decay=fc_decay)
            # scheduler = StepLR(optimizer, 10, 0.1)
            # reduceLROnPlateauScheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

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
        process = 0

        for inputs, labels in train_loader:
            process += 1
            print(f"{epoch + 1} epoch processing training... {process}")
            inputs, labels = inputs.to(my_device), labels.to(my_device)

            if is_differ:
                backbone_opt.zero_grad()
                fc_opt.zero_grad()
            else:
                optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            if is_differ:
                backbone_opt.step()
            else:
                optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_acc = 100 * correct / total
        epoch_loss = running_loss / len(train_loader)

        train_accs.append(epoch_acc)
        train_losses.append(epoch_loss)

        train_results = (f"Epoch #{epoch + 1}/{EPOCHS} ------- "
                         f"Training Accuracy: {epoch_acc:.4f}%, "
                         f"Loss: {epoch_loss:.4f}")

        # VALIDATION

        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0

        with no_grad():
            for inputs, labels in valid_loader:
                process += 1
                print(f"{epoch + 1} epoch processing training... {process}")

                inputs, labels = inputs.to(my_device), labels.to(my_device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                valid_loss += loss.item()
                _, predicted = outputs.max(1)
                valid_total += labels.size(0)
                valid_correct += predicted.eq(labels).sum().item()

            epoch_acc = 100 * valid_correct / valid_total
            epoch_loss = valid_loss / len(valid_loader)

            # if is_pretrained and is_differ:
            #     BoneReduceLROnPlateauScheduler.step(epoch_acc)
            #     FCReduceLROnPlateauScheduler.step(epoch_acc)
            # elif not is_pretrained and is_differ:
            #     BoneReduceLROnPlateauScheduler.step(epoch_acc)
            #     FCReduceLROnPlateauScheduler.step(epoch_acc)
            # elif not is_differ:
            #     reduceLROnPlateauScheduler.step(epoch_acc)

            valid_results = (f"\nValidation Accuracy: {epoch_acc:.4f}%, "
                             f"Loss: {epoch_loss:.4f}")

            print(train_results)
            print(valid_results)

            results.append(train_results + valid_results)

            valid_accs.append(epoch_acc)
            valid_losses.append(epoch_loss)

    for elem in results:
        print(elem)

    if is_differ:
        show_stats(train_accs, valid_accs, train_losses, valid_losses, fc_lr, bone_lr, fc_decay, bone_decay, number,
                   to_save_stats)
    else:
        show_stats(train_accs, valid_accs, train_losses, valid_losses, fc_lr, 0.0, fc_decay, 0.0,
                   number,
                   to_save_stats)

    return model, train_accs, valid_accs, train_losses, valid_losses


def test(model, test_loader, class_names, number=999, to_save=False):
    true_labels, predicted_labels = [], []

    my_device = device("cuda" if cuda.is_available() else "cpu")
    model.to(my_device)
    model.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(my_device), labels.to(my_device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(preds.cpu().numpy())

    report_dict = classification_report(true_labels, predicted_labels, target_names=class_names, output_dict=True)
    weighted_f1_score = report_dict["weighted avg"]["f1-score"]
    test_accuracy = 100 * sum(1 for x, y in zip(true_labels, predicted_labels) if x == y) / len(true_labels)

    print(classification_report(true_labels, predicted_labels, target_names=class_names))

    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    if to_save:
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)

        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=45)

        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.title("Confusion Matrix Test")
        plt.savefig(rf"..\stats\conf-matrix{number}.png")
        plt.clf()
    plt.show()

    return test_accuracy, weighted_f1_score


def show_stats(train_accs, valid_accs, train_losses, valid_losses,
               fc_lr, bone_lr, fc_decay, bone_decay, number, to_save=False):
    hyperparams = (f"FC LR: {fc_lr:},\nBackbone LR: {bone_lr},\n"
                   f"FC decay: {fc_decay:},\nBone decay: {bone_decay:}")
    epochs = range(1, EPOCHS + 1)

    if to_save:
        plt.plot(epochs, train_accs, 'y', label='Train accuracy')
        plt.plot(epochs, valid_accs, 'r', label='Valid accuracy')
        plt.plot([], [], ' ', label=hyperparams)
        plt.title('Training and Validation Accuracies')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.savefig(rf"..\stats\accuracies{number}.png")
        plt.clf()

        plt.plot(epochs, train_losses, 'y', label='Train loss')
        plt.plot(epochs, valid_losses, 'r', label='Valid loss')
        plt.plot([], [], ' ', label=hyperparams)
        plt.title('Training and Validation Losses')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(rf"..\stats\losses{number}.png")
        plt.clf()


def make_plots(fold_train_accs, fold_valid_accs, fold_train_losses, fold_valid_losses, k_folds):
    epochs = len(fold_train_accs[0])

    plt.figure(figsize=(12, 5))
    for fold in range(k_folds):
        plt.plot(range(1, epochs + 1), np.array(fold_train_accs[fold]), label=f'Fold {fold + 1}')
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy (%)')
    plt.title('Training Accuracy over Epochs')
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig(rf"..\stats\train_accs_folds.png")
    plt.clf()

    plt.figure(figsize=(12, 5))
    for fold in range(k_folds):
        plt.plot(range(1, epochs + 1), np.array(fold_valid_accs[fold]), label=f'Fold {fold + 1}')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Validation Accuracy over Epochs')
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig(rf"..\stats\valid_accs_folds.png")
    plt.clf()

    plt.figure(figsize=(12, 5))
    for fold in range(k_folds):
        plt.plot(range(1, epochs + 1), fold_train_losses[fold], label=f'Fold {fold + 1}')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig(rf"..\stats\train_losses_folds.png")
    plt.clf()

    plt.figure(figsize=(12, 5))
    for fold in range(k_folds):
        plt.plot(range(1, epochs + 1), fold_valid_losses[fold], label=f'Fold {fold + 1}')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss over Epochs')
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig(rf"..\stats\valid_losses_folds.png")
    plt.clf()
