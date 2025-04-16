import numpy as np
import torchsummary as summary
from matplotlib import pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset, Subset
from basic_image_eda import BasicImageEDA
from misc import (dataset_to_train, dataset_to_val, dataset_to_test, deploy_pre_trained,
                  numbers_of_files, deploy_model, train, test, make_plots,
                  show_image_and_label, count_occurences_per_classes, depict_class_images_share)
from constants import (TRAIN_PATH, VAL_PATH, TEST_PATH, DATASET_PATH,
                       AUGMENT_TRANSFORM as aug, DEFAULT_TRANSFORM as no_aug,
                       PRETRAINED_AUGMENT_TRANSFORM as pretrain_aug)
import matplotlib
import optuna
from sklearn.model_selection import KFold
from optuna import trial, create_trial, create_study, samplers
from torch.optim.lr_scheduler import StepLR

matplotlib.use('TkAgg')


def kfold_cross_validate(train_folds_part, k_folds):
    kf = KFold(n_splits=k_folds)

    fold_train_accs, fold_valid_accs, fold_train_losses, fold_valid_losses = [], [], [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_folds_part)):
        print(f"Fold {fold + 1}/{k_folds} processing...")

        train_subset = Subset(train_folds_part, train_idx)
        val_subset = Subset(train_folds_part, val_idx)

        fold_train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=4)
        fold_val_loader = DataLoader(val_subset, batch_size=32, shuffle=True, num_workers=4)

        my_cnn = deploy_model()
        _, train_accs, valid_accs, train_losses, valid_losses = train(my_cnn, fold_train_loader, fold_val_loader,
                                                                      1.648555347420118e-05,
                                                                      0.00014930731216792622,
                                                                      2.129394268397047e-05,
                                                                      0.00048070406755845133, 9)

        fold_train_accs.append(train_accs)
        fold_valid_accs.append(valid_accs)
        fold_train_losses.append(train_losses)
        fold_valid_losses.append(valid_losses)

    make_plots(fold_train_accs, fold_valid_accs, fold_train_losses, fold_valid_losses, k_folds)

    avg_train_acc = np.mean(fold_train_accs)
    avg_valid_acc = np.mean(fold_valid_accs)

    print(f"Average train accuracy across {k_folds} folds: {avg_train_acc:.2f}%\n")
    print(f"Average valid accuracy across {k_folds} folds: {avg_valid_acc:.2f}%\n")


def objective(my_trial: trial.Trial):
    backbone_lr = my_trial.suggest_float("backbone_lr", 1e-5, 1e-2, log=True)
    fc_lr = my_trial.suggest_float("fc_lr", 1e-5, 1e-2, log=True)
    backbone_decay = my_trial.suggest_float("backbone_decay", 1e-5, 1e-2, log=True)
    fc_decay = my_trial.suggest_float("fc_decay", 1e-5, 1e-2, log=True)

    my_cnn = deploy_model()

    _, _, best_val_accs, _, _ = train(my_cnn, train_loader, val_loader, fc_lr, backbone_lr, fc_decay, backbone_decay)
    best_val_acc = max(best_val_accs)

    return best_val_acc


def show_dataset_info():
    data_analysis = BasicImageEDA.explore(DATASET_PATH)
    print(data_analysis)

    images, labels = next(iter(train_loader))

    show_image_and_label(images[3], labels[3], train_set.classes)
    show_image_and_label(images[5], labels[5], train_set.classes)
    show_image_and_label(images[14], labels[14], train_set.classes)
    show_image_and_label(images[8], labels[8], train_set.classes)
    show_image_and_label(images[26], labels[26], train_set.classes)
    show_image_and_label(images[16], labels[16], train_set.classes)
    show_image_and_label(images[11], labels[11], train_set.classes)

    train_labels_occurrences = count_occurences_per_classes(train_loader)
    val_labels_occurrences = count_occurences_per_classes(val_loader)
    test_labels_occurrences = count_occurences_per_classes(test_loader)

    depict_class_images_share(classes, train_labels_occurrences, "Train")
    depict_class_images_share(classes, val_labels_occurrences, "Valid")
    depict_class_images_share(classes, test_labels_occurrences, "Test")


def automative_optimization():

    sampler = samplers.QMCSampler()
    study = create_study(sampler=sampler, direction="maximize")
    study.optimize(objective, n_trials=10)

    # Best is trial 1 with value: VALID ACC - 74.10926365795724.
    # Best LR and Decay values
    # found: {'backbone_lr': 0.00014930731216792622, 'fc_lr': 1.648555347420118e-05,
    #         'backbone_decay': 0.00048070406755845133, 'fc_decay': 2.129394268397047e-05}
    best_params = study.best_params
    print("Best LR and Decay values found: ", best_params)

    return best_params


def manual_optimization():
    my_cnn = deploy_model()
    summary.summary(my_cnn, (3, 254, 254), 32)

    train(my_cnn, train_loader, val_loader, 1e-3, 1e-4, 1e-4, 1e-5, 1)
    test(my_cnn, test_loader, test_set.classes, 1)

    my_cnn = deploy_model()
    train(my_cnn, train_loader, val_loader, 1e-4, 1e-5, 1e-5, 1e-6, 2)
    test(my_cnn, test_loader, test_set.classes, 2)

    my_cnn = deploy_model()
    train(my_cnn, train_loader, val_loader, 1e-3, 1e-4, 1e-4, 1e-5, 3)
    test(my_cnn, test_loader, test_set.classes, 3)

    my_cnn = deploy_model()
    train(my_cnn, train_loader, val_loader, 1e-2, 1e-3, 1e-3, 1e-4, 4)
    test(my_cnn, test_loader, test_set.classes, 4)

    my_cnn = deploy_model()
    train(my_cnn, train_loader, val_loader, 1e-3, 1e-4, 1e-4, 1e-5, 5)
    test(my_cnn, test_loader, test_set.classes, 5)

    my_cnn = deploy_model()
    train(my_cnn, train_loader, val_loader, 1e-3, 1e-4, 0, 0, 6)
    test(my_cnn, test_loader, test_set.classes, 6)

    my_cnn = deploy_model()
    train(my_cnn, train_loader, val_loader, 1e-4, 1e-5, 0, 0, 7)
    test(my_cnn, test_loader, test_set.classes, 7)

    my_cnn = deploy_model()
    train(my_cnn, train_loader, val_loader, 1e-2, 1e-3, 0, 0, 8)
    test(my_cnn, test_loader, test_set.classes, 8)


def launch_cnn_with_best_hyperparameters():
    best_hyperparams = automative_optimization()
    best_backbone_lr = best_hyperparams["backbone_lr"]
    best_fc_lr = best_hyperparams["fc_lr"]
    best_backbone_decay = best_hyperparams["backbone_decay"]
    best_fc_decay = best_hyperparams["fc_decay"]
    manual_optimization()

    my_cnn = deploy_model()
    train(my_cnn, train_loader, val_loader,
          best_fc_lr,
          best_backbone_lr,
          best_fc_decay,
          best_backbone_decay, 10, True)
    test(my_cnn, test_loader, test_set.classes, 8)


if __name__ == '__main__':
    # 7-2-1 of 10 shares
    train_set = ImageFolder(TRAIN_PATH, transform=aug)
    val_set = ImageFolder(VAL_PATH, transform=no_aug)
    test_set = ImageFolder(TEST_PATH, transform=no_aug)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=True, num_workers=4)
    full_dataset = ConcatDataset([train_set, val_set, test_set])

    print("Train part: ", len(train_set.samples))
    print("Validation part: ", len(val_set.samples))
    print("Test part: ", len(test_set.samples))

    classes = train_set.classes
    print("Classes of datasets: ", classes)
    # deploy_pre_trained()
    # show_dataset_info()
    # automative_optimization()

    my_cnn = deploy_model()
    print(my_cnn)
    train(my_cnn, train_loader, val_loader,
          1e-4,
          1e-5,
          1e-5,
          1e-6, 34, True, True)
    test(my_cnn, test_loader, test_set.classes, 34, True)

    # pretrained_model = deploy_pre_trained()
    # train(pretrained_model, train_loader, val_loader, 1e-4, 1e-5, 1e-5, 1e-6, 28, True, True, True)
    # test(pretrained_model, test_loader, train_set.classes, 28, True)
