# Eyes Diseases Classification

**Eyes diseases classfication task by Convolutional Neural Network**

This project is included in "Deep Learning for Computer Vision" course provided by Kyiv-Mohyla Academy University and Grid Dynamics. 
The purpose of the project is an implementation of eyes diseases classification by Convolutional Neural Networks (CNN). The dataset used in the project is <a href="https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification/data">Eyes Diseases Classification by Guna Venkat Doddi<a/>.
The dataset contains 4 classes that depict glaucoma, cataract, diabetic retinopathy diseases and healthy state.

The following libraries used in the project:

+ **PyTorch (torch, torchvision)** - constructing a CNN, setting optimizers, training-validation-testing, data augmentation
+ **MatPlotLib** - dataset samples visualising, setting accuracies and losses charts
+ **Seaborn** - constructing a confusion matrix for test stage
+ **Scikit-learn** - K-Fold Cross-validation implementation
+ **Optuna** - Automative model hyperparameters search
+ **BasicEDA** - Exploratory Data Analysis package, computing the normalization parameters

## Dataset Analysis

### General info

The dataset consist of 4 217 images of <formats> formats and split to 4 classes `glaucoma` `cataract` `diabetic retinopathy` and `healthy`.

![Diabetic retinopathy](https://github.com/TimMatthew/EyesDiseasesClassification/blob/master/readme_samples/Figure_8.png)
![Cataract](https://github.com/TimMatthew/EyesDiseasesClassification/blob/master/readme_samples/Figure_3.png)
![Healthy eye](https://github.com/TimMatthew/EyesDiseasesClassification/blob/master/readme_samples/Figure_4.png)
![Glaucoma](https://github.com/TimMatthew/EyesDiseasesClassification/blob/master/readme_samples/Figure_6.png)

Images dimensions are .png, .jpg, .jpeg. Using BasicEDA the following images characteristics were found:
+ min-max image width: **256, 2592**
+ min-max image height: **256, 1728**
+ median width: **512**
+ median height: **512**
+ mean width: **532.86**
+ mean height: **492.56**
+ mean height/width ratio: **~0.9244**

The dataset was split by `train`, `val` and `test` parts with the 8-1-1 share:
+ Train part:  **3369**
+ Validation part:  **421**
+ Test part:  **427**

### The images proportion per class

![Train images share per class](https://github.com/TimMatthew/EyesDiseasesClassification/blob/master/readme_samples/train_histogram.png)
![Validation images share per class](https://github.com/TimMatthew/EyesDiseasesClassification/blob/master/readme_samples/val_histogram.png)
![Test images share per class](https://github.com/TimMatthew/EyesDiseasesClassification/blob/master/readme_samples/test_histogram.png)

### Dataset Outliers


## CNN Architecture

The main layers block of the model is actually an adaptation of Conv2dNormAction layer. 
But SiLU function was taken as an activation instead of ReLU.

````
        nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(feature_maps * 2, 1e-05, 0.1, True),
        nn.MaxPool2d(kernel_size=2, stride=1),
        nn.SiLU()
````

Then we have FC-layer:

````
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(baseline_maps * 32, baseline_maps * 16),
            nn.Dropout(0.3),
            nn.Linear(baseline_maps * 16, baseline_maps * 4),
            nn.Dropout(0.3),
            nn.SiLU(),
            nn.Linear(baseline_maps * 4, classes)
        )
````

Hence, the CNN architecture

+ 6 Conv2dNormAction with SiLU
+ Stochastic depth with 0.19 probability
+ AdaptiveAvgPool2d
+ FC

In all cases the following augmention was used:

````
AUGMENT_TRANSFORM = v2.Compose([
    transforms.Resize((254, 254)),
    transforms.RandomResizedCrop(254, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(MY_MEAN, MY_STD),
])

DEFAULT_TRANSFORM = v2.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MY_MEAN, MY_STD),  # BasicEDA Custom parameters
])
````


## Metrics Analysis

### Custom CNN with a single optimizer

**Accuracies**

![](https://github.com/TimMatthew/EyesDiseasesClassification/blob/master/stats/accuracies6.png)

Model accuracy. Learning rate: 1e-3, Weight decay: 1e-4

![](https://github.com/TimMatthew/EyesDiseasesClassification/blob/master/stats/accuracies7.png)

Model accuracy .Learning rate: 1e-4, Weight decay: 1e-5

**Losses**

![](https://github.com/TimMatthew/EyesDiseasesClassification/blob/master/stats/losses6.png)

Model loss. Learning rate: 1e-3, Weight decay: 1e-4

![](https://github.com/TimMatthew/EyesDiseasesClassification/blob/master/stats/losses7.png)

Model loss. Learning rate: 1e-4, Weight decay: 1e-5

**Confusion matrices**

![](https://github.com/TimMatthew/EyesDiseasesClassification/blob/master/stats/conf-matrix6.png)

Model loss. Learning rate: 1e-3, Weight decay: 1e-4

![](https://github.com/TimMatthew/EyesDiseasesClassification/blob/master/stats/conf-matrix7.png)

Model testing confusion matrix. Learning rate: 1e-4, Weight decay: 1e-5


### Custom CNN with multiple optimizers

![](https://github.com/TimMatthew/EyesDiseasesClassification/blob/master/stats/accuracies2.png)

Model accuracy. FC Learning rate: 1e-4, FC Weight decay: 1e-5, Backbone Learning rate: 1e-5, Backbone Weight decay: 1e-6

![](https://github.com/TimMatthew/EyesDiseasesClassification/blob/master/stats/losses2.png)

Model loss. FC Learning rate: 1e-4, FC Weight decay: 1e-5, Backbone Learning rate: 1e-5, Backbone Weight decay: 1e-6

![](https://github.com/TimMatthew/EyesDiseasesClassification/blob/master/stats/conf-matrix20.png)

Model confusion matrix. FC Learning rate: 1e-4, FC Weight decay: 1e-5, Backbone Learning rate: 1e-5, Backbone Weight decay: 1e-6


### ResNet50 with a single optimizer

![](https://github.com/TimMatthew/EyesDiseasesClassification/blob/master/resnet50_stats/accuracies22.png)

ResNet50 accuracy. Learning rate: 1e-4, Weight decay: 1e-5

![](https://github.com/TimMatthew/EyesDiseasesClassification/blob/master/resnet50_stats/losses22.png)

ResNet50 losses. Learning rate: 1e-4, Weight decay: 1e-5

![](https://github.com/TimMatthew/EyesDiseasesClassification/blob/master/resnet50_stats/conf-matrix22.png)

ResNet50 testing confusion matrix. Learning rate: 1e-4, Weight decay: 1e-5


### ResNet50 with multiple optimizer

![](https://github.com/TimMatthew/EyesDiseasesClassification/blob/master/resnet50_stats/accuracies21.png)

ResNet50 accuracy. FC Learning rate: 1e-4, Backbone Learning rate: 1e-5, FC Weight decay: 1e-5, Backbone Weight decay: 1e-6

![](https://github.com/TimMatthew/EyesDiseasesClassification/blob/master/resnet50_stats/losses21.png)

ResNet50 loss. FC Learning rate: 1e-4, Backbone Learning rate: 1e-5, FC Weight decay: 1e-5, Backbone Weight decay: 1e-6

![](https://github.com/TimMatthew/EyesDiseasesClassification/blob/master/resnet50_stats/conf-matrix21.png)

ResNet50 confusion matrix. FC Learning rate: 1e-4, Backbone Learning rate: 1e-5, FC Weight decay: 1e-5, Backbone Weight decay: 1e-6

### ResNet50 with multiple optimizer. 2 Freezed layers

![](https://github.com/TimMatthew/EyesDiseasesClassification/blob/master/resnet50_stats/accuracies26_layer3-4_freezed.png)

ResNet50 accuracy. FC Learning rate: 1e-4, Backbone Learning rate: 1e-5, FC Weight decay: 1e-5, Backbone Weight decay: 1e-6

![](https://github.com/TimMatthew/EyesDiseasesClassification/blob/master/resnet50_stats/losses26_layer3-4_freezed.png)

ResNet50 loss. FC Learning rate: 1e-4, Backbone Learning rate: 1e-5, FC Weight decay: 1e-5, Backbone Weight decay: 1e-6

![](https://github.com/TimMatthew/EyesDiseasesClassification/blob/master/resnet50_stats/conf-matrix_layer3-4_freezed.png)

ResNet50 confusion matrix. FC Learning rate: 1e-4, Backbone Learning rate: 1e-5, FC Weight decay: 1e-5, Backbone Weight decay: 1e-6

### Automative optimization


The optimal hyperparameters values were found in range of 1e-5 and 1e-2.

````
def objective(my_trial: trial.Trial):
    backbone_lr = my_trial.suggest_float("backbone_lr", 1e-5, 1e-2, log=True)
    fc_lr = my_trial.suggest_float("fc_lr", 1e-5, 1e-2, log=True)
    backbone_decay = my_trial.suggest_float("backbone_decay", 1e-5, 1e-2, log=True)
    fc_decay = my_trial.suggest_float("fc_decay", 1e-5, 1e-2, log=True)

    my_cnn = deploy_model()

    _, _, best_val_accs, _, _ = train(my_cnn, train_loader, val_loader, fc_lr, backbone_lr, fc_decay, backbone_decay)
    best_val_acc = max(best_val_accs)

    return best_val_acc
````

Then Quasi Monte carlo (QMS) sampler were used for search as more advanced method to find the best values:

````
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
````

The following results were extracted:

**Custom CNN with multiple opitmiziers. Automative hyperparameters optimization**
![](https://github.com/TimMatthew/EyesDiseasesClassification/blob/master/stats/accuracies13.png)

![](https://github.com/TimMatthew/EyesDiseasesClassification/blob/master/stats/losses13.png)

![](https://github.com/TimMatthew/EyesDiseasesClassification/blob/master/stats/conf-matrix13.png)


### 5-Folds Cross-validation

Next the experiments with K-Folds have been performedm where opitmized hyperparameters were used for training

FC LR: 1.648555347420118e-05,
Bone LR: 0.00014930731216792622,
FC Decay: 2.129394268397047e-05,
Bone Deacy: 0.00048070406755845133

![](https://github.com/TimMatthew/EyesDiseasesClassification/blob/master/stats/folds_train_accs.jpg)
![](https://github.com/TimMatthew/EyesDiseasesClassification/blob/master/stats/folds_valid_accs.jpg)
![](https://github.com/TimMatthew/EyesDiseasesClassification/blob/master/stats/folds_train_loss.jpg)
![](https://github.com/TimMatthew/EyesDiseasesClassification/blob/master/stats/folds_valid_loss.jpg)



