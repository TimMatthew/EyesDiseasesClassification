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

![Train images share per class](https://github.com/TimMatthew/EyesDiseasesClassification/blob/master/readme_samples/train_histgoram.png)
![Validation images share per class](https://github.com/TimMatthew/EyesDiseasesClassification/blob/master/readme_samples/val_histgoram.png)
![Test images share per class](https://github.com/TimMatthew/EyesDiseasesClassification/blob/master/readme_samples/test_histgoram.png)

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


## Metrics Analysis

### Loss-Accuracy on training and validation
1st configuration is custom CNN

![]()

### Testing metrics and Confusion matrix

### Automative optimization

### Transfer learining

