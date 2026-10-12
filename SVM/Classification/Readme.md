# Support Vector Machine (SVM) Classifier

## Introduction

Support Vector Machine (SVM) is a supervised machine learning algorithm used for classification.

SVM attempts to find an optimal hyperplane that separates different classes while maximizing the margin between the decision boundary and the nearest training samples.

This project implements SVM classification using the Iris dataset.

## Dataset Selection

The Iris dataset is used because it contains four numerical flower measurements and three classes.

The target column contains the known flower species labels.

## Dataset Features

- `sepal length (cm)`: Length of the sepal.
- `sepal width (cm)`: Width of the sepal.
- `petal length (cm)`: Length of the petal.
- `petal width (cm)`: Width of the petal.
- `target`: Flower species label.

The target column is used as the output label, while the four measurements are used as input features.

## Objectives

- Understand the optimal hyperplane.
- Understand support vectors and margins.
- Learn about kernel functions.
- Compare different SVM kernels.
- Tune hyperparameters using GridSearchCV.
- Evaluate classification performance.

## Optimal Hyperplane

A hyperplane is a decision boundary used to separate classes.

For a linear SVM, the hyperplane is represented as:

w · x + b = 0

where:

- `w` represents the weights.
- `x` represents the input features.
- `b` represents the bias.

SVM attempts to maximize the margin between the hyperplane and the nearest training samples.

## Support Vectors

Support vectors are the training samples closest to the decision boundary.

These samples play an important role in determining the position of the SVM decision boundary.

## Kernel Trick

The kernel trick allows SVM to represent nonlinear decision boundaries without explicitly computing all coordinates of a higher-dimensional feature transformation.

### Linear Kernel

Suitable when a linear decision boundary works well.

### Polynomial Kernel

Can represent more complex relationships using polynomial combinations of features.

### RBF Kernel

Can model nonlinear decision boundaries and is a useful general-purpose kernel.

### Sigmoid Kernel

Uses a sigmoid-shaped kernel function.

Different kernels can perform differently depending on the dataset.

## Feature Scaling

SVM is sensitive to feature scales because it relies on distances and dot products.

StandardScaler is used to standardize the input features before training.

The scaler is fitted on the training data and then applied to the test data.

## Hyperparameters

### C

Controls the penalty for classification errors.

A smaller C allows more training errors in exchange for a wider margin. A larger C penalizes errors more strongly and can increase model complexity.

### Gamma

Controls the influence range of training samples for RBF, polynomial, and sigmoid kernels.

A high gamma value can produce a more complex decision boundary.

### Degree

Controls the polynomial degree for the polynomial kernel.

### Kernel

Determines the type of decision boundary the model can represent.

## GridSearchCV

GridSearchCV tests different hyperparameter combinations using cross-validation.

This project uses five-fold cross-validation and accuracy scoring to select the best model configuration.

The test dataset is kept separate for final evaluation.

## Evaluation Metrics

### Accuracy

Measures the proportion of correctly classified samples.

### Precision

Measures how many samples predicted as a particular class actually belong to that class.

### Recall

Measures how many samples belonging to a class were correctly identified.

### F1-Score

The harmonic mean of precision and recall.

### Confusion Matrix

Shows the number of correct and incorrect predictions for each class.

## Workflow

1. Load the Iris dataset.
2. Inspect the dataset.
3. Check class distribution and missing values.
4. Separate input features and target.
5. Split the dataset into training and testing sets.
6. Standardize the features.
7. Train a baseline linear SVM.
8. Compare different kernels.
9. Tune hyperparameters using GridSearchCV.
10. Evaluate the best model.
11. Compare baseline and tuned performance.

## Libraries Used

- pandas
- NumPy
- Matplotlib
- scikit-learn