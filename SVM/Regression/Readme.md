# Support Vector Regression (SVR)

## Introduction

Support Vector Regression (SVR) is a supervised machine learning algorithm used to predict continuous numerical values.

SVR is based on Support Vector Machines. It tries to fit a function while keeping prediction errors within an epsilon margin where possible.

This project applies SVR to the Diabetes dataset and explores kernels and hyperparameter tuning.

## Dataset

The project uses the original Diabetes dataset stored at:

`Dataset/diabetes_original.csv`

The dataset contains 442 observations, ten original input features, and a continuous target representing a measure of disease progression one year after baseline.

The target is not a diabetes/no-diabetes classification label.

### Features

The original input features are:

- `age`
- `sex`
- `bmi`
- `bp`
- `s1`
- `s2`
- `s3`
- `s4`
- `s5`
- `s6`

The `sex` feature is removed for consistency with the other regression projects. The remaining nine features are used for training.

## Why SVR?

SVR is useful when we want to predict a continuous target and investigate linear or nonlinear relationships between features and the target.

The Diabetes dataset is appropriate because its target is numerical rather than a class label.

## Objectives

- Understand Support Vector Regression.
- Explore different kernel functions.
- Learn the purpose of C, epsilon, gamma, and degree.
- Understand why feature scaling is important.
- Tune hyperparameters using GridSearchCV.
- Evaluate predictions using regression metrics.

## How SVR Works

SVR tries to find a function that predicts target values while keeping errors within an epsilon-insensitive margin where possible.

The model balances a relatively simple function against penalties for errors outside the margin.

## Kernels

### Linear Kernel

Models a linear relationship between the input features and the target.

### Polynomial Kernel

Allows the model to represent polynomial relationships.

### RBF Kernel

The Radial Basis Function kernel can represent nonlinear relationships. It is a useful general-purpose option when the underlying relationship is unknown.

### Sigmoid Kernel

Uses a sigmoid-shaped kernel function. Its performance depends on the dataset and parameter settings.

No kernel is guaranteed to perform best on every dataset.

## Feature Scaling

SVR is sensitive to feature scales. A feature with a large numerical range can dominate calculations if the inputs are not standardized.

StandardScaler is fitted using only the training data. The fitted scaler is then used to transform the training and test features.

## Hyperparameters

### C

Controls the penalty for errors outside the epsilon margin.

A larger C places more emphasis on reducing these errors and may produce a more complex model.

### Epsilon

Defines the margin around the predicted function where errors do not incur the usual epsilon-insensitive loss penalty.

### Gamma

Controls how far the influence of individual samples extends for RBF, polynomial, and sigmoid kernels.

### Degree

Controls the polynomial degree for the polynomial kernel.

### Kernel

Determines the type of relationship the model can represent.

## GridSearchCV

GridSearchCV tests combinations of hyperparameters using cross-validation.

The notebook uses five-fold cross-validation and R² scoring to select the best configuration.

The test dataset is kept separate and used only for final evaluation.

## Evaluation Metrics

### Mean Absolute Error (MAE)

Measures the average absolute difference between actual and predicted values. Lower is generally better.

### Mean Squared Error (MSE)

Measures the average squared prediction error. Lower is generally better.

### Root Mean Squared Error (RMSE)

The square root of MSE. It uses the same units as the target. Lower is generally better.

### R² Score

Measures predictive performance relative to a mean-target baseline. Higher is generally better, but test R² can be negative.

## Workflow

1. Load the Diabetes dataset.
2. Inspect the data and missing values.
3. Separate features and target.
4. Split the data into training and test sets.
5. Drop the `sex` feature.
6. Standardize the numerical features.
7. Compare linear, polynomial, RBF, and sigmoid kernels.
8. Tune hyperparameters using GridSearchCV.
9. Predict the test target values.
10. Evaluate the model using MAE, MSE, RMSE, and R².
11. Plot actual and predicted values.

## Libraries Used

- pandas
- NumPy
- Matplotlib
- scikit-learn
