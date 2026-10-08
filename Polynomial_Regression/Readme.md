# Polynomial Regression

## About the Project

This project implements Polynomial Regression using the Diabetes dataset.

Polynomial Regression is used to handle non-linear relationships between the features and the target. Polynomial features are created first, and then Ridge Regression is used to reduce overfitting.

GridSearchCV is used to find the best polynomial degree and Ridge alpha value.

## Dataset

The project uses the original Diabetes dataset from scikit-learn.

The dataset contains 442 rows and 10 input features.

The target is the disease progression value one year after the baseline measurement.

Dataset file:

`Dataset/diabetes_original.csv`

## Features

The dataset contains:

- age
- sex
- bmi
- bp
- s1
- s2
- s3
- s4
- s5
- s6

The target column is:

- target

Only the `sex` column is removed in this project.

The features used for training are:

- age
- bmi
- bp
- s1
- s2
- s3
- s4
- s5
- s6

## How the Model Works

The model follows this order:

Polynomial Features → Ridge Regression → GridSearchCV → Prediction

### 1. Polynomial Features

`PolynomialFeatures` is used first to create polynomial and interaction features.

For example, with degree 2, features can produce terms such as:

`bmi²`

and interaction terms such as:

`bmi × bp`

This helps the model learn non-linear relationships.

### 2. Ridge Regularization

After creating polynomial features, Ridge Regression is applied.

Polynomial features can make the model more complex and may cause overfitting.

Ridge regularization controls the size of the model coefficients and helps reduce overfitting.

The `alpha` value controls the strength of regularization.

### 3. GridSearchCV

GridSearchCV is used to find the best combination of:

- Polynomial degree
- Ridge alpha

The polynomial degrees tested are:

- 1
- 2
- 3

The alpha values tested are:

- 0.01
- 0.1
- 1
- 10
- 100

5-fold cross-validation is used to compare the models.

## Project Workflow

1. Load the Diabetes dataset.
2. Check the dataset.
3. Separate features and target.
4. Split the data into training and testing sets.
5. Check feature correlation.
6. Drop only the `sex` column.
7. Create polynomial features.
8. Apply Ridge Regression.
9. Use GridSearchCV to find the best degree and alpha.
10. Make predictions on the test data.
11. Evaluate the model.
12. Plot actual and predicted values.

## Evaluation Metrics

The model is evaluated using:

### MAE

Mean Absolute Error measures the average difference between actual and predicted values.

Lower MAE is better.

### MSE

Mean Squared Error measures the average squared error.

Lower MSE is better.

### RMSE

Root Mean Squared Error is the square root of MSE.

Lower RMSE is better.

### R² Score

R² shows how well the model explains the variation in the target.

A value closer to 1 generally indicates better performance.

## Graph

The notebook contains an Actual vs Predicted graph.

It compares the actual target values with the values predicted by the model.

## Libraries Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn