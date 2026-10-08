# KNN Regression

## About the Project

This project implements the K-Nearest Neighbors (KNN) algorithm for regression using the Diabetes dataset.

KNN Regression is used when the target variable is continuous.

Instead of predicting a class, KNN Regression predicts a numerical value using the target values of the nearest training observations.

The main concepts covered in this project are:

- KNN Regression
- K-value selection
- Distance metrics
- Uniform weighting
- Distance weighting
- Feature scaling
- Train-test split
- Cross-validation
- GridSearchCV
- Hyperparameter tuning
- MAE
- MSE
- RMSE
- R² score
- Actual vs Predicted visualization

## Dataset

The Diabetes dataset is used for this regression problem.

The dataset contains numerical features related to patients and a continuous target representing a quantitative measure of disease progression.

The dataset is stored in:

    Dataset/diabetes_original.csv

The original dataset contains 442 observations and 10 input features along with the target.

The features are:

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

The target represents a quantitative measure of disease progression.

## Feature Used in the Model

The `sex` feature is dropped before training.

The remaining features are used for KNN regression:

- age
- bmi
- bp
- s1
- s2
- s3
- s4
- s5
- s6

Only the `sex` feature is removed.

## Why Diabetes Dataset?

The Diabetes dataset is suitable for KNN Regression because the target variable is continuous.

For example, the target can have values such as:

    120
    180
    220
    275

These are numerical values rather than categories.

Therefore, KNN Regression can estimate a numerical target using the target values of nearby observations.

## What is KNN Regression?

K-Nearest Neighbors Regression is a supervised machine learning algorithm used to predict a continuous numerical value.

For a new observation, KNN:

1. Calculates the distance between the new observation and training observations.
2. Finds the nearest observations.
3. Selects the K nearest observations.
4. Takes their actual target values.
5. Calculates an average or distance-weighted average.
6. Uses that value as the prediction.

The basic process is:

    New Data Point
    → Calculate Distances
    → Find K Nearest Points
    → Take Their Actual Target Values
    → Average or Weighted Average
    → Predicted Value

## Example of KNN Regression

Suppose K = 3.

The three nearest training observations have these actual target values:

    200
    220
    250

With:

    weights = "uniform"

all three observations have equal importance.

The prediction is:

    (200 + 220 + 250) / 3

    = 223.33

Therefore, the predicted target value is:

    223.33

## Why Do We Need Distance?

The final prediction uses the actual target values of the nearest observations, but first KNN needs to determine which observations are actually nearest.

Suppose there are 100 training observations.

For a new observation, KNN calculates distances such as:

    Point A → Distance = 1.2
    Point B → Distance = 1.8
    Point C → Distance = 2.1
    Point D → Distance = 5.7
    Point E → Distance = 8.3

If:

    K = 3

then the model selects:

    Point A
    Point B
    Point C

Only their target values are used.

Therefore, distance is used to decide:

    Which observations should be used?

After selecting the observations, their actual target values are used to make the prediction.

The complete process is:

    Calculate Distance
    → Find Nearest Points
    → Select K Points
    → Take Actual Target Values
    → Calculate Average
    → Prediction

## K-Value

K represents the number of nearest observations used for prediction.

For example:

    K = 1

uses only the closest observation.

    K = 5

uses the five closest observations.

    K = 10

uses the ten closest observations.

The value of K affects how the model behaves.

## Small K

A small K makes the model more sensitive to nearby observations.

For example:

    K = 1

means the prediction is based only on the nearest observation.

Advantages:

- Captures local patterns.
- Can respond strongly to nearby observations.

Disadvantages:

- Sensitive to noise.
- Can overfit the training data.

## Large K

A large K considers more observations.

Advantages:

- Less sensitive to individual observations.
- Produces smoother predictions.

Disadvantages:

- May ignore local patterns.
- Can underfit if K becomes too large.

Therefore, different K values are tested using GridSearchCV.

## Distance Metrics

KNN Regression requires a distance metric to determine which observations are closest to the new observation.

The distance metrics tested in this project are:

- Euclidean
- Manhattan
- Minkowski

## Euclidean Distance

Euclidean distance represents the straight-line distance between two observations.

For two points:

    (x1, y1)

and:

    (x2, y2)

the distance is:

    d = sqrt((x2 - x1)^2 + (y2 - y1)^2)

The observations with smaller Euclidean distance are considered more similar according to this metric.

## Manhattan Distance

Manhattan distance calculates the sum of the absolute differences between the feature values.

The formula is:

    d = |x2 - x1| + |y2 - y1|

It provides another way of measuring similarity between observations.

## Minkowski Distance

Minkowski distance is a generalized distance metric.

It can represent different distance calculations depending on its parameter.

It is also tested during hyperparameter tuning.

## Distance Determines Which Target Values Are Averaged

This is an important concept in KNN Regression.

Suppose the distances from a new observation are:

    Point A → 1.0
    Point B → 1.5
    Point C → 2.0
    Point D → 5.0
    Point E → 7.0

If:

    K = 3

then only:

    A, B, C

are selected.

Suppose their actual target values are:

    A → 200
    B → 220
    C → 250

With uniform weighting:

    Prediction = (200 + 220 + 250) / 3

    Prediction = 223.33

Points D and E are not included because they are farther away and are outside the selected K nearest observations.

Therefore:

    Distance → Finds the neighbors

    Actual target values → Produce the prediction

## Uniform Weights

With:

    weights = "uniform"

every selected neighbor has equal importance.

For example:

    K = 3

Target values:

    200
    220
    250

Prediction:

    (200 + 220 + 250) / 3

    = 223.33

The distance is used only to select the three nearest observations.

After selection, all three observations contribute equally.

## Distance Weights

With:

    weights = "distance"

closer observations receive more influence.

For example, if one neighbor is very close to the new observation and another is farther away, the closer neighbor will have a greater effect on the final prediction.

Therefore:

    Uniform weighting
    → All selected neighbors have equal importance

    Distance weighting
    → Closer neighbors have greater importance

## Feature Scaling

Feature scaling is extremely important for KNN because KNN depends on distance.

Suppose one feature has values between:

    1 and 10

and another feature has values between:

    1000 and 10000

Without scaling, the larger-valued feature can dominate the distance calculation.

This can change which observations are selected as nearest neighbors.

Therefore, StandardScaler is used.

## StandardScaler

StandardScaler transforms the numerical features to a comparable scale.

The scaler is fitted using the training data.

The same fitted scaler is then used to transform the test data.

The process is:

    Training Data
    → Fit StandardScaler
    → Transform Training Data
    → Transform Test Data
    → KNN Regression

The scaler is not fitted separately on the test set.

## GridSearchCV

GridSearchCV is used to find suitable KNN hyperparameters.

The parameter grid is:

    param_grid = {
        "n_neighbors": range(1, 21),
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"]
    }

This tests:

    20 K values
    ×
    2 weighting methods
    ×
    3 distance metrics

Total:

    20 × 2 × 3 = 120 combinations

Each combination is evaluated using 5-fold cross-validation.

For regression, R² is used as the scoring metric.

The parameter combination with the best cross-validation R² score is selected.

## Cross-Validation

Five-fold cross-validation is used during GridSearchCV.

The training data is divided into five parts.

Four parts are used for training and one part is used for validation.

This process is repeated five times.

Each part becomes the validation set once.

The results are then combined to calculate the cross-validation score.

This helps compare different parameter combinations more reliably.

## Best Parameters

After GridSearchCV, the model provides:

- Best K
- Best weighting method
- Best distance metric
- Best cross-validation R² score

These parameters are used to create the final KNN regression model.

## Regression Evaluation

The KNN regression model is evaluated using:

- Mean Absolute Error
- Mean Squared Error
- Root Mean Squared Error
- R² Score

## Mean Absolute Error

MAE measures the average absolute difference between the actual and predicted values.

Formula:

    MAE = Average(|Actual - Predicted|)

For example:

Actual = 200

Predicted = 190

Absolute Error:

    |200 - 190| = 10

MAE calculates this error for all test observations and takes the average.

Lower MAE indicates better performance.

## Mean Squared Error

MSE calculates the average squared difference between actual and predicted values.

Formula:

    MSE = Average((Actual - Predicted)^2)

Large errors have a greater effect because the errors are squared.

Lower MSE indicates better performance.

## Root Mean Squared Error

RMSE is the square root of MSE.

Formula:

    RMSE = sqrt(MSE)

RMSE is useful because it is expressed in the same units as the target variable.

Lower RMSE indicates better performance.

## R² Score

R² measures how well the model explains the variation in the target variable.

An R² value closer to 1 generally indicates better performance.

For example:

    R² = 0.80

means the model explains approximately 80% of the variation in the target according to the evaluation.

## Actual vs Predicted Graph

An Actual vs Predicted graph is used to compare the real target values with the values predicted by the KNN regression model.

The graph contains:

- Actual target values
- Predicted target values

If the predictions are close to the actual values, the two lines should follow a similar pattern.

This gives a visual understanding of the regression model's performance.

## Regression Workflow

The complete workflow is:

    Diabetes Dataset
    → Data Loading
    → Data Inspection
    → Feature and Target Separation
    → Train-Test Split
    → Drop Sex
    → Feature Scaling
    → KNN Regressor
    → Parameter Grid
    → GridSearchCV
    → Best KNN Parameters
    → Predictions
    → MAE
    → MSE
    → RMSE
    → R²
    → Actual vs Predicted Graph

## Important KNN Parameters

### n_neighbors

Controls how many nearest observations are used.

Example:

    n_neighbors = 5

means the five nearest observations are considered.

### weights

Controls the importance of the selected neighbors.

Possible values:

    "uniform"
    "distance"

### metric

Controls how the distance between observations is calculated.

Possible values:

    "euclidean"
    "manhattan"
    "minkowski"

## Classification vs Regression

The main difference between KNN Classification and KNN Regression is how the selected neighbors are used.

For classification:

    Find K nearest points
    → Look at their classes
    → Majority voting
    → Predicted class

For regression:

    Find K nearest points
    → Look at their actual target values
    → Average or weighted average
    → Predicted numerical value

Classification predicts a category.

Regression predicts a numerical value.

## Libraries Used

### Pandas

Used for:

- Reading the CSV file
- Data inspection
- Data manipulation
- Creating results tables

    import pandas as pd

### NumPy

Used for numerical operations such as calculating RMSE.

    import numpy as np

### Matplotlib

Used for plotting the Actual vs Predicted graph.

    import matplotlib.pyplot as plt

### Scikit-learn

Used for:

- Train-test split
- StandardScaler
- KNeighborsRegressor
- GridSearchCV
- Regression metrics

## Project Structure

    KNN_Regression/
    ├── knn_regression.ipynb
    └── README.md

Dataset:

    Dataset/
    └── diabetes_original.csv

## Advantages of KNN Regression

- Simple to understand.
- Easy to implement.
- Does not require a specific mathematical relationship between features and target.
- Can model non-linear relationships.
- Can work well when similar observations have similar target values.
- Can be used for continuous target prediction.

## Disadvantages of KNN Regression

- Prediction can be slow for large datasets.
- Requires feature scaling.
- Sensitive to the choice of K.
- Sensitive to the distance metric.
- Sensitive to irrelevant features.
- Sensitive to noisy observations.
- Can perform poorly when the number of dimensions becomes very large.
- Requires storing the training data.

## What I Learned

Through this project, I learned:

- How KNN Regression works.
- Why distance calculation is necessary.
- How distance determines which observations become neighbors.
- How the actual target values of the nearest observations are used for prediction.
- The difference between uniform and distance weighting.
- How K affects KNN predictions.
- Why feature scaling is important.
- The difference between Euclidean, Manhattan, and Minkowski distance.
- How GridSearchCV can tune K, weights, and distance metric.
- How 5-fold cross-validation works.
- How to evaluate regression using MAE, MSE, RMSE, and R².
- How to compare actual and predicted values visually.

## Conclusion

KNN Regression is a simple supervised learning algorithm that predicts a continuous numerical value using the target values of nearby observations.

The Diabetes dataset was used for this regression problem.

The algorithm first calculates distances between the new observation and training observations. These distances are used to identify the K nearest observations.

After finding the nearest observations, their actual target values are used to produce the prediction.

With uniform weights, the selected target values contribute equally.

With distance weights, closer observations have more influence.

Different K values, weighting methods, and distance metrics were tested using GridSearchCV and 5-fold cross-validation.

Feature scaling was applied because KNN is a distance-based algorithm.

The final model was evaluated using MAE, MSE, RMSE, and R², along with an Actual vs Predicted graph.

This project helped in understanding how KNN can be used not only for classification but also for predicting continuous numerical values.