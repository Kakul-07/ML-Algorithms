# KNN Classification

## About the Project

This project implements the K-Nearest Neighbors (KNN) algorithm for classification using the Iris dataset.

KNN is a supervised machine learning algorithm that predicts the class of a new data point by looking at the classes of the nearest training data points.

The main concepts covered in this project are:

- K-Nearest Neighbors classification
- K-value selection
- Distance metrics
- Uniform and distance-based weights
- Feature scaling
- Train-test split
- Cross-validation
- GridSearchCV
- Hyperparameter tuning
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## Dataset

The Iris dataset is used for this classification problem.

The dataset contains measurements of iris flowers belonging to three different classes.

The three classes are:

- Setosa
- Versicolor
- Virginica

The input features are:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The target variable represents the flower class.

The dataset is stored in:

    Dataset/iris.csv

## Why Iris Dataset?

The Iris dataset is suitable for KNN classification because:

- The target is categorical.
- It contains three different classes.
- The input features are numerical.
- The dataset is relatively small.
- KNN can easily calculate distances between the observations.
- It is useful for understanding how nearest-neighbor classification works.

## What is KNN?

K-Nearest Neighbors is a supervised learning algorithm based on similarity between data points.

Instead of learning a fixed mathematical equation, KNN stores the training observations and uses them when making predictions.

For a new data point, KNN:

1. Calculates the distance between the new point and the training points.
2. Sorts the training points according to their distance.
3. Selects the K nearest points.
4. Looks at the classes of those K points.
5. Uses majority voting to determine the predicted class.

The basic process is:

New Data Point
→ Calculate Distances
→ Find K Nearest Neighbors
→ Check Their Classes
→ Majority Voting
→ Predicted Class

## Example of KNN Classification

Suppose K = 5.

The five nearest training points have the following classes:

    Setosa
    Setosa
    Versicolor
    Setosa
    Versicolor

Setosa occurs three times, while Versicolor occurs two times.

Therefore, the predicted class is:

    Setosa

This is called majority voting.

## K-Value

K represents the number of nearest neighbors used to make the prediction.

For example:

    K = 1

means only the closest point is considered.

    K = 5

means the five closest points are considered.

    K = 10

means the ten closest points are considered.

The value of K can have a large effect on the model.

## Small K

A small K makes the model more sensitive to individual observations.

For example:

    K = 1

The model looks only at the closest observation.

Advantages:

- Can capture local patterns.
- Can work well when nearby points are very similar.

Disadvantages:

- Sensitive to noise.
- Can overfit the training data.

## Large K

A large K considers more neighboring observations.

Advantages:

- Less sensitive to individual noisy points.
- Produces a smoother decision.

Disadvantages:

- Can ignore local patterns.
- Can underfit the data if K is too large.

Therefore, instead of manually selecting K, different K values are tested using GridSearchCV.

## Distance Metrics

Distance is an important part of KNN.

The algorithm needs to determine which training points are closest to a new data point.

Different distance metrics can calculate this distance in different ways.

The following metrics are tested in this project:

- Euclidean
- Manhattan
- Minkowski

## Euclidean Distance

Euclidean distance represents the straight-line distance between two points.

For two points:

    (x1, y1)

and:

    (x2, y2)

the distance is:

    d = sqrt((x2 - x1)^2 + (y2 - y1)^2)

It is one of the most commonly used distance metrics in KNN.

## Manhattan Distance

Manhattan distance calculates the sum of the absolute differences between the coordinates.

The formula is:

    d = |x2 - x1| + |y2 - y1|

It can be visualized as travelling along horizontal and vertical paths rather than directly between two points.

## Minkowski Distance

Minkowski distance is a generalized distance metric.

It can represent different distance calculations depending on its parameter.

It is also included in the hyperparameter search to determine whether it performs better for the dataset.

## Why Do We Need Distance?

KNN classification uses the classes of nearby observations, but first it needs to determine which observations are nearby.

For example, suppose there are 100 training observations.

A new observation is given to the model.

The model calculates distances such as:

    Point A → 1.2
    Point B → 1.8
    Point C → 2.1
    Point D → 5.7
    Point E → 8.3

If K = 3, the model selects:

    Point A
    Point B
    Point C

Only after finding these nearest points does KNN look at their classes.

Therefore:

Distance is used to find the nearest neighbors.

The classes of those neighbors are then used for voting.

## Feature Scaling

Feature scaling is important for KNN because KNN depends on distance.

Consider two features:

    Feature A → values from 1 to 10
    Feature B → values from 1000 to 10000

Without scaling, Feature B can have a much larger effect on the distance calculation.

This can affect which points are considered nearest.

Therefore, StandardScaler is used before applying KNN.

The scaler is fitted on the training data and then used to transform both training and testing data.

## StandardScaler

StandardScaler standardizes the numerical features so that they are on a comparable scale.

The general process is:

    Training Data
    → Fit StandardScaler
    → Transform Training Data
    → Transform Test Data
    → Apply KNN

The scaler is not fitted separately on the test data because that would introduce information from the test set into the preprocessing process.

## Uniform Weights

KNN supports different ways of assigning importance to neighbors.

With:

    weights = "uniform"

all K neighbors have equal importance.

For classification, each neighbor gets one vote.

For example:

    K = 5

Classes:

    Setosa
    Setosa
    Versicolor
    Setosa
    Virginica

Setosa receives three votes.

Therefore:

    Prediction = Setosa

## Distance Weights

With:

    weights = "distance"

closer neighbors receive more importance.

A neighbor that is very close to the new observation has a larger influence than a neighbor that is farther away.

This can be useful when the closest observations are more representative of the new observation.

## GridSearchCV

GridSearchCV is used to find suitable KNN hyperparameters.

The parameter grid is:

    param_grid = {
        "n_neighbors": range(1, 21),
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"]
    }

This tests:

20 different K values
×
2 weighting methods
×
3 distance metrics

Total:

    20 × 2 × 3 = 120 combinations

Each combination is evaluated using 5-fold cross-validation.

The combination with the best cross-validation accuracy is selected.

## Cross-Validation

Five-fold cross-validation is used during GridSearchCV.

The training data is divided into five parts.

Four parts are used for training and one part is used for validation.

This process is repeated five times so that each part is used as the validation set once.

The results are then combined to calculate the cross-validation score.

This gives a more reliable estimate than evaluating the model using only one validation split.

## Best Parameters

After GridSearchCV finishes, the following parameters can be obtained:

- Best K
- Best weight method
- Best distance metric
- Best cross-validation accuracy

The best parameters are then used to make predictions on the test data.

## Classification Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Support
- Confusion Matrix

## Accuracy

Accuracy measures how many predictions are correct out of all predictions.

Formula:

    Accuracy = Correct Predictions / Total Predictions

For example, if 95 out of 100 predictions are correct:

    Accuracy = 95 / 100

    Accuracy = 0.95

Therefore, the accuracy is 95%.

## Precision

Precision measures how many of the observations predicted as a particular class actually belong to that class.

High precision means that the model does not make many incorrect positive predictions for that class.

## Recall

Recall measures how many of the actual observations belonging to a class were correctly identified.

High recall means the model successfully finds most observations belonging to that class.

## F1-Score

F1-score combines precision and recall.

It is useful when both precision and recall are important.

A higher F1-score generally indicates better performance.

## Support

Support represents the number of actual observations belonging to each class in the test dataset.

## Confusion Matrix

The confusion matrix compares actual classes with predicted classes.

For the Iris dataset, the matrix represents:

- Setosa
- Versicolor
- Virginica

The diagonal values represent correctly classified observations.

The off-diagonal values represent incorrect classifications.

The confusion matrix helps identify which classes are being confused by the model.

## Classification Workflow

The complete workflow is:

    Iris Dataset
    → Data Loading
    → Data Inspection
    → Feature and Target Separation
    → Train-Test Split
    → Feature Scaling
    → Parameter Grid
    → GridSearchCV
    → Best KNN Parameters
    → Predictions
    → Accuracy
    → Classification Report
    → Confusion Matrix

## Libraries Used

### Pandas

Used for:

- Reading the CSV file
- Inspecting the dataset
- Manipulating the data
- Creating result tables

    import pandas as pd

### NumPy

Used for numerical operations.

    import numpy as np

### Matplotlib

Used for visualizing results.

    import matplotlib.pyplot as plt

### Scikit-learn

Used for:

- Train-test split
- StandardScaler
- KNeighborsClassifier
- GridSearchCV
- Classification metrics
- Confusion matrix

## Project Structure

    KNN_Classification/
    ├── knn_classification.ipynb
    └── README.md

Dataset:

    Dataset/
    └── iris.csv

## Advantages of KNN Classification

- Simple to understand.
- Easy to implement.
- Can handle multiple classes.
- Does not require a specific mathematical relationship between features and target.
- Can model non-linear decision boundaries.
- Works well on small and medium-sized datasets.

## Disadvantages of KNN Classification

- Prediction can become slow for large datasets.
- Requires feature scaling.
- Sensitive to the choice of K.
- Sensitive to the distance metric.
- Sensitive to irrelevant features.
- Can be affected by noisy observations.
- Performance can decrease with many dimensions.

## What I Learned

Through this project, I learned:

- How KNN classification works.
- How K is used to select neighbors.
- Why distance metrics are required.
- The difference between Euclidean, Manhattan, and Minkowski distance.
- Why feature scaling is important for distance-based algorithms.
- The difference between uniform and distance weighting.
- How GridSearchCV finds suitable hyperparameters.
- How 5-fold cross-validation works.
- How to evaluate a classification model using accuracy, precision, recall, F1-score, and confusion matrix.

## Conclusion

KNN Classification is a simple supervised learning algorithm that predicts the class of a new observation using the classes of its nearest neighbors.

The Iris dataset was used to demonstrate the algorithm.

Different K values, weighting methods, and distance metrics were tested using GridSearchCV and 5-fold cross-validation.

Feature scaling was also applied because KNN depends on distance calculations.

The final model was evaluated using classification metrics and a confusion matrix.

This project helped in understanding how KNN uses distance and neighboring observations to perform classification.