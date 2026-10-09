# Naive Bayes Classification

## Overview
Naive Bayes is a supervised machine learning algorithm used for classification problems. It is based on Bayes Theorem and conditional probability.
For this project, the Iris dataset is used to classify flowers into three different classes:
- Setosa
- Versicolor
- Virginica
Since the Iris dataset contains continuous numerical features, **Gaussian Naive Bayes** is used.

## Dataset
The Iris dataset contains 150 samples and 4 input features.
The features are:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width
The target contanis three classes:
- 0 = Setosa
- 1 = Versicolor
- 2 = Virginica

## What is Naive Bayes?
Naive Bayes is a classification algorithm based on Bayes Theorem.
It calculates the probability of a class given the available features.
The basic Bayes Theorem is:
**P(Class | Features) = P(Features | Class) × P(Class) / P(Features)**
The model calculates the probability for every possible class and selects the class with the highest probability.

## Conditional Probability
Conditional probability tells us the probability of an event when another event is already known.
For example, in this project we want to find the probability of a flower belonging to a particular class when its measurements are known.
For example:
`P(Setosa | Sepal Length, Sepal Width, Petal Length, Petal Width)`
The same calculation is performed for Versicolor and Virginica.
The class with the highest probability becomes the prediction.

## Why is it called "Naive"?
Naive Bayes makes an assumption that the input features are conditionally independent given the class.
For the Iris dataset, it assumes:
`P(Features | Class)`
can be represented approximately as:
`P(Sepal Length | Class) × P(Sepal Width | Class) × P(Petal Length | Class) × P(Petal Width | Class)`
In real-world data, features may not always be completely independent. However, the algorithm can still perform very well even with this assumption.

## Gaussian Naive Bayes
There are different types of Naive Bayes algorithms.
For this project, **GaussianNB** is used because the Iris features are continuous numerical values.
Gaussian Naive Bayes assumes that the values of each feature follow a Gaussian or normal distribution within each class.
The model estimates the mean and variance of each feature for every class and uses these values when calculating probabilities.

## Why Gaussian Naive Bayes for Iris?
Gaussian Naive Bayes is suitable for the Iris dataset because:
- The input features are numerical.
- The features are continuous measurements.
- There are multiple classes.
- The dataset is relatively small.
- GaussianNB works well with continuous features.
Unlike KNN, Gaussian Naive Bayes does not calculate distances between data points, so feature scaling is not required.

## Workflow
The project follows these steps:

1. Load the Iris dataset.
2. Inspect the dataset.
3. Check columns and missing values.
4. Separate features and target.
5. Check the target classes.
6. Split the data into training and testing sets.
7. Understand Bayes Theorem.
8. Understand conditional probability.
9. Understand the Naive independence assumption.
10. Apply Gaussian Naive Bayes.
11. Make predictions.
12. Tune `var_smoothing` using GridSearchCV.
13. Evaluate the final model.
14. Display the confusion matrix.

## Train-Test Split
The dataset is divided into:
- 80% training data
- 20% testing data
`random_state=42` is used so that the same split can be reproduced.
`stratify=y` is used to maintain the proportion of all three classes in both training and testing data.

## Model Training
The model used is:
```python
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)

Hyperparameter Tuning
GridSearchCV is used to find the best value of:
var_smoothing
5-fold cross-validation is used to compare the different values.
The value that produces the highest cross-validation accuracy is selected as the best parameter.

What is var_smoothing?
var_smoothing is a parameter of Gaussian Naive Bayes that helps with numerical stability when calculating feature variances.
A very small variance can cause problems while calculating probabilities. var_smoothing adds a small amount to the variances to make the calculations more stable.
GridSearchCV helps select a suitable value automatically.

Evaluation Metrics
The model is evaluated using:

Accuracy
Accuracy shows the percentage of predictions that are correct.
Precision
Precision tells us how many of the samples predicted as a particular class actually belong to that class.
Recall
Recall tells us how many samples of a particular class were correctly identified.
F1-Score
F1-score combines precision and recall into a single metric.
Support
Support represents the number of actual samples belonging to each class.

Confusion Matrix
The confusion matrix shows the actual classes against the predicted classes.
It helps identify which classes are being correctly classified and which classes are being confused with each other.

Results
The notebook displays:
* Best var_smoothing
* Cross-validation accuracy
* Test accuracy
* Classification report
* Confusion matrix
* Actual vs predicted classes

The exact results are generated when the notebook is executed.

Libraries Used
The following Python libraries are used:
* Pandas
* NumPy
* Matplotlib
* Scikit-learn

Conclusion
Naive Bayes provides a simple probability-based approach to classification.
In this project, Gaussian Naive Bayes was used on the Iris dataset because its input features are continuous numerical measurements.
The model uses Bayes Theorem and conditional probability to calculate the probability of each flower class. GridSearchCV was also used to select a suitable var_smoothing value.
This project helped in understanding how probability can be used directly to make machine learning classification predictions.