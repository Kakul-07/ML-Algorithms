# Polynomial Regression
This project is based on Polynomial Regression using the Diabetes dataset from scikit-learn.
The main purpose of this project is to understand how Polynomial Regression can be used when the relationship between the input features and the target is not completely linear. I also used Ridge Regression for regularization because adding polynomial features increases the complexity of the model and can lead to overfitting.

## Dataset
For this project, I used the Diabetes dataset provided by scikit-learn.
The dataset contains:
- 442 samples
- 10 input features
- 1 target column

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
The target represents a quantitative measure of disease progression one year after the baseline measurements.
The target is a continuous numerical value, so this is a regression problem.

## Why Polynomial Regression?
In Multiple Linear Regression, the model assumes a linear relationship between the input features and the target.
The basic form is:
y = b0 + b1x1 + b2x2 + ... + b10x10
But real-world relationships are not always completely linear.
Polynomial Regression helps the model capture possible non-linear relationships by creating additional features.
For example, if we have:
bmi
Polynomial Regression can create:
bmi²
It can also create interaction terms such as:
age × bmi
bmi × bp
age × bp
So the model gets more information from the original features.

## Polynomial Features
I used:
PolynomialFeatures(degree=2, include_bias=False)
The degree was set to 2.
This means the model creates:
- original features
- squared features
- interaction features
Since the original dataset has 10 features, degree 2 increases the number of features to 65.
This makes the model more flexible, but it also increases the possibility of overfitting.

## Overfitting
When we increase the polynomial degree, the number of features increases quickly.
For example:
Degree 1 → original features
Degree 2 → squared + interaction features
Degree 3 → even more features
Degree 4 → even more complexity
A very complex model can start learning the training data too closely instead of learning patterns that generalize to new data.

This is called overfitting.
To check this, I compared the training and testing R² scores for different polynomial degrees.
The results show that increasing the degree from 1 to 3 did not give a large improvement.
The training and testing scores also remained relatively close, so there was no strong overfitting in these experiments.
I selected degree 2 because it adds non-linear and interaction features without adding unnecessary complexity.

## Ridge Regression
Polynomial features increase the complexity of the model, so I used Ridge Regression to control this complexity.
Ridge Regression is a type of regularized linear regression.
The normal regression objective tries to minimize the prediction error:
Loss = Sum of squared errors
Ridge adds an extra penalty for large coefficients:
Loss = Prediction Error + alpha × Sum of squared coefficients
The alpha value controls the strength of regularization.
A larger alpha means stronger regularization.

## Why Ridge Regression?
The main reason for using Ridge Regression in this project is to reduce the risk of overfitting caused by polynomial features.
Ridge generally makes the coefficients smaller instead of allowing them to become unnecessarily large.

## Testing Different Alpha Values
I tested different values of alpha to see how regularization affects the model.
From these results, alpha = 0.01 gave the highest test R² among the tested values.
When alpha became very large, the model was regularized too strongly and its performance dropped considerably.
For example, at alpha = 10, the test R² dropped to 0.1613.
This shows that the model was becoming too restricted and was starting to underfit the data.
Therefore, from the tested values, I selected:
alpha = 0.01

## Pipeline
I used a scikit-learn Pipeline to combine Polynomial Feature creation and Ridge Regression.
The final structure is:
Pipeline([
    ("poly", PolynomialFeatures(
        degree=2,
        include_bias=False
    )),
    ("ridge", Ridge(alpha=0.01))
])

Using a Pipeline also makes the code cleaner because the transformation and model are treated as one complete model.

## Train-Test Split
The dataset was divided into training and testing data.
Here:
- 80% of the data is used for training.
- 20% is used for testing.
- random_state=42 keeps the split reproducible.
The model learns from the training data and is then tested on data that it has not seen during training.

## Model Training
The final model is trained using:
final_model.fit(X_train, y_train)
During training, the Pipeline first creates the polynomial features and then trains the Ridge Regression model on those features.

## Prediction
After training, predictions are made using:
final_pred = final_model.predict(X_test)
The model takes the test features and predicts the target value for each test sample.
The predictions are then compared with the actual values:
comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": final_pred
})
This helps me see how close the predictions are to the actual target values.

## Evaluation Metrics
I evaluated the model using four metrics.

### MAE - Mean Absolute Error
MAE measures the average absolute difference between actual and predicted values.
MAE = Average of |Actual - Predicted|
Lower MAE is better.

### MSE - Mean Squared Error
MSE calculates the average squared error.
MSE = Average of (Actual - Predicted)²
Lower MSE is better.
Because the errors are squared, larger errors have a bigger effect.

### RMSE - Root Mean Squared Error
RMSE is the square root of MSE.
RMSE = √MSE
Lower RMSE is better.
It is useful because it is expressed in the same units as the target.

### R² Score
R² tells us how much of the variation in the target is explained by the model.
A higher R² generally means a better fit.
R² should not be interpreted as percentage accuracy.

## Actual vs Predicted
I also plotted the actual and predicted values to visually compare the model's performance.
The actual values represent the real target values from the test dataset, while the predicted values are produced by the Polynomial Regression model.
If the predicted values are close to the actual values, the model is performing better.

## What I Learned From This Project
Through this project, I understood that Polynomial Regression is not a completely different type of regression model from Linear Regression.
Instead, Polynomial Regression first creates additional polynomial and interaction features and then uses a regression model to learn from them.
I also understood that increasing the polynomial degree does not automatically make the model better.
More features can make the model more complex and can lead to overfitting.
This is why regularization is important.
Ridge Regression helps control the model by penalizing large coefficients.
The alpha parameter controls how strong this regularization is.
In my experiments, a very high alpha caused underfitting, while alpha = 0.01 gave the best test R² among the values I tested.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

Main Scikit-learn components used:

- train_test_split
- PolynomialFeatures
- Ridge
- Pipeline
- mean_absolute_error
- mean_squared_error
- r2_score

## Conclusion
Polynomial Regression was used to extend the basic linear model by creating polynomial and interaction features.
For this project, I used degree 2 polynomial features because they provide additional flexibility without unnecessarily increasing the complexity of the model.
Since polynomial features can increase the risk of overfitting, I used Ridge Regression for regularization.
I compared different polynomial degrees and different alpha values to understand their effect on training and testing performance.
From the tested alpha values, alpha = 0.01 produced the highest test R² of approximately 0.4842.
Overall, this project helped me understand the relationship between model complexity, overfitting, regularization, polynomial features, and generalization to unseen data.