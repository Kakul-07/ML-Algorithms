# Multiple Linear Regression

## About the Project

This project is an implementation of **Multiple Linear Regression** using the original Diabetes dataset provided by Scikit-learn.

The main purpose of this project is to understand how multiple input features can be used together to predict a continuous numerical value.

The target variable in this dataset represents the **disease progression one year after the baseline measurements**.

In this project, I followed the complete machine learning workflow starting from understanding the dataset, separating the features and target, splitting the data, checking correlation, removing the `sex` feature, training the model, making predictions, evaluating the model and checking its performance using 5-fold cross-validation.

---

## Dataset

The dataset used in this project is the original **Diabetes dataset from Scikit-learn**.

It contains:

- 442 samples
- 10 input features
- 1 target variable

The dataset is stored in:

```text
Dataset/diabetes_original.csv
```

### Features in the Dataset

The original input features are:

```text
age
sex
bmi
bp
s1
s2
s3
s4
s5
s6
```

The target column is:

```text
target
```

The features represent different measurements related to the patients, such as age, body mass index, blood pressure, cholesterol-related measurements, HDL, triglycerides-related measurement and blood glucose-related measurement.

The `target` is a continuous numerical value, which makes this a **regression problem**.

---

## Algorithm Used

### Multiple Linear Regression

Multiple Linear Regression is used when we want to predict one continuous value using multiple input features.

The general equation of Multiple Linear Regression is:

```text
y = b0 + b1x1 + b2x2 + ... + bnxn
```

Where:

- `y` = predicted value
- `b0` = intercept
- `b1, b2, ... bn` = coefficients
- `x1, x2, ... xn` = input features

The model learns the coefficients from the training data and uses them to make predictions.

---
```
---

## 1. Loading the Dataset

The dataset is loaded using Pandas.

```python
df = pd.read_csv("../Dataset/diabetes_original.csv")
```

After loading the dataset, the first few rows are checked to understand how the data looks.

Basic information about the dataset is also checked before starting the model training process.

---

## 2. Understanding the Dataset

Before training the model, I checked the basic structure of the dataset.

The following things were checked:

- Number of rows
- Number of columns
- Column names
- Data types
- Missing values
- Statistical summary

Some of the commands used are:

```python
df.shape
df.columns
df.info()
df.isnull().sum()
df.describe()
```

This step helps in understanding the dataset and checking whether there are any missing or unusual values that need to be handled.

---

## 3. Separating Features and Target

The `target` column is the value that we want the model to predict.

The input features and target are separated using:

```python
X = df.drop("target", axis=1)
y = df["target"]
```

Here:

- `X` contains the input features.
- `y` contains the target values.

Initially, `X` contains all 10 input features.

---

## 4. Train-Test Split

The dataset is divided into training and testing data.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
```

I used:

- 80% data for training
- 20% data for testing

The training data is used to teach the model, while the testing data is kept separate and used to check how well the model performs on unseen data.

`random_state=42` is used so that the same train-test split can be reproduced.

---

## 5. Checking Correlation

Before removing the feature, I checked the correlation between the input features and the target.

The correlation is checked using the training data:

```python
correlation = X_train.copy()
correlation["target"] = y_train

print(correlation.corr()["target"].sort_values(ascending=False))
```

Correlation gives an idea about the relationship between a feature and the target.

A positive correlation means that the feature and target tend to increase together.

A negative correlation means that when one value increases, the other tends to decrease.

A negative correlation does not mean that a feature is useless. A feature can still be important even if its correlation with the target is negative.

The correlation was checked before feature removal so that the relationship of the features with the target could be understood first.

---

## 6. Feature Selection

After checking the correlation, the `sex` feature was removed from both the training and testing data.

```python
X_train = X_train.drop(columns=["sex"])
X_test = X_test.drop(columns=["sex"])
```

This removes only the `sex` feature.

The remaining features used for training are:

```text
age
bmi
bp
s1
s2
s3
s4
s5
s6
```

Therefore, the model is trained using **9 features**.

The `target` is not included in the input features because it is the value that the model needs to predict.

---

## 7. Training the Multiple Linear Regression Model

The Multiple Linear Regression model is created using Scikit-learn.

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

The `fit()` function trains the model using the training features and training target.

During training, the model learns:

- An intercept
- A coefficient for each input feature

These learned values are later used to make predictions.

---

## 8. Checking Model Coefficients

The coefficients learned by the model are displayed using a DataFrame.

```python
coefficients = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": model.coef_
})

coefficients
```

The coefficient tells us the direction of the relationship between a feature and the predicted target while the other features are considered by the model.

For example:

- A positive coefficient means an increase in that feature is associated with an increase in the prediction.
- A negative coefficient means an increase in that feature is associated with a decrease in the prediction.

The coefficient values should be interpreted carefully because the features are not all measured on the same scale.

---

## 9. Checking the Intercept

The intercept learned by the model is also checked.

```python
print("Intercept:", model.intercept_)
```

The intercept is the predicted value when all input features are zero according to the regression equation.

---

## 10. Making Predictions

After training the model, predictions are made using the test data.

```python
y_pred = model.predict(X_test)
```

The model uses the 9 selected features from `X_test` and produces a predicted value for every test sample.

---

## 11. Comparing Actual and Predicted Values

The actual and predicted values are placed together in a DataFrame.

```python
comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

comparison.head(10)
```

This makes it easier to compare what the actual target value was with what the model predicted.

---

## 12. Model Evaluation

The model is evaluated using four common regression metrics:

- MAE
- MSE
- RMSE
- R² Score

The metrics are calculated using:

```python
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)
```

---

## 13. Mean Absolute Error (MAE)

Mean Absolute Error measures the average absolute difference between the actual and predicted values.

The formula is:

```text
MAE = average(|Actual - Predicted|)
```

A lower MAE means that the predictions are, on average, closer to the actual values.

---

## 14. Mean Squared Error (MSE)

Mean Squared Error calculates the average of the squared prediction errors.

The formula is:

```text
MSE = average((Actual - Predicted)²)
```

Because the errors are squared, larger errors have a greater effect on MSE.

A lower MSE indicates better performance.

---

## 15. Root Mean Squared Error (RMSE)

Root Mean Squared Error is calculated by taking the square root of MSE.

```text
RMSE = √MSE
```

RMSE is useful because it is expressed in the same unit as the target variable.

A lower RMSE indicates that the predictions are closer to the actual values.

---

## 16. R² Score

R², or the coefficient of determination, tells us how much of the variation in the target variable is explained by the model.

The formula is:

```text
R² = 1 - (Sum of Squared Errors / Total Sum of Squares)
```

An R² value closer to 1 generally indicates better model performance.

---

## 17. Results Table

The evaluation metrics are also stored in a simple table:

```python
results = pd.DataFrame({
    "Metric": ["MAE", "MSE", "RMSE", "R²"],
    "Value": [mae, mse, rmse, r2]
})

results
```

This gives all the main test-set evaluation results in one place.

---

## 18. 5-Fold Cross-Validation

Along with the normal train-test evaluation, 5-fold cross-validation is also performed.

```python
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    LinearRegression(),
    X.drop(columns=["sex"]),
    y,
    cv=kf,
    scoring="r2"
)

print("5-Fold Cross-Validation R² Scores:")
print(cv_scores)

print("Mean Cross-Validation R²:", cv_scores.mean())
```

In 5-fold cross-validation, the dataset is divided into five parts.

For each run:

- Four parts are used for training.
- One part is used for validation.

This process is repeated five times so that every part of the dataset gets a chance to be used for validation.

The five R² scores are then averaged.

The mean cross-validation R² gives a better idea of how consistently the model performs across different parts of the dataset.

---

## 19. Visualization 1 - Actual vs Predicted Values

A simple line graph is used to compare the actual and predicted values.

```python
plt.figure(figsize=(8, 5))

plt.plot(y_test.values, label="Actual")
plt.plot(y_pred, label="Predicted")

plt.xlabel("Test Samples")
plt.ylabel("Target Value")
plt.title("Actual vs Predicted Values")
plt.legend()
plt.show()
```

The graph contains two lines:

- `Actual` represents the real target values.
- `Predicted` represents the values predicted by the model.

If the two lines are close to each other, it means that the predictions are closer to the actual values.

This graph gives a simple visual idea of how the model is performing.

---


## What I Learned

Through this project, I learned:

- How Multiple Linear Regression works.
- How to work with a real-world dataset.
- How to separate input features and the target.
- How to split data into training and testing sets.
- How to check correlations between features and the target.
- How correlation can be positive or negative.
- How to remove an unwanted feature from the dataset.
- How to train a Multiple Linear Regression model.
- How regression coefficients work.
- How to make predictions using a trained model.
- How to compare actual and predicted values.
- How MAE, MSE, RMSE and R² are calculated and interpreted.
- How 5-fold cross-validation works.
- How to visualize model predictions.
- How to visualize the coefficients learned by the model.

---

## Conclusion

This project helped me understand the complete workflow of Multiple Linear Regression using a real dataset.

I started by exploring the dataset and separating the input features from the target. After splitting the data into training and testing sets, I checked the correlation between the features and the target.

Based on the feature selection step, I removed the `sex` feature and used the remaining 9 features for training the model.

After training, I made predictions on the test data and evaluated the model using MAE, MSE, RMSE and R². I also used 5-fold cross-validation to check the consistency of the model's performance.

Finally, I used simple graphs to compare actual and predicted values and to understand the coefficients learned by the model.

Overall, this project gave me a practical understanding of how Multiple Linear Regression works from data preparation to model training, evaluation and visualization.
