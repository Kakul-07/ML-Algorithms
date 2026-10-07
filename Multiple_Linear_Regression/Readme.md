*Multiple Linear Regression*

- About This Project
This is the first algorithm that I implemented in my ML Algorithms project.

In this project, I have used Multiple Linear Regression on the Diabetes dataset. The main idea is to understand how a regression model works when there is more than one input feature.
The model takes different measurements as input and tries to predict the disease progression value.
This is a regression problem because the output is a numerical value, not a category.

- Dataset
For this project, I am using the Diabetes dataset available through scikit-learn.
The dataset contains:
* 442 rows
* 10 input features
* 1 target column
* 11 columns in total
The 10 input features are:
- age
- sex 
- bmi - Body Mass Index
- bp - Blood Pressure
- s1 - Total Cholesterol
- s2 - Bad Cholesterol(LDL)
- s3 - Good Cholesterol(HDL)
- s4 - Total Cholesterol / HDL
- s5 - Triglyceride level
- s6 - Glucose level

The last column is:
target

What does the target mean?
The target is a quantitative measure of disease progression one year after the baseline measurements.
It is important to understand that this dataset is used for regression. The target does not mean:
0 = No diabetes
1 = Diabetes
Instead, the target is a numerical value.


Features in the Dataset

1. age
This represents an age-related measurement.
The values may look unusual because the features in this version of the dataset have been standardized.

2. sex
This represents a sex-related measurement.
It is stored as a numerical standardized value rather than normal labels such as male or female.

3. bmi
BMI stands for Body Mass Index.
It is related to body weight and height and is one of the health measurements used by the model.

4. bp
This represents the average blood pressure measurement.

5. s1
This represents a blood serum measurement related to total cholesterol.

6. s2
This represents another blood serum measurement related to LDL.

7. s3
This represents a blood serum measurement related to HDL.

8. s4
This represents a cholesterol-related ratio.

9. s5
This is another blood serum measurement.

10. s6
This represents a blood glucose-related measurement.

The s1 to s6 names are the short feature names used in the dataset.


What is Multiple Linear Regression?
Linear Regression is used to predict a numerical value.
When we use only one input feature, it is called Simple Linear Regression.
When we use multiple input features, it is called Multiple Linear Regression.
In this project, we use all 10 features together.
The general equation is:
y = b0 + b1x1 + b2x2 + b3x3 + ... + b10x10

Here:
y  = predicted target
b0 = intercept
b1...b10 = coefficients
x1...x10 = input features
For this project:
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

The model learns the relationship between the input features and the target from the training data.

Why Regression?
I used regression because the target in this dataset is a continuous numerical value.
For example, the model can predict values such as:
125.4
180.7
95.2
210.6
These are numerical predictions.

Project Workflow
The notebook follows the normal machine learning workflow.

Step 1: Import Libraries
The required Python libraries are imported first.
I used:
Pandas
NumPy
Matplotlib
Scikit-learn
Pandas is used for working with the dataset.
NumPy is used for numerical calculations.
Matplotlib is used for visualization.
Scikit-learn provides the Linear Regression model and evaluation metrics.

Step 2: Load the Dataset
The dataset is loaded using Pandas.
df = pd.read_csv("../Dataset/diabetes.csv")
The dataset is stored in the Dataset folder.

Step 3: Understand the Dataset
Before training the model, I check:
* Number of rows
* Number of columns
* Column names
* Data types
* Missing values
* Basic statistics
This helps to understand the dataset before applying the algorithm.

Step 4: Separate Features and Target
The target column is separated from the other columns.
X = df.drop("target", axis=1)
y = df["target"]
Here:
X = input features
y = output/target
So X contains the 10 features and y contains the target.

Step 5: Train-Test Split
The dataset is divided into two parts:
Training data → 80%
Testing data  → 20%
The training data is used to teach the model.
The testing data is kept separate so that we can check how the model performs on data it has not seen during training.

Step 6: Create the Model
The Multiple Linear Regression model is created using:
model = LinearRegression()


Step 7: Train the Model
The model is trained using:
model.fit(X_train, y_train)
During training, the model learns the coefficients for the different features.

Step 8: Make Predictions
After training, predictions are made using the test features:
y_pred = model.predict(X_test)
The predicted values are then compared with the actual values.

Model Evaluation
Simply training a model is not enough.
We also need to check how well the model is performing.
For this project, I used four metrics.

1. MAE
MAE stands for Mean Absolute Error.
It tells us the average absolute difference between the actual value and predicted value.
Formula:
MAE = (1/n) Σ |Actual - Predicted|
A smaller MAE means the predictions are closer to the actual values.

2. MSE
MSE stands for Mean Squared Error.
Formula:
MSE = (1/n) Σ (Actual - Predicted)²
The errors are squared, so larger errors have a bigger effect on the result.
A smaller MSE is better.

3. RMSE
RMSE stands for Root Mean Squared Error.
Formula:
RMSE = √MSE
RMSE is easier to interpret because it is on the same scale as the target.
A smaller RMSE generally means better predictions.

4. R² Score
R² is called the coefficient of determination.
It tells us how much of the variation in the target can be explained by the model.
The value is generally interpreted as:
Higher R² → better fit
Lower R² → weaker fit

Actual vs Predicted Graph
The notebook also creates a scatter plot between:
Actual values vs Predicted values

The purpose of this graph is to visually check how close the predictions are to the actual values.
If the predictions follow the same pattern as the actual values, the model is doing a better job.

Coefficients
The model also gives a coefficient for each feature.

Each feature gets its own coefficient.
The coefficients are used by the model in the regression equation.
Because the dataset features are standardized, the coefficients should be interpreted in the context of the standardized features rather than as changes in raw units.

Dataset
The Dataset folder contains the dataset used for the algorithm.

Notebook
The Notebook folder contains the complete implementation of Multiple Linear Regression.

README.md
This file explains the dataset, algorithm, workflow and evaluation.

Libraries Used
python3 -m pip install pandas numpy matplotlib scikit-learn jupyter

What I Learned From This Algorithm
Through this implementation, I learned the basic workflow of a regression problem.
The main things covered are:

* Loading a dataset
* Understanding the features
* Separating features and target
* Splitting data into training and testing sets
* Creating a Multiple Linear Regression model
* Training the model
* Making predictions
* Calculating MAE
* Calculating MSE
* Calculating RMSE
* Calculating R² score
* Understanding model coefficients
* Visualizing actual and predicted values


Conclusion

Multiple Linear Regression is a simple but important regression algorithm.

In this project, I used the 10 features of the Diabetes dataset to predict the numerical disease progression target.

The complete process starts from understanding the dataset and ends with evaluating the predictions using different regression metrics.
