# Decision Tree Regression

## Introduction
Decision Tree Regression is a supervised machine learning algorithm used to predict continuous numerical values. It works by dividing the dataset into smaller groups using conditions based on input features. These splits are performed recursively until a stopping condition is reached.

Unlike classification trees, which predict categories, regression trees predict numerical values. Each leaf node generally predicts the mean target value of the training samples that reach that leaf.

In this project, I implemented Decision Tree Regression using the Diabetes dataset. The model predicts a quantitative measure of disease progression one year after the baseline measurements.

I first trained a baseline model, evaluated its performance, and examined how decision trees make recursive splits. After that, I used GridSearchCV to tune the model's hyperparameters and applied cost-complexity pruning to control tree complexity.

## Objectives

The main objectives of this project are:

- Understand how Decision Tree Regression works.
- Learn how recursive splitting divides the dataset.
- Understand how Mean Squared Error is used to evaluate regression splits.
- Train a baseline Decision Tree Regressor.
- Identify possible overfitting by comparing training and testing performance.
- Tune hyperparameters using GridSearchCV.
- Understand cost-complexity pruning.
- Evaluate predictions using MAE, MSE, RMSE and R².
- Compare the baseline model with the tuned model.
- Examine feature importance.

## Dataset Used

For this project, I used the Diabetes dataset 

### Dataset Information

| Property | Description |
| Dataset name | Diabetes Dataset |
| Total samples | 442 |
| Original input features | 10 |
| Target column | `target` |
| Problem type | Regression |
| Missing values | Check the dataset during preprocessing |
| Data type | Numerical |

### Features Used

The original dataset contains the following input features:

| Feature | Description |
|---|---|
| `age` | Age-related measurement |
| `sex` | Encoded sex-related variable |
| `bmi` | Body mass index |
| `bp` | Average blood pressure |
| `s1` | Total serum cholesterol |
| `s2` | Low-density lipoprotein-related measurement |
| `s3` | High-density lipoprotein-related measurement |
| `s4` | Total cholesterol to HDL ratio-related measurement |
| `s5` | Log-transformed serum triglyceride-related measurement |
| `s6` | Blood glucose-related measurement |
| `target` | Quantitative disease progression measure |

The exact definitions of the original dataset's biochemical measurements are more specific than their short column names suggest.

In this project, I removed only the `sex` feature and retained the remaining nine features for training.

## What Is Decision Tree Regression?

Decision Tree Regression is a supervised learning algorithm that learns decision rules from labelled training data.

It divides the feature space into smaller regions. Each region corresponds to a leaf node, which provides a numerical prediction.

For example, the tree might learn a condition based on BMI and send samples to different branches depending on whether their BMI is above or below a selected threshold.

The actual splitting conditions are learned from the training data.

Once the tree has been trained, a new sample follows the decision rules until it reaches a leaf node. The value stored in that leaf is used as the prediction.

## Recursive Splitting

Recursive splitting is the process of repeatedly dividing the dataset into smaller groups.

The algorithm begins with the complete training dataset at the root node.

It then performs the following steps:

1. Examines the available features.
2. Considers possible split thresholds.
3. Evaluates the prediction error produced by candidate splits.
4. Selects a split that reduces the error.
5. Divides the samples into child nodes.
6. Repeats the process for the resulting nodes until a stopping condition is reached.

This process creates a tree-like structure containing decision nodes, branches and leaf nodes.

A leaf node predicts the mean target value of the training samples that reach it when using the standard squared-error regression criterion.

Recursive splitting allows the model to represent nonlinear relationships and interactions between features.

## Splitting Criterion: Mean Squared Error

In Decision Tree Regression, the algorithm needs a way to decide whether a split improves the quality of the predictions.

One common criterion is squared error.

Mean Squared Error measures the average squared difference between actual and predicted values.

MSE = (1 / n) * sum((y_i - y_pred_i)^2)

Here:

- `n` is the number of samples.
- `y_i` is the actual target value.
- `y_pred_i` is the predicted value.

The squared-error criterion favours splits that reduce the variation of target values within the resulting groups.

The algorithm compares candidate splits and selects one that provides a suitable reduction in squared error.

### What About Entropy?

Entropy is commonly used when discussing Decision Tree Classification. It measures how mixed the classes are within a node.

For classification, entropy can be used to calculate information gain when selecting splits.

However, this project uses Decision Tree Regression, where the target is continuous. The model therefore uses `squared_error` as its main splitting criterion rather than classification entropy.

The `friedman_mse` criterion is another regression splitting option that can be explored during hyperparameter tuning.

## Training and Testing Split

Before training the model, I divided the dataset into training and testing sets.

The split uses:

- 80% of the data for training.
- 20% of the data for testing.
- `random_state=42` for reproducibility.

For a dataset containing 442 samples, this produces approximately 353 training samples and 89 testing samples.

The training set is used to learn the tree's decision rules. The test set is kept separate until final evaluation to measure how well the trained model performs on unseen samples.

I also examined feature correlations using the training data. Correlation provides information about the direction and strength of linear relationships, but a low correlation does not automatically mean a feature is unimportant to a decision tree.

## Baseline Decision Tree Regressor

I first trained a basic Decision Tree Regressor using the `DecisionTreeRegressor` class from scikit-learn.

The baseline model used the `squared_error` splitting criterion and `random_state=42`.

At this stage, I did not restrict the maximum tree depth or apply additional pruning constraints.

This provided a starting point for understanding the model's performance before hyperparameter tuning.

After training, I used the model to predict the target values of the test samples.

## Overfitting in Decision Tree Regression

Overfitting occurs when a model learns the training data too closely and does not generalize well to unseen samples.

Decision trees can overfit when they grow too deep and create many branches that capture small details or noise in the training data.

A common indication of overfitting is very high training performance combined with noticeably weaker testing performance.

For regression, I compared the training and testing R² scores to examine the model's generalization.

A large difference between these scores can indicate overfitting, although the difference should be considered alongside the actual values of both scores.

### How to Control Overfitting

Several hyperparameters can help control decision tree complexity:

- `max_depth`: Limits the maximum depth of the tree.
- `min_samples_split`: Specifies the minimum number of samples required to split an internal node.
- `min_samples_leaf`: Specifies the minimum number of samples required in a leaf.
- `ccp_alpha`: Controls the strength of cost-complexity pruning.

These parameters prevent the tree from creating unnecessary branches and help balance model complexity with predictive performance.

A tree that is too restricted can also underfit the data, so the parameters need to be selected carefully.

## Hyperparameter Tuning Using GridSearchCV

After training the baseline model, I used GridSearchCV to search for suitable hyperparameters.

GridSearchCV evaluates different parameter combinations using cross-validation and selects the combination with the best average validation score.

In this project, I used five-fold cross-validation with R² as the scoring metric.

The parameter grid includes the following settings:

| Hyperparameter | Values Tested | Purpose |
|---|---|---|
| `criterion` | `squared_error`, `friedman_mse` | Determines how regression splits are evaluated |
| `max_depth` | `2`, `3`, `4`, `5`, `6`, `8`, `None` | Controls the maximum depth of the tree |
| `min_samples_split` | `2`, `5`, `10` | Controls when a node can be split |
| `min_samples_leaf` | `1`, `2`, `4` | Controls the minimum number of samples in each leaf |
| `ccp_alpha` | `0.0`, `0.001`, `0.01`, `0.1` | Controls cost-complexity pruning |

The grid contains 504 parameter combinations. With five-fold cross-validation, it can require up to 2,520 model fits.

The best parameters depend on the cross-validation results, so the selected values should be taken from the actual notebook output.

### Why Use GridSearchCV?

Instead of manually choosing hyperparameters, GridSearchCV compares several combinations systematically.

It helps identify settings that achieve good validation performance while controlling the tree's complexity.

The `cv=5` setting divides the training data into five folds. In each round, four folds are used for fitting and the remaining fold is used for validation. This process is repeated so that every fold serves as the validation fold once.

The `scoring="r2"` setting tells GridSearchCV to select the model with the highest mean cross-validation R² score.

The `n_jobs=-1` setting allows the search to use available CPU processors for parallel execution.

The `refit=True` setting fits the selected best estimator again using the complete training dataset after the search finishes.

## Cost-Complexity Pruning

Pruning is used to control the complexity of a decision tree by removing branches that do not justify their additional complexity.

A large tree may fit the training samples very closely, but some of its branches might not improve performance on unseen data.

Cost-complexity pruning balances the model's fit against the complexity of the tree.

The `ccp_alpha` hyperparameter controls the strength of pruning.

- A value of `0.0` applies no additional cost-complexity pruning.
- A small positive value can remove branches that contribute relatively little.
- A larger value generally produces a simpler tree.
- An excessively large value can cause underfitting.

In this project, `ccp_alpha` is included in GridSearchCV so that different pruning strengths can be evaluated through cross-validation.

## Model Evaluation

After selecting the best model, I evaluated its predictions on the held-out test dataset.

The evaluation uses four regression metrics:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

### Mean Absolute Error (MAE)

MAE measures the average absolute difference between actual and predicted values.

MAE = (1 / n) * sum(|y_i - y_pred_i|)

A lower MAE indicates that the predictions are closer to the actual target values on average.

### Mean Squared Error (MSE)

MSE measures the average squared difference between actual and predicted values.

MSE = (1 / n) * sum((y_i - y_pred_i)^2)

Because the errors are squared, larger prediction errors have a greater effect on this metric.

A lower MSE generally indicates better predictions.

### Root Mean Squared Error (RMSE)

RMSE is the square root of MSE.

RMSE = sqrt(MSE)

It is expressed in the same units as the target variable, which makes it easier to interpret than MSE.

A lower RMSE generally indicates smaller prediction errors.

### R² Score

R² measures how well the model explains variation in the target compared with a baseline that predicts the mean target value.

An R² value of 1 indicates perfect predictions. A value of 0 indicates performance equivalent to the mean-prediction baseline under the standard R² definition. A negative value means the model performs worse than that baseline on the evaluated data.

A higher R² generally indicates better predictive performance.

## Baseline Model vs Tuned Model

I compared the baseline model with the tuned model using the same held-out test dataset.

The comparison includes:

- MAE
- MSE
- RMSE
- R² Score

This comparison helps determine whether hyperparameter tuning improved the model's performance.

For MAE, MSE and RMSE, lower values are generally better. For R², a higher value is generally better.

Hyperparameter tuning does not guarantee that every test metric will improve. GridSearchCV selects parameters using cross-validation, while the final test set provides an independent evaluation of the selected model.

The actual scores should be taken from the notebook output rather than assumed in advance.

## Actual vs Predicted Graph

I plotted the actual target values against the model's predictions for the test samples.

This visualization helps compare the predictions with the real target values.

If the predicted values are close to the actual values, the model is making relatively accurate predictions for those samples. Large differences indicate prediction errors.

The graph provides a visual comparison, while MAE, MSE, RMSE and R² provide numerical measures of model performance.

## Feature Importance

Decision Tree Regressor provides feature importance values based on how much each feature contributes to reducing the splitting criterion throughout the tree.

In this project, I examined the feature importance values of the tuned model.

These values help identify which features contributed most to the model's learned decision rules.

Feature importance does not prove that a feature causes a change in disease progression. It describes how the trained tree used the available features to make predictions.

## Libraries Used

The project uses the following Python libraries:

- **Pandas:** For loading and exploring the dataset.
- **NumPy:** For numerical operations, including calculating RMSE.
- **Matplotlib:** For visualizing the decision tree, predictions and feature importance.
- **Scikit-learn:** For splitting the dataset, training the regression tree, tuning hyperparameters and evaluating predictions.

## Advantages of Decision Tree Regression

- Can model nonlinear relationships.
- Does not require feature scaling for its standard splitting procedure.
- Can learn interactions between input features.
- Is relatively easy to understand and visualize.
- Can predict continuous numerical values.
- Supports hyperparameter tuning and pruning.
- Provides feature importance estimates.

## Limitations of Decision Tree Regression

- Deep trees can overfit the training data.
- Small changes in the training data can produce a different tree.
- A tree that is too restricted can underfit.
- Greedy splitting does not guarantee a globally optimal tree.
- Individual decision trees may not generalize as well as ensemble methods on some datasets.
- Feature importance does not establish causal relationships.

## Conclusion

In this project, I implemented Decision Tree Regression using the Diabetes dataset.

I trained a baseline model, examined recursive splitting and evaluated the initial predictions using regression metrics. I then used GridSearchCV to tune the splitting criterion, maximum depth, minimum sample requirements and cost-complexity pruning strength.

Finally, I evaluated the tuned model on the test dataset, compared it with the baseline model and examined feature importance.

This project helped me understand how regression trees make numerical predictions, how overfitting occurs, and how hyperparameter tuning and pruning can help control tree complexity.