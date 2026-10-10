# Random Forest Regression and Ensemble Comparison

## Introduction
This project predicts the continuous target in the original scikit-learn Diabetes dataset. It compares Decision Tree Regression, Random Forest Regression, Extra Trees Regression, and Gradient Boosting Regression.

## Dataset
Expected file: `Dataset/diabetes.csv`

## Algorithms
- **Decision Tree:** learns recursive feature-based splits and can overfit when too complex.
- **Random Forest:** trains multiple trees using bootstrap samples and randomized feature subsets, then averages their predictions.
- **Extra Trees:** trains a tree ensemble with additional randomness in split selection.
- **Gradient Boosting:** builds trees sequentially, with later trees correcting errors from earlier stages.

## Bagging and overfitting
Bagging trains models on bootstrap samples (samples drawn with replacement) and combines their predictions. Random Forest averages individual tree predictions, which often reduces variance compared with a single tree. This can improve generalization, but it does not guarantee a high score.

A large gap between training R² and test R² can indicate overfitting. Limiting depth and increasing minimum leaf size may reduce model complexity.

## Hyperparameter tuning
The notebook uses `RandomizedSearchCV` with five-fold cross-validation to tune Random Forest, Extra Trees, and Gradient Boosting. Randomized search samples a fixed number of combinations rather than testing every possible combination.

Key parameters:
- `n_estimators`: number of trees or boosting stages.
- `max_depth`: maximum tree depth.
- `min_samples_split`: minimum samples needed to split a node.
- `min_samples_leaf`: minimum samples in a leaf.
- `max_features`: features considered at a split.
- `max_samples`: samples drawn for each bootstrap tree.
- `learning_rate`: contribution of each Gradient Boosting tree.
- `subsample`: fraction of rows used per boosting stage.

Model selection uses training-set cross-validation. The test set is reserved for final evaluation.

## Evaluation metrics
- **MAE:** average absolute prediction error; lower is better.
- **MSE:** average squared error; lower is better.
- **RMSE:** square root of MSE, in the target's units; lower is better.
- **R²:** proportion of target variation explained; higher is generally better. Test R² may be negative.
- **CV Mean R²:** mean score across validation folds on training data.

## Visualizations
- Actual-versus-predicted scatter plot with a diagonal reference line.
- Test R² comparison chart.
- Feature-importance chart for the selected model.

Feature importance describes the model's split-based use of predictors; it does not prove causation.

## Workflow
1. Load and inspect the CSV.
2. Separate features and target.
3. Split into training and test sets.
4. Drop only `sex`.
5. Train baseline models.
6. Tune Random Forest, Extra Trees, and Gradient Boosting with RandomizedSearchCV.
7. Compare cross-validation and test metrics.
8. Select a final model using cross-validation.
9. Plot predictions and feature importance.

## Libraries
- pandas
- NumPy
- Matplotlib
- scikit-learn

## Advantages
- Tree ensembles can learn nonlinear relationships.
- Random Forest and Extra Trees often reduce variance compared with a single tree.
- RandomizedSearchCV explores hyperparameters with a fixed search budget.
- Cross-validation gives a more robust selection signal than one training split.

## Limitations
- The dataset is small, so scores can vary.
- Tuning does not guarantee improved test performance.
- Feature importance can be affected by correlated features.
- RandomizedSearchCV does not guarantee finding the global best parameter combination.

## Conclusion
This project demonstrates tree-based regression, bagging, boosting, overfitting, randomized hyperparameter tuning, cross-validation, and evaluation using MAE, MSE, RMSE, and R². Discuss the actual measured results rather than assuming tuning must improve the score.
