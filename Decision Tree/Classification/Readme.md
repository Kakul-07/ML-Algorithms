# Decision Tree Classifier

## Introduction
A Decision Tree Classifier is a supervised machine learning algorithm used to classify data into different categories. It works by learning a set of decision rules from the training data and using those rules to predict the class of new observations.
I implemented this algorithm using the Iris dataset. The main purpose of this project is to understand how decision trees split data, how they decide which feature to use at each split, and how to control overfitting using hyperparameters and pruning.

## Objectives
The main objectives of this project are:
- Understand how a Decision Tree Classifier works.
- Learn how a decision tree splits the dataset into smaller groups.
- Understand Gini impurity and entropy.
- Train a baseline classification model.
- Identify overfitting by comparing training and testing accuracy.
- Tune hyperparameters using GridSearchCV.
- Apply cost-complexity pruning to control tree complexity.
- Evaluate the model using accuracy, precision, recall and F1-score.
- Interpret the confusion matrix.
- Understand feature importance.

## Dataset Used
The Iris dataset is commonly used for classification problems. It contains measurements of three different species of Iris flowers. Each flower belongs to one of the following classes:
- Setosa
- Versicolor
- Virginic
The dataset contains 150 samples, four input features and one target column.

### Features Used
The model uses all four numerical features to classify the flowers.
 Feature | Description
 `sepal length (cm)` 
 `sepal width (cm)`
 `petal length (cm)` 
 `petal width (cm)` 
 `target` 
The target column contains three class labels:
- `0` – Setosa
- `1` – Versicolor
- `2` – Virginica
The exact mapping should match the encoding used in the dataset.

## What Is a Decision Tree Classifier?
A Decision Tree Classifier is a supervised learning algorithm that learns decision rules from labelled training data.
It represents these rules in a tree-like structure consisting of:
- **Root node:** The first node, where the initial split takes place.
- **Decision nodes:** Internal nodes that apply conditions to feature values.
- **Branches:** Paths followed depending on whether a condition is satisfied.
- **Leaf nodes:** Final nodes that provide the predicted class.
The algorithm continues splitting nodes until a stopping condition is reached.
When a new flower is given to the trained model, its measurements pass through the learned rules until a leaf node provides the predicted class.

## How Does Decision Tree Splitting Work?
The main purpose of splitting is to divide the data into groups that contain samples belonging to similar classes.
At each decision node, the algorithm considers different features and possible thresholds. It evaluates how well each candidate split separates the classes and chooses a split according to the selected criterion.
For example, a split might look like this:
`petal length (cm) <= 2.45`
Samples satisfying the condition follow one branch, while the remaining samples follow the other branch.
The selected feature and threshold depend on the training data. The example above is illustrative and is not guaranteed to be the exact first split learned by every model.
The process continues recursively until a stopping condition is met, such as reaching the maximum depth or having too few samples to split a node.

## Gini Impurity
Gini impurity measures how mixed the classes are within a node.
Its formula is:
Gini = 1 - sum(p_i²)
Here, `p_i` represents the proportion of samples belonging to class `i` in the node.
A Gini impurity of zero means that all samples in the node belong to the same class.
When `criterion="gini"`, the decision tree evaluates candidate splits using the reduction in impurity.

## Entropy
Entropy is another measure of impurity used by decision trees.
Its formula is:
Entropy = -sum(p_i * log2(p_i))
Here, `p_i` represents the proportion of samples belonging to class `i`.
A node containing only one class has zero entropy. A node containing a mixture of classes has higher entropy.
When `criterion="entropy"`, the algorithm evaluates splits using information gain, which measures the reduction in entropy after a split.
Both Gini impurity and entropy can be used to build a classification tree. Their results may differ because they evaluate impurity differently.

## Training and Testing Split
Before training the model, I divided the dataset into training and testing sets.
I used:
- 80% of the data for training.
- 20% of the data for testing.
- `random_state=42` to make the split reproducible.
- `stratify=y` to preserve class proportions in both sets.
This produces 120 training samples and 30 testing samples for the Iris dataset.
The training set is used to learn the decision rules. The test set is kept separate so that the final evaluation measures performance on samples the model did not use during training.

## Baseline Decision Tree Classifier
I first trained a basic Decision Tree Classifier using the following model:
`DecisionTreeClassifier(random_state=42)`
At this stage, I did not specify a maximum tree depth or other complexity constraints.
The model was trained using the training data and then used to predict the classes of the test samples.
This baseline provides a reference point for evaluating whether hyperparameter tuning improves the model's performance.

## Overfitting in Decision Trees
Overfitting occurs when a model learns the training data too closely, including patterns that may not generalize to new observations.
Decision trees can overfit because they can continue creating branches until they form very specific rules for the training samples.
A common indication of overfitting is very high training accuracy combined with noticeably lower testing accuracy.
However, a difference between training and testing accuracy does not automatically prove overfitting. The size of the difference, the absolute scores and the characteristics of the dataset should also be considered.

### How Can Overfitting Be Controlled?
Several decision tree hyperparameters help control tree complexity:
- `max_depth`: Limits the maximum depth of the tree.
- `min_samples_split`: Sets the minimum number of samples required to split an internal node.
- `min_samples_leaf`: Sets the minimum number of samples required in a leaf node.
- `max_leaf_nodes`: Limits the number of leaf nodes.
- `ccp_alpha`: Controls cost-complexity pruning.
These parameters help prevent the tree from creating unnecessary branches.
The aim is not simply to make the tree smaller. The aim is to find a tree that performs well on unseen data.

## Hyperparameter Tuning Using GridSearchCV
After training the baseline model, I used GridSearchCV to search for suitable hyperparameters.
GridSearchCV trains and evaluates models using different parameter combinations and selects the combination with the best average cross-validation score.
In this project, I used five-fold cross-validation and accuracy as the scoring metric.
The parameter grid contains 270 combinations. With five-fold cross-validation, it can require up to 1,350 model fits.
The best combination depends on the dataset and the validation scores, so the selected values should be taken from the actual GridSearchCV results.

### Why Use GridSearchCV?
Instead of manually choosing one set of hyperparameters, GridSearchCV evaluates multiple combinations systematically.
It helps identify settings that provide good validation performance while controlling the complexity of the tree.
The `n_jobs=-1` setting allows the search to use available CPU processors for parallel execution.
The `refit=True` setting ensures that, after selecting the best parameters, the model is fitted again on the complete training dataset.

## Cost-Complexity Pruning
Pruning is used to reduce unnecessary complexity in a decision tree.
A fully grown tree may contain branches that fit specific training samples but do not improve generalization.
Cost-complexity pruning balances the tree's fit against its complexity. The `ccp_alpha` hyperparameter controls the strength of pruning.
- A smaller `ccp_alpha` generally permits a larger tree.
- A larger `ccp_alpha` generally removes more branches.
- An excessively large value can make the tree too simple and cause underfitting.
In this project, `ccp_alpha` is included in GridSearchCV so that different pruning strengths can be evaluated using cross-validation.

## Model Evaluation
After selecting the best model, I evaluated it on the test dataset.
The evaluation uses accuracy, precision, recall, F1-score and the confusion matrix.

### Accuracy
Accuracy measures the proportion of test samples classified correctly.
Accuracy = Correct Predictions / Total Predictions
It provides an overall measure of classification performance.

### Precision
Precision measures the proportion of predicted samples for a class that actually belong to that class.
Precision = True Positives / (True Positives + False Positives)
It helps show how reliable the model's predictions are for each class.

### Recall
Recall measures the proportion of actual samples of a class that the model identifies correctly.
Recall = True Positives / (True Positives + False Negatives)
It helps show whether the model misses samples belonging to a particular class.

### F1-Score
The F1-score is the harmonic mean of precision and recall.
F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
It provides a combined measure of precision and recall.

### Support
Support is the number of actual test samples belonging to each class.
In the Iris test set, there are 10 samples per class when the split is performed using the specified stratified 80:20 split.

## Confusion Matrix
The confusion matrix compares the actual class labels with the predicted class labels.
For this project, the matrix contains three rows and three columns because there are three flower classes.
- Rows represent actual classes.
- Columns represent predicted classes.
- Diagonal values represent correct predictions.
- Off-diagonal values represent incorrect predictions.
The confusion matrix helps identify which classes the model classifies correctly and which classes it confuses.

## Baseline Model vs Tuned Model
I evaluated both the baseline model and the tuned model on the same test set.
The comparison includes:
- Baseline training accuracy.
- Baseline testing accuracy.
- Tuned training accuracy.
- Tuned testing accuracy.
- Best cross-validation accuracy.
This comparison helps determine whether the hyperparameter search improved the model's performance and whether the final tree appears to generalize well.
The best cross-validation accuracy and test accuracy are different measurements because they are calculated on different data.
A model with the highest training accuracy is not necessarily the best model. The final goal is to achieve reliable performance on unseen samples.
The actual scores should be taken from the notebook output rather than assumed in advance.

## Feature Importance
Decision trees can estimate the importance of input features based on how much they contribute to reducing impurity across the tree.
In this project, I examined the feature importance values of the tuned model.
The importance values can help identify which flower measurements contributed most to the learned splitting rules.
Feature importance does not prove that a feature causes a particular flower class. It describes how the trained tree used the available features.

## Libraries Used
The project uses the following Python libraries:
- **Pandas:** For loading and exploring the dataset.
- **NumPy:** For numerical operations.
- **Matplotlib:** For visualizing the decision tree, confusion matrix and feature importance.
- **Scikit-learn:** For splitting the data, training the classifier, tuning hyperparameters and evaluating predictions.
`../Dataset/iris.csv`, so the relative path assumes that the notebook is inside the `Decision_Tree_Classifier` folder.

## Advantages of Decision Tree Classifier
- Easy to understand and interpret.
- Can model nonlinear decision boundaries.
- Does not require feature scaling for its standard splitting procedure.
- Can handle multiclass classification.
- Provides a visual representation of decision rules.
- Supports different splitting criteria and pruning techniques.
- Provides feature importance values.

## Limitations of Decision Tree Classifier
- A deep tree can overfit the training data.
- Small changes in the training data can produce a different tree.
- An overly restricted tree can underfit.
- Greedy splitting does not guarantee a globally optimal tree.
- Feature importance values can be misleading when features are correlated or have different characteristics.

## What I Learned
Through this project, I learned how a Decision Tree Classifier divides data using feature-based conditions and how Gini impurity and entropy help select splits.
I also learned that decision trees can overfit when they become too complex. Parameters such as `max_depth`, `min_samples_split`, `min_samples_leaf` and `ccp_alpha` help control this complexity.

Using GridSearchCV gave me a systematic way to compare hyperparameter combinations through cross-validation instead of relying on manual selection.

Finally, evaluating the model with a confusion matrix, precision, recall, F1-score and accuracy helped me understand its performance in more detail than accuracy alone.

## Conclusion
In this project, I implemented a Decision Tree Classifier using the Iris dataset.
I trained a baseline model, examined its predictions and evaluated its performance. I then used GridSearchCV to tune the splitting criterion, tree depth, minimum sample requirements and pruning strength.
The final model was evaluated on the held-out test dataset, and its confusion matrix and feature importance values were examined.
This project helped me understand decision tree splitting, overfitting, hyperparameter tuning and pruning, along with the importance of evaluating a classification model on data it has not seen during training.