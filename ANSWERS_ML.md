# Day 2–3 Lab: Machine Learning Answers

## 1. Why compare a regression model with a baseline?

A regression baseline provides a simple reference point, often by predicting the mean target value for every sample. A useful model should perform better than this baseline on unseen test data. MAE and RMSE measure prediction error, while R² measures how much variation the model explains relative to a mean-prediction baseline.

In my experiment, linear regression achieved a test MAE of approximately 0.355 and RMSE of 0.471, compared with baseline MAE of approximately 0.952 and RMSE of 1.301. The linear model performed better on this test split.

## 2. Why can a dummy classifier have high accuracy but poor recall?

Accuracy measures the proportion of all predictions that are correct. Recall measures the proportion of actual positive cases that the model identifies.

When positive cases are rare, a dummy classifier that predicts only the majority class can achieve high accuracy while having zero recall for the positive class. For an earthquake-alert classification task, this could mean missing important events despite apparently high accuracy.

Therefore, accuracy should not be evaluated alone. Precision, recall, F1-score, and the confusion matrix help explain model performance.

## 3. How does the classification threshold affect earthquake alerts?

A classification model often produces a probability score. The threshold determines whether that score becomes a positive prediction.

A lower threshold generally identifies more positive cases and may improve recall, but it can also produce more false alarms. A higher threshold generally reduces false alarms but may miss more positive cases.

The threshold should be selected according to the consequences of missed events and false alerts. It should be selected using training or validation data, not by repeatedly tuning against the final test set.

In my experiments, the training workflow selected a threshold using out-of-fold training predictions. The appropriate operational threshold would still need to be evaluated against the intended alerting requirements.

## 4. What is data leakage, and why is it a problem?

Data leakage occurs when information unavailable at prediction time enters model training or evaluation. For example, fitting a scaler or imputer on the complete dataset before splitting it can allow information from the test set to influence training.

This can make evaluation results look better than the model's real-world performance. A safer approach is to split the data first and fit preprocessing steps only on training data. A scikit-learn Pipeline helps ensure that preprocessing is fitted correctly during cross-validation.

Features that directly reveal the target or contain future information should also be excluded.

## 5. How should polynomial regression degrees 1, 2, and 4 be compared?

Polynomial degree controls the complexity of the relationship the model can learn. Degree 1 represents a linear relationship, while higher degrees can represent more complex curves.

Higher degrees can overfit the training data, produce unstable predictions, and increase sensitivity to feature scales. Regularization can help control model complexity.

The models should be compared using the same data split and evaluation metrics, with cross-validation performed on the training data. The test set should be reserved for final evaluation.

I tested linear regression and a degree-3 polynomial model in this run. Linear regression performed better on the observed test split. I have not recorded results for polynomial degrees 1, 2, and 4 here, so no experimental results for those degrees are claimed.

## 6. How does gradient-descent linear regression differ from scikit-learn LinearRegression?

Scikit-learn's LinearRegression typically solves ordinary least squares using numerical linear algebra. A gradient-descent implementation updates model parameters iteratively to reduce a loss function.

Gradient descent depends on choices such as the learning rate, number of iterations, initialization, and stopping criteria. A learning rate that is too high can cause unstable updates, while one that is too low can make training slow.

Feature scaling is particularly important for gradient descent because features with very different numerical ranges can make optimization difficult. Standardizing numerical features often helps gradient descent converge more reliably.

Both implementations should be evaluated on the same train/test split and with the same metrics before comparing their performance.
