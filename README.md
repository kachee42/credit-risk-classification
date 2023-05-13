# Creditworthiness Report

## Overview of the Analysis

When a loan is applied for at a bank or other financial institution it is critical for the health of the institution that only the most credit worthy applicants be approved. The institution runs the risk of collapse if too many bad loans are accepted. This harms not only the institution and all of its employees, but also the borrowers of healthy loans. With this in mind, the following analysis uses the sklearn module to create machine learning models to predict whether or not a loan will default based on past loans.

The data used to train the models for this analysis included the size of the loan, interest rate of the loan, income of the borrower, debt-to-income ratio of the borrow, number of accounts the borrower has, number of derogatory marks the borrower has, total debt of the borrower and the status of the loan for over 77,000 loans. As the target labels variable the loan status columns was separated from the other features. Features were set equal to X and labels were set equal to y. These variables were then split 75% to 25% into training and testing datasets respectively. A logistic regression model was instantiated and the training X and y data were fit to the model. The trained model was then used to predict the loan status for the test feature data. The balanced accuracy score was calculated along with a confusion matrix and precision, recall, and f1-scores for both healthy and unhealthy loans checked against the actual test labels.

The entire process was repeated again, but with resampled data. There was a large discrepancy between the number of healthy loans compared the the unhealthy loans. The model's accuracy in predicting healthy loans was very good, but was noticeably less accurate at predicting unhealthy loans. The imblearn oversampling module was used to oversample the loans in default to provide more training data to try to increase the accuracy.

## Results

* Machine Learning Model 1:
  * Balanced accuracy score:  0.9520
  * Healthy loan precision:   1.00
  * Healthy loan recall:      0.99
  * High-risk loan precision: 0.85
  * High-risk loan recall:    0.91

* Machine Learning Model 2:
  * Balanced accuracy score:  0.9937
  * Healthy loan precision:   1.00
  * Healthy loan recall:      0.99
  * High-risk loan precision: 0.84
  * High-risk loan recall:    0.99

## Summary

Both of the above models were excellent at predicting the healthy loans, with a tiny proportion of the predictions being false positives. The first model had more false positives than the second oversampled model, however the oversampled model had a few more false negatives, meaning that it incorrectly categorized a few more healthy loans as high risk. The difference in false negatives, however, doesn't outweigh the significant improvement in reducing false positives. For the banks who do not want to approve high-risk loans, the second oversampled model is a safer model to use to ensure that as few bad loans as possible are lent out. Note that because the loans used to train these models were already composed of applicants that were deemed creditworthy, the porportion of defaulted loans will always be much lower than healthy loans. Because of this it will always be difficult to train a model that has enough data on bad loans to be able to predict them to the best accuracy.
