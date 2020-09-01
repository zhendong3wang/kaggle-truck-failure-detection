# Truck APS Failure Classification Using Machine Learning - IDA 2016

## Data 

[Data sources](https://ida2016.blogs.dsv.su.se/?page_id=1387) are provided by Scania, which are also available in [Kaggle Datasets](https://www.kaggle.com/uciml/aps-failure-at-scania-trucks-data-set). All the feature names are anonymized for proprietary reasons. In this [Jupyter notebook](./notebooks/1.0-data-exploration.ipynb), we performed a data exploration analysis.

## Evaluation Methods

In order to find the optimal model, F_beta score is applied in cross validation. beta equals to 2 indicates that we have a higher weight on the recall when selecting models . In other words, type II error will be weigh higher.

When evaluating the test data, the main measure for deciding the optimal model will be the challenge metric (Total cost = Cost1 · `#type I failures` + Cost2 · `#type II failures`). Besides, F_beta score and classification reports of precision/recall will be applied when running the script as a reference for the model performance. Together with challenge metrics, F_beta scores are recorded as F2 scores.

`#type I failures` is the count of type I failures (false positive), which indicates no faulty systems were reported positive falsely and leads to unnecessary mechanical check. While `#type II failures` is the count of type II failures (false negative), which indicates these problematic systems were reported no failure. The costs are 10 and 500 respectively.

## Implementation Set-ups

The objective is to develop a good prediction model for classifying the system failure. In addition to prediction accuracy of models, we would like to find an approach which can also minimize maintenance cost. The maintenance cost is defined later in the evaluation methods. Besides, there is another set of test data that we need to classify and then save the predictions for further check.

This failure classification task is implemented in multiple ways:

1. Two methods: XGBoost ([notebook](./notebooks/1.1-model-comparison-without-imputation(XGBoost).ipynb)) vs Logistic Regression ([notebook](./notebooks/1.2-model-comparison-with-imputation(LogisticRegression).ipynb));
2. ExtraTrees ([notebook](./notebooks/1.3-model-comparison-feature-selection.ipynb)) is applied for feature selection purpose after we choose the optimal approaches from two previous methods;
3. The whole set of training data file is split into 80/20 as training/testing data during the model comparison. Training data will be used in fitting models in order to execute grid search/randomized search (using 5-fold cross validation) for finding optimal parameters. While the testing data is used to get the evaluation scores of different models for comparison;
4. 'random\_state' is fixed in data-splitting/model-training phases in order to get consistent results.

## Model Comparison Results

According to XGBoost results as below, the best model is the one with parameter ‘scale_pos_weight’ (59) by comparing both F2 score and total cost. From the raw outputs, we can also observed that the actual predictions of this model are quite balanced. There are 31 type II failures and 23 type I failures respectively, with precision at 0.88 and recall at 0.84 for positive cases.

Model | F2 score | Total Cost 
--- | --- | --- 
Baseline (default) | 0.7726 | 25640 
Weight scaled (59) | 0.8518 | 15730 
Grid search CV | 0.8300 | 17340 
Randomized search CV | 0.7708 | 26120 

Compared to XGBoost, overall F2 scores are lower in Logistic Regression models - the difference is about 0.1 in the following table. However, if we look at the output in Appendix, the counts of type II and type I failures in confusion matrix are 11 and 284 compared to 31 and 23 in XGBoost’s optimal model. Due to this reason, the total cost of Logistic Regression is much lower than XGBoost. Logistic Regression combined with grid search CV for finding optimal regularization strength (the third approach) has the lowest total cost 8340, in comparison with 15730 from the optimal XGBoost model.

Model | F2 score | Total Cost 
--- | --- | --- 
Baseline (default) | 0.4874 | 55740  
Weight balanced | 0.7142 | 9280  
Grid search CV | 0.7423 | 8340  
SMOTE + Weight balanced  | 0.7354 | 10670  

As we can see from the table below for feature selection, there is not a significant performance boost by reducing the features by importance of 0.005 and 0.01. The total costs of models using reduced features are still higher than the previous optimal ones. However, the training speed is much faster than previous models. We could probably apply this approach when we have limited training time in real life usages.

Model | F2 score | Total Cost  
--- | --- | --- 
0.005 + LR  | 0.7235 | 8670  
0.005 + XGBoost  | 0.7913 | 21850  
0.01 + LR  | 0.7009 | 9970  
0.01 + XGBoost  | 0.7662 | 22620  

Back to the objective, we find that that Logistic Regression combined with grid search CV models can achieve the lowest the maintenance cost. Even though XGBoost models have better F2 scores in general, Logistic Regression handles the imbalanced class problem better in this case. In conclusion, Logistic Regression combined with grid search CV for finding the optimal regularization parameter C is the optimal model.

In the end, we fit the optimal model using Logistic Regression with all the training data. And then we predict all the instances from testing data set by the final model. The Jupyter notebook is available [here](./notebooks/1.4-make-prediction.ipynb) The results are saved as a [text file](./results.txt).