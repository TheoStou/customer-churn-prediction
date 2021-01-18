# Customer Churn Prediction
(Predict whether a customer will change telco provider.) </br>

Customer churn or customer attrition is one of the most fundamental problems and major challenges for several large companies. In this report, we train various machine learning models and we aim to predict whether a customer will change a telecommunication provider. In the first part of our effort, we apply and evaluate our models using cross-validation on a dataset provided by Kaggle (train.csv), regarding the churn prediction. For the purpose of our experiment, we use XGBoost, Random Forest, Gradient Boosting, and Catboost. Sequentially, we try to implement the Ensemble method with soft voting to achieve better results. To pinpoint the importance of the contained features we utilize the XGBoost classifier. Finally, in order to ascertain our best model, we consider the confusion matrix, as well as the precision, the recall, and the f1-score for each of the classifiers.


## Data & Problem Description
The dataset is consisted in total of 5000 instances, of two separate files, a training set with 4250 cases and a testing set with 750 cases. The training set is composed of 19 features and one Boolean variable “churn”, which consists our target variable. The given columns from the training set are summarized in the following table: 

























