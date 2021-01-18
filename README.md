# Customer Churn Prediction
(Predict whether a customer will change telco provider.) </br>


## Table of Contents




## Introduction
Customer churn or customer attrition is one of the most fundamental problems and major challenges for several large companies. In this report, we train various machine learning models and we aim to predict whether a customer will change a telecommunication provider. In the first part of our effort, we apply and evaluate our models using cross-validation on a dataset provided by Kaggle (train.csv), regarding the churn prediction. For the purpose of our experiment, we use XGBoost, Random Forest, Gradient Boosting, and Catboost. Sequentially, we try to implement the Ensemble method with soft voting to achieve better results. To pinpoint the importance of the contained features we utilize the XGBoost classifier. Finally, in order to ascertain our best model, we consider the confusion matrix, as well as the precision, the recall, and the f1-score for each of the classifiers.

## Data
The dataset is consisted in total of 5000 instances, of two separate files, a training set with 4250 cases and a testing set with 750 cases. The training set is composed of 19 features and one Boolean variable “churn”, which consists our target variable. The given columns from the training set are summarized in the following table: </br>
ATTRIBUTES OF THE TRAINING SET  </br>
| **Col. No** | **Attribute Name** | **Type** | **Description of the Attribute** |
| :--- | :--- | :--- | :--- |
| 1 | state | string | 2-letter code of the US state of customer residence. |
| 2 | account_length | numerical | Number of months the customer has been with the current telco provider. |
| 3 | area_code | string | 3 digit area code. |
| 4 | international_plan | boolean | The customer has international plan. |
| 5 | voice_mail_plan | boolean | The customer has voice mail plan. |
| 6 | number_vmail_messages | numerical | Number of voice-mail messages. |
| 7 | total_day_minutes | numerical | Total minutes of day calls. |
| 8 | total_day_calls | numerical | Total number of day calls. |
| 9 | total_day_charge | numerical | Total charge of day calls. |
| 10 | total_eve_minutes | numerical | Total minutes of evening calls. |
| 11 | total_eve_calls | numerical | Total number of evening calls. |
| 12 | total_eve_charge | numerical | Total charge of evening calls. |
| 13 | total_night_minutes | numerical | Total minutes of night calls. |
| 14 | total_night_calls | numerical | Total number of night calls. |
| 15 | total_night_charge | numerical | Total charge of night calls. |
| 16 | total_intl_minutes | numerical | Total minutes of international calls. |
| 17 | total_intl_calls | numerical | Total number of international calls. |
| 18 | total_intl_charge | numerical | Total charge of international calls. |
| 19 | number_customer_service_calls | numerical | Number of calls to customer service. |
| 20 | churn | boolean | Customer churn - target variable. |


## Exploratory Data Analysis

A significant task before moving with the manipulation and transformation of the data is the exploratory data analysis which may provide us with some valuable indications regarding the two separate cases ("churn", "non-churn") customers.

### Cheching for imbalance




















