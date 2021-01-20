# Customer Churn Prediction
(Predict whether a customer will change telco provider.) </br>

The project is based on the Kaggle competition "Customer Churn Prediction 2020", as you can find [here](https://www.kaggle.com/c/customer-churn-prediction-2020/overview/description). </br> The evaluation is based on the test Accuracy criterion: (Accuracy = Number of correct predictions/Number of total test samples). 


## Table of Contents




## Introduction
Customer churn or customer attrition is one of the most fundamental problems and major challenges for several large companies. In this project, we train various machine learning models and we aim to predict whether a customer will change a telecommunication provider. In the first part of our effort, we apply and evaluate our models using cross-validation on a dataset provided by Kaggle (train.csv), regarding the churn prediction. For the purpose of our experiment, we use XGBoost, Random Forest, Gradient Boosting, and Catboost. Sequentially, we try to implement the Ensemble method with soft voting to achieve better results. To pinpoint the importance of the contained features we utilize the XGBoost classifier. Finally, in order to ascertain our best model, we consider the confusion matrix, as well as the precision, the recall, and the f1-score for each of the classifiers.

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
Estimate the churn percentage: 
```ruby
y_True = train["churn"][train["churn"] == 'yes']
print ("Churn Percentage = "+str( (y_True.shape[0] / train["churn"].shape[0]) * 100 ))
```
Output: Churn Percentage = 14.070588235294117

Create a bar plot for churn:
```ruby
y = train["churn"].value_counts()
sns.barplot(y.index, y.values)
```
![Number of cases, target attribute(churn)](https://user-images.githubusercontent.com/74372152/105177577-87828300-5b2f-11eb-8129-088f1a2a32c1.png) </br>
Number of cases, target attribute(churn) </br>


### Visualizing catagorical features regarding the churn value
![Donat chart for area_code](https://user-images.githubusercontent.com/74372152/105177865-f8299f80-5b2f-11eb-91b0-f0b28ba806cf.png) </br>
Donat chart for area_code </br> </br>
![Donat chart for international_plan](https://user-images.githubusercontent.com/74372152/105177907-0677bb80-5b30-11eb-8e8f-e9281ba169bc.png) </br>
Donat chart for international_plan </br> </br>
![Donat chart for voice_mail_plan](https://user-images.githubusercontent.com/74372152/105177945-11325080-5b30-11eb-845e-bd53dc56988e.png) </br>
Donat chart for voice_mail_plan </br> </br>


### Visualizing numerical features regarding the churn value
![Histogram of account_length](https://user-images.githubusercontent.com/74372152/105178633-f7453d80-5b30-11eb-81ea-b5f008fca5d8.png) </br>
Histogram of account_length </br> </br> 
![Histogram of number_vmail_messages](https://user-images.githubusercontent.com/74372152/105178682-0926e080-5b31-11eb-93d6-94c40a19739d.png) </br>
Histogram of number_vmail_messages </br> </br>
NOTE: Additional plots are provided in Plots folder.


## Feature Engineering
Maybe the most considerable phase, in order to achieve the best and most efficient results is that of the feature engineering. To accomplice our aims, we begin with transforming the boolean attributes to binary and we proceed with the combination of some features. Furthermore, a significant procedure that contributes to obtaining superior results is that of the dummy’s method to create dummy variables. Finally, considering the correlation matrix and the feature importance we remove the undesired columns.

### A. Transform boolean variables to binary variables
Often it is more effective to convert a string or a boolean variable into a binary variable. With this way, an algorithm may handle more appropriately a dataset and achieve better performance.

```ruby
# 0 and 1 for binary attributes

# Binarize churn 
df_train['churn'] = train['churn']
df_train['churn'] = np.where(df_train['churn'] == 'yes', 1, 0) # change churn to 0 for no and 1 for yes

# Binarize international_plan
df_train['international_plan'] = train['international_plan']
df_train['international_plan'] = np.where(df_train['international_plan'] == 'yes', 1, 0) 

# Binarize voice_mail_plan
df_train['voice_mail_plan'] = train['voice_mail_plan']
df_train['voice_mail_plan'] = np.where(df_train['voice_mail_plan'] == 'yes', 1, 0) 
```
### B. Feature combination
Three new features are created based on the combination of other features. 
total_minutes: total_day_minutes + total_eve_minutes + total_night_minutes + total_intl_minutes
total_calls: total_day_calls + total_eve_calls + total_night_calls + total_intl_calls
total_charge: total_day_charge + total_eve_charge + total_night_charge + total_intl_charge









