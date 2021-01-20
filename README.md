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
- total_minutes: total_day_minutes + total_eve_minutes + total_night_minutes + total_intl_minutes
- total_calls: total_day_calls + total_eve_calls + total_night_calls + total_intl_calls
- total_charge: total_day_charge + total_eve_charge + total_night_charge + total_intl_charge


### C. Dummies for categorical features
Concerning the categorical variable, sometimes is more useful to convert them into binary variables. Nonetheless, such an operation frequently is not sufficient to enhance the improvement of a classifier. To this point, get_dummies is a common method utilized to offer better outcomes.
```ruby
# Using dummies for area_code and state

# One hot encode the area_code column
df_area_code_one_hot = pd.get_dummies(df_train['area_code'], prefix='area_code')

# One hot encode the state column
df_state_one_hot = pd.get_dummies(df_train['state'], prefix='state')

# Combine the one hot encoded columns with df_train                                    
df_train = pd.concat([df_train, df_area_code_one_hot, df_state_one_hot], axis=1)

# Drop the original categorical columns (because now they've been one hot encoded)
df_train = df_train.drop(['area_code', 'state'], axis=1)
```


### D. Removing the undesired columns
![Correlation matrix of the features before getting dummies](https://user-images.githubusercontent.com/74372152/105180765-a256f680-5b33-11eb-81a9-c524d2f70bd4.png) </br>
Correlation matrix of the features before getting dummies </br>


## Machine Learning Classifiers 
- XGBClassifier
- Random Forest
- Gradient Boosting Machine
- Catboost

## Implement an Essemble Method
One very common method that is being utilized in many cases, is that of the ensemble technique. With the term ensemble method, we define this technique where we combine several basic models in order to generate an optimal predictive model. In other words, rather than creating one model that can hopefully make an accurate prediction, we take into account several different models to produce one final classifier.
Using Ensemble Vote Classifier (Soft Voting): 
```ruby
from sklearn.model_selection import RepeatedStratifiedKFold
clf1 = GradientBoostingClassifier()
clf2 = RandomForestClassifier(random_state=1)
clf3 = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,colsample_bytree=1, gamma=0.01, learning_rate=0.1, max_delta_step=0,max_depth=7,
                    min_child_weight=5, missing=None, n_estimators=20,n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,reg_alpha=0, 
                    reg_lambda=1, scale_pos_weight=1, seed=None, silent=True, subsample=1).fit(X_train, y_train)
ensemble = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], voting='soft')
 
# evaluate a give model using cross-validation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
print('10-RepeatedStratifiedKFold cross validation:\n')
 
labels = ['GradientBoostingClassifier', 'Random Forest', 'XGBClassifier','Ensemble']
results = []
for clf, label in zip([clf1, clf2, clf3, ensemble], labels):
 
    scores = model_selection.cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1, error_score='raise')
    results.append(scores)
    print("Accuracy: %0.3f (+/- %0.2f) [%s]"% (scores.mean(), scores.std(), label))
 
plt.boxplot(results, labels=labels, showmeans=True)
plt.show()
``` 
Output: 10-RepeatedStratifiedKFold cross validation:

Accuracy: 0.979 (+/- 0.01) [GradientBoostingClassifier] </br>
Accuracy: 0.974 (+/- 0.01) [Random Forest] </br>
Accuracy: 0.978 (+/- 0.01) [XGBClassifier] </br>
Accuracy: 0.979 (+/- 0.01) [Ensemble] </br>

![Boxplot of ensmble method](https://user-images.githubusercontent.com/74372152/105181442-79833100-5b34-11eb-945d-911c61f7154b.png)


## Evaluation of classifiers
All classifiers are assessed in terms of Precision, Recall, Accuracy, F-measure, specificity, learning curves and ROC(Receiver Operating Characteristics). Additionally, a confusion matrix is provided for each case. An example of the XGBoost clasifier evaluation is demonstrated down below. </br>
![Learning Curve for XGBoost](https://user-images.githubusercontent.com/74372152/105182036-3bd2d800-5b35-11eb-98df-fc984f68d681.png) </br>
Learning Curve for XGBoost </br> </br>
![Learning Curve for XGBoost2](https://user-images.githubusercontent.com/74372152/105182790-2316f200-5b36-11eb-8cd2-f7aec8cb9ef9.png) </br>
ROC curve for XGBoost </br> </br>
![Confution Matrix for XGBoost](https://user-images.githubusercontent.com/74372152/105182928-4b9eec00-5b36-11eb-8498-0d3c172d1763.png) </br>
Confution Matrix for XGBoost </br> </br> 

NOTE: Additional plots are provided in Plots folder. </br> </br> 
SUMMARAZATION RESULTS </br>
| **Classifiers** | **Precision(%)** | **Recall(%)** | **Accuracy(%)** | **F-measure(%)** |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | 100 | 84.3 | 97.8 | 91.5 |
| **Random Forest** | 100 | 80.9 | 97.2 | 89.5 |
| **Gradient Boosting** | 98.3 | 86.5 | 97.9 | 92.0 |
| **Catboost** | 99.0 | 84.9 | 98.0 | 91.4 |
| **Ensemble** | 100 | 84.9 | 97.9 | 91.9 |






