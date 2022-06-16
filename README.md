# Customer churn prediction of telco provider

In this project we use [this data](https://www.kaggle.com/competitions/customer-churn-prediction-2020/overview) from _Kaggle_. The main goal of the project is to predict whether a customer will change telco provider.

## Overview of the data
The training dataset contains 4250 samples. Each sample contains 19 features and 1 boolean variable "churn" which indicates the class of the sample. The 19 input features and 1 target variable are:
1. **state**, string. 2-letter code of the US state of customer residence
2. **account_length**, numerical. Number of months the customer has been with the current telco provider
3. **area_code**, string="area_code_AAA" where AAA = 3 digit area code.
4. **international_plan**, (yes/no). The customer has international plan.
5. **voice_mail_plan**, (yes/no). The customer has voice mail plan.
6. **number_vmail_messages**, numerical. Number of voice-mail messages.
7. **total_day_minutes**, numerical. Total minutes of day calls.
8. **total_day_calls**, numerical. Total minutes of day calls.
9. **total_day_charge**, numerical. Total charge of day calls.
10. **total_eve_minutes**, numerical. Total minutes of evening calls.
11. **total_eve_calls**, numerical. Total number of evening calls.
12. **total_eve_charge**, numerical. Total charge of evening calls.
13. **total_night_minutes**, numerical. Total minutes of night calls.
14. **total_night_calls**, numerical. Total number of night calls.
15. **total_night_charge**, numerical. Total charge of night calls.
16. **total_intl_minutes**, numerical. Total minutes of international calls.
17. **total_intl_calls**, numerical. Total number of international calls.
18. **total_intl_charge**, numerical. Total charge of international calls
19. **number_customer_service_calls**, numerical. Number of calls to customer service
20. _**churn**_, (yes/no). Customer churn - target variable.

## Methods used
* Exploratory Data Analysis (EDA)
* Inferential Statistics
* Data Visualisation
* Oversampling & Undersampling for Class Imbalance
* Feature Engineering
* Feature Selection
* Cross Validation
* Clustering
* Predictive Modeling
* Machine Learning
* Hyperparameter Tuning

## Technlogies
* Python, Jupyter Notebook
* Pandas, numpy
* Seaborn, matplotlib
* ImbLearn
* Scikit-Learn (SkLearn), AutoSklearn
* MLFlow
* SHAP
* XGBoost, LightGBM, Catboost
* HyperOpt

## Notebooks & Python Scripts
| Notebook | Description |
| -------- | ----------- |
| [Research](research.ipynb) | First notebook with EDA and Data Visualisation |
| [Preprocessing](preprocessing.py) | Python script with functions for Data Handling |
| [Feature Engineering](data_notebooks/feature_engineering.ipynb) | Notebook with experiments for feature engineering (all necessary engineering techniques were included in `preprocessing.py`) |
| [KMeans Research](clustering_approach/kmeans_research.ipynb) | Notebook with first clustering approach (fails) |
