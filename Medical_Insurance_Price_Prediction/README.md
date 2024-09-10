# Insurance Price Prediction Project

This project aims to predict insurance charges based on various factors such as age, sex, BMI, number of children, smoking status, and region. Different machine learning models are used to train on the dataset, and their performances are compared.

## 1. Dataset Overview

The dataset used for this project contains information about individuals' age, sex, BMI, number of children, smoking habits, region, and the insurance charges incurred. This information serves as features that the machine learning models use to predict the `charges` variable.

## 2. Data Preprocessing

Before feeding the data into machine learning models, several preprocessing steps are performed:

- **Missing Data Handling**: The dataset is checked for missing values. Fortunately, there are no missing data, so no further imputation is needed.
- **Encoding Categorical Features**: Categorical variables such as `sex`, `smoker`, and `region` are encoded to numeric values. This transformation is necessary for machine learning models to process non-numeric data.
- **Outlier Detection**: For features like `bmi`, the interquartile range (IQR) method is used to identify and cap outliers, ensuring that extreme values don't unduly affect model performance.

## 3. Exploratory Data Analysis (EDA)

A variety of visualizations are created to better understand the distribution and relationships within the dataset. For example, pie charts are used to visualize the distribution of categorical features like `sex`, `smoker`, and `region`. Bar charts and scatter plots help examine how various features impact insurance charges, with special focus on features like `age`, `bmi`, and `smoker`.

## 4. Machine Learning Models

### 4.1 Linear Regression

Linear regression is used to establish a baseline for predicting insurance charges. It assumes a linear relationship between features and the target variable, `charges`. After fitting the model, both training and testing accuracy are evaluated, along with cross-validation scores to assess the model's generalizability.

### 4.2 Support Vector Regressor (SVR)

Support Vector Regressor (SVR) is implemented to capture non-linear relationships in the data. The SVR model is trained, and R-squared scores are calculated for both training and testing datasets. Cross-validation is also applied to evaluate the model's robustness.

### 4.3 Random Forest Regressor

Random Forest is chosen due to its capability of handling both linear and non-linear relationships and reducing overfitting by averaging multiple decision trees. Grid search is performed to find the optimal hyperparameters for the model, and the performance is measured based on R-squared scores and cross-validation.

### 4.4 Gradient Boosting Regressor

Gradient Boosting is another ensemble method applied to improve prediction accuracy. It uses sequential decision trees to minimize errors iteratively. Grid search is again used for hyperparameter tuning, and both training and testing R-squared scores are evaluated after optimization.

### 4.5 XGBoost Regressor

XGBoost is used as the final model due to its strong performance on structured datasets and ability to handle missing values and outliers effectively. It is tuned using hyperparameters such as `n_estimators` and `max_depth`, and the final performance is evaluated using R-squared scores and cross-validation.

## 5. Model Evaluation and Feature Importance

Each model's performance is compared using R-squared scores and cross-validation to determine the best model for predicting insurance charges. Feature importance is analyzed to identify the most influential features in predicting insurance charges. For instance, smoking status and age are shown to have significant effects on insurance costs.

## 6. Saving the Final Model

The final XGBoost model is saved as a pickle file for future use. This allows predictions to be made on new data, making the model applicable in real-world scenarios.

## 7. Conclusion

This project demonstrates the application of various machine learning models to predict insurance charges. From simpler models like linear regression to more complex ensemble models like Random Forest and XGBoost, each provides insights into the relationships between features and the target variable. XGBoost stands out as the best performer, achieving the highest R-squared score.
