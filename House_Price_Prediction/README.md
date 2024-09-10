# House Price Prediction Project

This project focuses on predicting house prices using a variety of machine learning models. The process includes data preprocessing, feature encoding, and applying several regression models to achieve the most accurate predictions.

## 1. Dataset Overview

The project starts with loading the dataset (`HousePricePrediction.xlsx`) to analyze and predict house prices. We first inspect the dataset to understand the structure, including the number of rows, columns, and the type of data contained. This helps set the foundation for the preprocessing steps.

## 2. Data Preprocessing

### 2.1 Data Types and Variable Counts

We categorize the features into different types: categorical, integer, and float. This distinction is necessary because the preprocessing steps for categorical features differ from those for numerical features. For example, categorical variables need to be encoded before they can be used in machine learning models, while numerical features may require scaling or normalization.

### 2.2 Correlation Analysis

A heatmap is used to visualize correlations between the numerical variables in the dataset. This allows us to identify which features might be highly correlated with the target variable (`SalePrice`) or with each other. By understanding these relationships, we can make informed decisions about feature selection and data transformation.

### 2.3 Categorical Feature Exploration

We explore the categorical variables to understand their distribution and the number of unique values each feature contains. Visualizing these variables gives insights into how these features are structured and whether any feature has too many or too few unique values, which could affect the model's ability to generalize.

### 2.4 Handling Missing Data

To handle missing data, we fill the missing values in the target variable `SalePrice` with the mean. Additionally, we drop features that are irrelevant to prediction, such as `Id`, to avoid introducing noise into the model. After these steps, we ensure that no missing data remains in the dataset.

## 3. One-Hot Encoding for Categorical Features

Categorical variables need to be converted into a numerical format for machine learning models to process them. One-Hot Encoding is employed here because it converts categorical variables into binary vectors, which is a standard approach for handling categorical data in machine learning tasks. This method avoids assigning ordinal relationships where none exist and ensures the models treat these features appropriately.

## 4. Model Training and Evaluation

We split the dataset into training and validation sets to train different machine learning models and evaluate their performance. This helps to prevent overfitting and gives an indication of how well the models might perform on unseen data.

### 4.1 Support Vector Regressor (SVR)

SVR was chosen for its ability to handle both linear and non-linear relationships in the data. It's a robust model that can efficiently find the optimal decision boundary, especially in cases where there are high-dimensional spaces. The model was evaluated using the mean absolute percentage error (MAPE), which is suitable for continuous target variables like house prices.

### 4.2 Random Forest Regressor (RFR)

The Random Forest Regressor was applied as it is an ensemble method that reduces variance by averaging the results of multiple decision trees. This method is particularly effective when dealing with noisy data or when there are many features that could have non-linear interactions. The model’s performance was also measured using MAPE, and it provided a stable prediction.

### 4.3 Linear Regression

Linear Regression is the simplest model we applied to establish a baseline for house price prediction. While it assumes a linear relationship between features and the target, it often serves as a good starting point for regression tasks. The model is straightforward to interpret and was evaluated using MAPE for consistency with other models.

### 4.4 CatBoost Regressor

Finally, CatBoost was selected as it is a gradient boosting model that handles categorical variables natively without needing extensive preprocessing like One-Hot Encoding. This makes it a strong choice for datasets with categorical data. The model's performance was evaluated using the R-squared score, which measures how well the predicted values match the actual target values.

## 5. Conclusion

Throughout this project, various regression models were applied to predict house prices. From simple linear regression to advanced ensemble methods like Random Forest and CatBoost, each model was evaluated based on its predictive performance. The results from each model provide insights into which methods are most effective for this particular dataset, giving a holistic view of house price prediction using machine learning.
