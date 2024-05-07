# Fraudulent Transaction Detection System: Enhancing Financial Security with XGBoost

## Project Overview

This project aims to develop an accurate and efficient machine learning model for fraud detection  that aligns with Blocker Fraud Company's aggressive expansion strategy in Brazil. The model is trained on a dataset containing transaction information and is designed to identify potentially fraudulent transactions.

## Problem Statement
### 1. Business Problem
Blocker Fraud Company specializes in detecting fraud in financial transactions made through mobile devices. The company offers a service called "Blocker Fraud," guaranteeing the blocking of fraudulent transactions. The company's business model is based on a service type with monetization tied to the performance of the service, where the user pays a fixed fee based on the success in detecting fraud in their transactions.

To accelerate customer acquisition in Brazil, the company has adopted an aggressive strategy. Under this strategy:
- Blocker Fraud receives 25% of the value of each transaction correctly detected as fraud.
- Blocker Fraud receives 5% of the value of each transaction incorrectly detected as fraud but is actually legitimate.
- Blocker Fraud refunds 100% of the value to the customer for each transaction incorrectly detected as legitimate but is actually fraud.

This strategy shifts the risk of failing to detect fraud to the company while ensuring revenue for successful fraud detection.

### 2. Business Assumptions
- Fraud prevention involves detecting and preventing fraudulent transactions or banking actions to prevent financial damage and protect the client's and financial institution's reputation.
- Financial fraud can occur through virtual and physical means, leading to increasing investments in security.
- The losses from fraud can be significant, reaching up to $ 1 billion, which is half of the amount institutions invest in information security technology annually.

### 3. Solution Strategy
The solution to this problem involves a data science project to develop a machine learning model for fraud detection in financial transactions. The project will follow these steps:

1. **Data Description**: Collect and study the data, handle missing values, and perform initial data description using descriptive statistics.
2. **Feature Engineering**: Create a mind map to generate hypotheses and new features, which will aid in exploratory data analysis (EDA) and potentially improve model performance.
3. **Data Filtering**: Remove columns or rows that are not relevant to the business, such as customer ID or hash codes.
4. **Exploratory Data Analysis**: Conduct univariate, bivariate, and multivariate analysis to understand the data and test hypotheses created in feature engineering.
5. **Data Preparation**: Transform the data for machine learning modeling by encoding, oversampling, subsampling, or rescaling.
6. **Feature Selection**: Use algorithms like Boruta to select the best columns for training the machine learning model, reducing dimensionality and overfitting.
7. **Machine Learning Modeling**: Train machine learning algorithms to predict fraudulent transactions, validate the model, and apply cross-validation to assess learning capacity.
8. **Hyperparameter Fine Tuning**: Fine-tune the parameters of the selected model to improve performance.
9. **Conclusions**: Test the model's prediction capacity using unseen data and answer business questions to demonstrate applicability.
10. **Model Deployment**: Create a Flask API and save the model and functions for implementation in the API.


## Getting Started
### Requirements

The following packages are required for this project:

- Python (3.x)

Clone the repository and install the required libraries::

    git clone https://github.com/slashercode/fraud_transaction_system.git
    cd fraud_transaction_system
    pip install -r requirements.txt


**Note:** Some of these packages may have dependencies that need to be installed as well.

## Data Insights
Analysis of the dataset revealed several insights into fraudulent transactions:

- **Transaction Types**: Fraudulent transactions are more likely to be of a certain type (e.g., 'debit') compared to legitimate transactions, indicating potential patterns that can be used for detection.

- **Transaction Amounts**: There may be a pattern in the transaction amounts of fraudulent transactions, such as clustering around specific values or higher average amounts, which can help in setting thresholds for fraud detection.

- **Account Types**: Certain account types (e.g., 'savings') may be more susceptible to fraud, either due to weaker security measures or higher transaction volumes, highlighting areas for targeted monitoring.

- **Fraud Frequency**: Understanding the frequency of fraudulent transactions relative to legitimate ones can help in assessing the overall impact of fraud on the business and in designing appropriate detection strategies.

- **TransactionID Patterns**: Analyzing patterns in TransactionID, such as sequential or random IDs, can provide insights into how fraudsters attempt to manipulate transaction records.

These insights can guide the feature selection process and model development, enabling the creation of a robust fraud detection system.

## Machine Learning Models

The machine learning models were evaluated using cross-validation to assess their performance. Here are the results with default parameters:

| Model                  | Balanced Accuracy | Precision       | Recall          | F1 Score        | Kappa           |
| ---------------------- | ----------------- | --------------- | --------------- | --------------- | --------------- |
| Dummy Model            | 0.499 +/- 0.0     | 0.0 +/- 0.0     | 0.0 +/- 0.0     | 0.0 +/- 0.0     | -0.001 +/- 0.0  |
| Logistic Regression    | 0.565 +/- 0.009   | 1.0 +/- 0.0     | 0.129 +/- 0.017 | 0.229 +/- 0.027 | 0.228 +/- 0.027 |
| K Nearest Neighbors    | 0.705 +/- 0.037   | 0.942 +/- 0.022 | 0.409 +/- 0.074 | 0.568 +/- 0.073 | 0.567 +/- 0.073 |
| Support Vector Machine | 0.595 +/- 0.013   | 1.0 +/- 0.0     | 0.19 +/- 0.026  | 0.319 +/- 0.037 | 0.319 +/- 0.037 |
| Random Forest          | 0.865 +/- 0.017   | 0.972 +/- 0.014 | 0.731 +/- 0.033 | 0.834 +/- 0.022 | 0.833 +/- 0.022 |
| XGBoost                | 0.88 +/- 0.016    | 0.963 +/- 0.008 | 0.761 +/- 0.033 | 0.85 +/- 0.023  | 0.85 +/- 0.023  |
| LightGBM               | 0.701 +/- 0.089   | 0.18 +/- 0.1    | 0.407 +/- 0.175 | 0.241 +/- 0.128 | 0.239 +/- 0.129 |


These results provide insights into the performance of each model, with **_Random Forest_** and **_XGBoost_** showing the highest balanced accuracy and F1 scores among the models tested. **_XGBoost_** demonstrated superior performance compared to other models tested, making it a promising candidate for further optimization and deployment. I have selected XGBoost for further optimization due to its ability to handle complex datasets and its high predictive power make it ideal for the fraud detection system.

## Machine Learning Model Performance
The selected model, **_XGBoost_**, was fine-tuned to optimize its parameters, resulting in improved performance metrics. The table below illustrates the model's learning capacity:

| Metric            | Score           |
| ----------------- | --------------- |
| Balanced Accuracy | 0.881 +/- 0.017 |
| Precision         | 0.963 +/- 0.007 |
| Recall            | 0.763 +/- 0.035 |
| F1 Score          | 0.851 +/- 0.023 |
| Kappa Score       | 0.851 +/- 0.023 |

The model's ability to generalize to unseen data was also evaluated, yielding the following results:

| Metric            | Score |
| ----------------- | ----- |
| Balanced Accuracy | 0.915 |
| Precision         | 0.944 |
| Recall            | 0.829 |
| F1 Score          | 0.883 |
| Kappa Score       | 0.883 |

## Business Impact
The Blocker Fraud Company's revenue model is based on detecting fraudulent transactions:

- 25% of each transaction value truly detected as fraud, potentially yielding $ 60,613,782.88.
- 5% of each transaction value detected as fraud, even if the transaction is legitimate, potentially totaling $ 183,866.98.
- Full reimbursement of the transaction value to the customer for each transaction detected as legitimate but actually fraudulent, resulting in a potential loss of $ 3,546,075.42.

## Financial Metrics
- **Precision and Accuracy**: The model achieves a precision of 94.4% and a balanced accuracy of 91.5% on unseen data.
- **Reliability**: The model demonstrates a capability to detect 76.3% +/- 3.5% of fraudulent transactions, with a recall of 0.829 on unseen data.
- **Expected Revenue**: By using the model to classify all transactions, the company could potentially generate revenue of $ 60,797,649.86, compared to zero revenue using the current method.
- **Potential Loss**: Misclassifying transactions with the model could lead to a loss of $ 3,546,075.42, significantly lower than the potential loss of $ 246,001,206.94 with the current method.
- **Expected Profit**: The model could lead to a profit of $ 57,251,574.44, in contrast to the current method's loss of $ 246,001,206.94.

## Conclusions
Despite the extreme class imbalance, the analysis and model development achieved good performance metrics.
The projected revenue of $ 57,251,574.44 demonstrates the effectiveness of the model and its potential to benefit the company.

## Key Findings
- Effective models can be created even with highly unbalanced classes.
- Models can accurately classify classes with less than 1% representation in the dataset.

## Future Steps
- Test at most 12 additional hypotheses to further improve the model.
- Implement oversampling or subsampling techniques to enhance model performance.
- Deploy the API on the Heroku platform for practical use.

## Acknowledgments

- This project was inspired by the need for more effective fraud detection methods in financial transactions.

## License

**NOT FOR COMMERCIAL USE**

_If you intend to use any of my code for commercial use please contact me and get my permission._

_If you intend to make money using any of my code please get my permission._
