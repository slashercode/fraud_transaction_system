# 1. IMPORT
import os
import warnings

import inflection
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from boruta import BorutaPy
from category_encoders import OneHotEncoder
from IPython.core.display import HTML
from IPython.display import Image
from lightgbm import LGBMClassifier
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Ignore warnings to keep the output clean
warnings.filterwarnings("ignore")

# Set a random seed for reproducibility
seed = 42
np.random.seed(seed)


# 2. HELPER FUNCTIONS
def save_plot(plot, file_name, figsize=(12, 12)):
    """
    Save the plot to a file in the 'plots' directory.

    Args:
    - plot: The plot object to be saved.
    - file_name: The name of the file to save the plot as.
    - figsize: Tuple specifying the width and height of the figure.

    Returns:
    None
    """
    plot_dir = "plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Set figure size and font size
    plot.set_size_inches(figsize)

    file_path = os.path.join(plot_dir, file_name)
    plot.savefig(file_path, bbox_inches="tight")
    print(f"Plot saved as '{file_name}'")


def ml_scores(model_name, y_true, y_pred):
    """
    Calculate and return various evaluation metrics for a machine learning model.

    Args:
    - model_name (str): Name of the model for which scores are calculated.
    - y_true (array-like): True labels.
    - y_pred (array-like): Predicted labels.

    Returns:
    - DataFrame: DataFrame containing the evaluation metrics.
    """
    # Calculate evaluation metrics
    accuracy = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    # Create a DataFrame with the evaluation metrics
    scores_df = pd.DataFrame(
        {
            "Model": [model_name],
            "Balanced Accuracy": np.round(accuracy, 2),
            "Precision": np.round(precision, 2),
            "Recall": np.round(recall, 2),
            "F1": np.round(f1, 2),
            "Kappa": np.round(kappa, 2),
        }
    )

    return scores_df


def calcCramerV(x, y):
    """
    Calculate Cramer's V statistic for two categorical variables.

    Args:
    - x (array-like): First categorical variable.
    - y (array-like): Second categorical variable.

    Returns:
    - float: Cramer's V statistic.
    """
    # Create a contingency table
    cm = pd.crosstab(x, y).values
    n = cm.sum()
    r, k = cm.shape

    # Calculate the chi-squared statistic and its corrected values
    chi2 = stats.chi2_contingency(cm)[0]
    chi2corr = max(0, chi2 - (k - 1) * (r - 1) / (n - 1))

    kcorr = k - (k - 1) ** 2 / (n - 1)
    rcorr = r - (r - 1) ** 2 / (n - 1)

    # Calculate Cramer's V statistic
    return np.sqrt((chi2corr / n) / (min(kcorr - 1, rcorr - 1)))


def ml_cv_results(model_name, model, x, y, verbose=1):
    """initial"""
    balanced_accuracies = []
    precisions = []
    recalls = []
    f1s = []
    kappas = []

    mm = MinMaxScaler()

    x_ = x.to_numpy()
    y_ = y.to_numpy()

    count = 0

    """cross-validation"""
    skf = StratifiedKFold(n_splits=5, shuffle=True)

    for index_train, index_test in skf.split(x_, y_):
        ## Showing the Fold
        if verbose > 0:
            count += 1
            print("Fold K=%i" % (count))

        ## selecting train and test
        x_train, x_test = x.iloc[index_train], x.iloc[index_test]
        y_train, y_test = y.iloc[index_train], y.iloc[index_test]

        ## applying the scale
        x_train = mm.fit_transform(x_train)
        x_test = mm.transform(x_test)

        ## training the model
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        ## saving the metrics
        balanced_accuracies.append(balanced_accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred))
        kappas.append(cohen_kappa_score(y_test, y_pred))

    """results"""
    accuracy_mean, accuracy_std = np.round(np.mean(balanced_accuracies), 22), np.round(
        np.std(balanced_accuracies), 2
    )
    precision_mean, precision_std = np.round(np.mean(precisions), 2), np.round(
        np.std(precisions), 2
    )
    recall_mean, recall_std = np.round(np.mean(recalls), 2), np.round(
        np.std(recalls), 2
    )
    f1_mean, f1_std = np.round(np.mean(f1s), 2), np.round(np.std(f1s), 2)
    kappa_mean, kappa_std = np.round(np.mean(kappas), 2), np.round(np.std(kappas), 2)

    ## saving the results in a dataframe
    return pd.DataFrame(
        {
            "Balanced Accuracy": "{} +/- {}".format(accuracy_mean, accuracy_std),
            "Precision": "{} +/- {}".format(precision_mean, precision_std),
            "Recall": "{} +/- {}".format(recall_mean, recall_std),
            "F1": "{} +/- {}".format(f1_mean, f1_std),
            "Kappa": "{} +/- {}".format(kappa_mean, kappa_std),
        },
        index=[model_name],
    )


# 3. DATA DESCRIPTION
# Read the CSV file into a DataFrame
df1 = pd.read_csv("fraud_0.1origbase.csv")

# Display the first few rows of the DataFrame
print(df1.head())
# Display the last few rows of the DataFrame
print(df1.tail())

# Get the current column names of the DataFrame as a list
cols_old = df1.columns.tolist()

# Define a lambda function to convert a string to snake_case
snakecase = lambda x: inflection.underscore(x)

# Apply the snakecase function to each column name using map() and convert the result to a list
cols_new = list(map(snakecase, cols_old))

# Assign the new column names to the DataFrame
df1.columns = cols_new
# Display the updated column names
print(df1.columns)

# Print the number of rows in the DataFrame
print("Number of Rows: {}".format(df1.shape[0]))

# Print the number of columns in the DataFrame
print("Number of Cols: {}".format(df1.shape[1]))

# Display information about the DataFrame, including the data types and non-null counts of each column
print(df1.info())

# Calculate the proportion of missing values in each column of the DataFrame
print(df1.isna().mean())

# Map the values in the 'is_fraud' column to 'yes' for 1 and 'no' for 0
df1["is_fraud"] = df1["is_fraud"].map({1: "yes", 0: "no"})

# Map the values in the 'is_flagged_fraud' column to 'yes' for 1 and 'no' for 0
df1["is_flagged_fraud"] = df1["is_flagged_fraud"].map({1: "yes", 0: "no"})

# Select numerical attributes (excluding 'object' types)
num_attributes = df1.select_dtypes(exclude="object")

# Select categorical attributes (including only 'object' types)
cat_attributes = df1.select_dtypes(include="object")

# Calculate basic statistics for numerical attributes and transpose the result
describe = num_attributes.describe().T

# Calculate the range for each numerical attribute and add it as a new column
describe["range"] = (num_attributes.max() - num_attributes.min()).tolist()

# Calculate the coefficient of variation for each numerical attribute and add it as a new column
describe["variation coefficient"] = (
    num_attributes.std() / num_attributes.mean()
).tolist()

# Calculate the skewness for each numerical attribute and add it as a new column
describe["skew"] = num_attributes.skew().tolist()

# Calculate the kurtosis for each numerical attribute and add it as a new column
describe["kurtosis"] = num_attributes.kurtosis().tolist()

# Display the DataFrame containing the descriptive statistics
print(describe)

# Calculate summary statistics for categorical attributes
cat_attributes_summary = cat_attributes.describe()

# Print the summary statistics
print(cat_attributes_summary)

# 4. FEATURE ENGINEERING
# Create a copy of the DataFrame df1
df2 = df1.copy()

# Create a new column 'step_days' by dividing the 'step' column by 24
df2["step_days"] = df2["step"].apply(lambda i: i / 24)

# Create a new column 'step_weeks' by dividing the 'step' column by (24*7)
df2["step_weeks"] = df2["step"].apply(lambda i: i / (24 * 7))

# Create a new column 'diff_new_old_balance' by subtracting 'oldbalance_org' from 'newbalance_orig'
df2["diff_new_old_balance"] = df2["newbalance_orig"] - df2["oldbalance_org"]

# Create a new column 'diff_new_old_destiny' by subtracting 'oldbalance_dest' from 'newbalance_dest'
df2["diff_new_old_destiny"] = df2["newbalance_dest"] - df2["oldbalance_dest"]

# Create a new column 'name_orig' by taking the first character of each value in 'name_orig'
df2["name_orig"] = df2["name_orig"].apply(lambda i: i[0])

# Create a new column 'name_dest' by taking the first character of each value in 'name_dest'
df2["name_dest"] = df2["name_dest"].apply(lambda i: i[0])

# 5. EXPLORATORY DATA ANALYSIS
# Create a copy of the DataFrame df2
df3 = df2.copy()

# Create a copy of the DataFrame df3
df4 = df3.copy()

# COUNT PLOT OF 'is_fraud'
# Create a count plot of 'is_fraud' column
ax = sns.countplot(y="is_fraud", data=df4)

# Get the total number of data points
total = df4["is_fraud"].size

# Iterate over each bar in the plot
for p in ax.patches:
    # Calculate the percentage of data points represented by the bar
    percentage = " {:.1f}%".format(100 * p.get_width() / total)
    # Set the position for annotating the percentage
    x = p.get_x() + p.get_width() + 0.02
    y = p.get_y() + p.get_height() / 2
    # Annotate the bar with the percentage
    ax.annotate(percentage, (x, y))
# Add title and labels
plt.title("Distribution of Fraudulent Transactions")
plt.xlabel("Number of Transactions")
plt.ylabel("Fraudulent Status")

# Save the plot
save_plot(ax.get_figure(), "is_fraud_countplot.png")


# NUMERICAL VARIABLES
# Select numerical attributes (excluding 'object' types)
num_attributes = df4.select_dtypes(exclude="object")

# Get the column names of the numerical attributes
columns = num_attributes.columns.tolist()

# Initialize a subplot index
j = 1

# Create subplots for each numerical attribute
for column in columns:
    # Create a subplot with 2 rows and 5 columns, and select the current subplot
    plt.subplot(2, 5, j)
    # Plot the distribution of the current numerical attribute
    ax = sns.distplot(num_attributes[column])

    # Increment the subplot index
    j += 1

# Add title
plt.title("Subplots of Numberical Variables")

# Save the plot
save_plot(ax.get_figure(), "num_variables.png")

# CATEGORICAL VARIABLES
# Select categorical attributes (including only 'object' types)
cat_attributes = df4.select_dtypes(include="object")

# Get the column names of the categorical attributes
columns = cat_attributes.columns.tolist()

# Initialize a subplot index
j = 1

# Create subplots for each categorical attribute
for column in columns:
    # Create a subplot with 3 rows and 2 columns, and select the current subplot
    plt.subplot(3, 2, j)
    # Plot the count of each category in the current categorical attribute
    ax = sns.countplot(y=column, data=cat_attributes)

    # Get the total number of data points for the current categorical attribute
    total = cat_attributes[column].size
    # Annotate each bar with the percentage of data points it represents
    for p in ax.patches:
        percentage = " {:.1f}%".format(100 * p.get_width() / total)
        x = p.get_x() + p.get_width() + 0.02
        y = p.get_y() + p.get_height() / 2
        ax.annotate(percentage, (x, y))

    # Increment the subplot index
    j += 1

# Add title
plt.title("Subplots of Categorical Variables")

# Save the plot
save_plot(ax.get_figure(), "cat_variables.png")

# COUNT PLOT OF 'name_orig' FOR FRAUDULENT TRANSACTION
# Create a subset of df4 where 'is_fraud' is 'yes'
aux1 = df4[df4["is_fraud"] == "yes"]

# Create a count plot of 'name_orig' column in the subset aux1
ax = sns.countplot(y="name_orig", data=aux1)

# Add title and labels
plt.title("Distribution of 'name_orig' for Fraudulent Transactions")
plt.xlabel("Number of Transactions")
plt.ylabel("Originator Name ('name_orig')")

# Save the plot
save_plot(ax.get_figure(), "name_orig_countplot.png")

# COUNT PLOT OF 'name_dest' FOR FRAUDULENT TRANSACTIONS
# Create a count plot of 'name_dest' column in the subset aux1
ax = sns.countplot(y="name_dest", data=aux1)

# Add title and labels
plt.title("Distribution of 'name_dest' for Fraudulent Transactions")
plt.xlabel("Number of Transactions")
plt.ylabel("Destination Name ('name_dest')")

# Save the plot
save_plot(ax.get_figure(), "name_dest_countplot.png")

# BAR PLOT OF AVERAGE TRANSACTION AMOUNT BY FRAUDULENT STATUS
# Create a bar plot of 'amount' against 'is_fraud'
ax = sns.barplot(y="amount", x="is_fraud", data=df4)

# Add title and labels
plt.title("Average Transaction Amount by Fraudulent Status")
plt.xlabel("Fraudulent Status")
plt.ylabel("Average Transaction Amount")

# Save the plot
save_plot(ax.get_figure(), "avg_transaction_amount_barplot.png")

# NUMBER OF TRANSACTION FOR TYPES OF TRANSACTIONS
# Create a subset of df4 where 'is_fraud' is 'yes'
aux1 = df4[df4["is_fraud"] == "yes"]

# Create a count plot of 'type' column in the subset aux1
ax = sns.countplot(y="type", data=aux1)

# Get the total number of data points for the 'type' column in the subset
total = aux1["type"].size

# Annotate each bar with the percentage of data points it represents
for p in ax.patches:
    percentage = " {:.1f}%".format(100 * p.get_width() / total)
    x = p.get_x() + p.get_width() + 0.02
    y = p.get_y() + p.get_height() / 2
    ax.annotate(percentage, (x, y))

# Add title and labels
plt.title("Number of Transaction by Transaction Type")
plt.xlabel("Number of Transacitons")
plt.ylabel("Transaction Type")

# Save the plot
save_plot(ax.get_figure(), "num_transaction_vs_transaction_type.png")

# NUMBER OF TRANSACTION FOR ALL TYPES OF TRANSACTIONS
# Create a count plot of 'type' column with bars colored by 'is_fraud'
ax = sns.countplot(y="type", hue="is_fraud", data=df4)

# Get the total number of data points in the DataFrame
total = df4["type"].size

# Annotate each bar with the percentage of data points it represents
for p in ax.patches:
    percentage = " {:.1f}%".format(100 * p.get_width() / total)
    x = p.get_x() + p.get_width() + 0.02
    y = p.get_y() + p.get_height() / 2
    ax.annotate(percentage, (x, y))

# Add title and labels
plt.title("Number of Transaction by All Transaction Types")
plt.xlabel("Number of Transacitons")
plt.ylabel("Transaction Type")

# Save the plot
save_plot(ax.get_figure(), "num_transaction_vs_all_transaction_type.png")

# TRANSACTION AMOUNT FOR ALL TYPES OF TRANSACTIONS
# Create a bar plot of 'amount' against 'type'
ax = sns.barplot(y="type", x="amount", data=df4)

# Get the total number of data points in the DataFrame
total = df4["type"].size

# Annotate each bar with the percentage of data points it represents
for p in ax.patches:
    percentage = " {:.1f}%".format(100 * p.get_width() / total)
    x = p.get_x() + p.get_width() + 0.02
    y = p.get_y() + p.get_height() / 2
    ax.annotate(percentage, (x, y))

# Add title and labels
plt.title("Transaction Type by Transaction Amount")
plt.xlabel("Transaction Amount")
plt.ylabel("Transaction Type")

# Save the plot
save_plot(ax.get_figure(), "transaction_amount_vs_all_transaction_type.png")

# SCATTER PLOT OF 'step_days' vs. 'amount' FOR FRAUDULENT TRANSACTIONS
# Create a subset of df4 where 'is_fraud' is 'yes'
aux1 = df4[df4["is_fraud"] == "yes"]

# Create a scatter plot with a linear regression line for 'step_days' against 'amount'
ax = sns.regplot(x="step_days", y="amount", data=aux1)

# Add title and labels
plt.title(
    "Relationship between Days Since Start and Transaction Amount (Fraudulent Transactions)"
)
plt.xlabel("Days Since Start ('step_days')")
plt.ylabel("Transaction Amount")

# Save the plot
save_plot(ax.get_figure(), "step_days_vs_amount_scatterplot.png")

# CORRELATION HEATMAP FOR NUMERICAL ATTRIBUTES
# Calculate the correlation matrix for numerical attributes
corr = num_attributes.corr()

# Create a mask to hide the upper triangle of the heatmap
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True

# Create a heatmap of the correlation matrix
with sns.axes_style("white"):
    ax = sns.heatmap(
        corr, annot=True, mask=mask, vmin=-1, center=0, vmax=1, square=True
    )

# Add title and labels
plt.title("Correlation Heatmap for Numerical Attributes")
plt.xlabel("Numerical Attribute")
plt.ylabel("Numerical Attribute")

# Save the plot
save_plot(ax.get_figure(), "numerical_attributes_corr_heatmap.png")

# CRAMER'S V CORRELATION HEATMAP FOR CATEGORICAL VARIABLES
# Initialize an empty dictionary to store the Cramer's V correlation values
dict_corr = {}
# Get the column names of the categorical attributes
columns = cat_attributes.columns.tolist()

# Calculate Cramer's V correlation for each pair of categorical attributes
for column in columns:
    dict_corr[column] = {}
    for column2 in columns:
        dict_corr[column][column2] = calcCramerV(
            cat_attributes[column], cat_attributes[column2]
        )

# Create a DataFrame from the dictionary of Cramer's V correlation values
corr = pd.DataFrame(dict_corr)

# Create a mask to hide the upper triangle of the heatmap
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True

# Create a heatmap of the Cramer's V correlation matrix
with sns.axes_style("white"):
    ax = sns.heatmap(corr, annot=True, mask=mask, vmin=0, vmax=1, square=True)

# Add title and labels
plt.title("Cramer's V Correlation Heatmap for Categorical Attributes")
plt.xlabel("Categorical Attribute")
plt.ylabel("Categorical Attribute")

# Save the plot
save_plot(ax.get_figure(), "categorical_attributes_cramersv_heatmap.png")

# 6. DATA PREPARATION
# Create a copy of the DataFrame df4
df5 = df4.copy()

# Splitting into Train, Valid, and Test
X = df5.drop(
    columns=[
        "is_fraud",
        "is_flagged_fraud",
        "name_orig",
        "name_dest",
        "step_weeks",
        "step_days",
    ],
    axis=1,
)
y = df5["is_fraud"].map({"yes": 1, "no": 0})

# Splitting into temp and test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Splitting into train and valid
X_train, X_valid, y_train, y_valid = train_test_split(
    X_temp, y_temp, test_size=0.2, stratify=y_temp
)

# One-hot encode the 'type' column in the training data and validation data
ohe = OneHotEncoder(cols=["type"], use_cat_names=True)
X_train = ohe.fit_transform(X_train)
X_valid = ohe.transform(X_valid)

# One-hot encode the 'type' column in the temporary data and test data
X_temp = ohe.fit_transform(X_temp)
X_test = ohe.transform(X_test)

# Specify the numerical columns for Min-Max scaling
num_columns = [
    "amount",
    "oldbalance_org",
    "newbalance_orig",
    "oldbalance_dest",
    "newbalance_dest",
    "diff_new_old_balance",
    "diff_new_old_destiny",
]

# Initialize MinMaxScaler
mm = MinMaxScaler()

# Create copies of the datasets to avoid modifying the original datasets
X_params = X_temp.copy()

# Apply Min-Max scaling to the numerical columns in the training and validation data
X_train[num_columns] = mm.fit_transform(X_train[num_columns])
X_valid[num_columns] = mm.transform(X_valid[num_columns])

# Apply Min-Max scaling to the numerical columns in the temporary data and test data
X_params[num_columns] = mm.fit_transform(X_temp[num_columns])
X_test[num_columns] = mm.transform(X_test[num_columns])

# 6. FEATURE SELECTION
# Convert X_params and y_temp to numpy arrays for Boruta processing
""" X_boruta = X_params.values
y_boruta = y_temp.values.ravel()

# Initialize BorutaPy with a Random Forest classifier and 'auto' n_estimators
boruta = BorutaPy(RandomForestClassifier(), n_estimators="auto")

# Fit Boruta to the data
boruta.fit(X_boruta, y_boruta)

# Get the selected features as a list of boolean values
cols_selected_boruta = boruta.support_.tolist()

# Get the names of the selected columns
columns_selected = X_params.loc[:, cols_selected_boruta].columns.tolist() """

# List of final columns selected based on domain knowledge or previous analysis
final_columns_selected = [
    "step",
    "oldbalance_org",
    "newbalance_orig",
    "newbalance_dest",
    "diff_new_old_balance",
    "diff_new_old_destiny",
    "type_TRANSFER",
]

# 7. MACHINE LEARNING MODELING
# Select the final columns for training and validation data
X_train_cs = X_train[final_columns_selected]
X_valid_cs = X_valid[final_columns_selected]

# Select the final columns for temporary and test data
X_temp_cs = X_temp[final_columns_selected]
X_test_cs = X_test[final_columns_selected]

# Select the final columns for X_params
X_params_cs = X_params[final_columns_selected]

# Initialize and fit a Dummy Classifier
dummy = DummyClassifier()
dummy.fit(X_train_cs, y_train)

# Make predictions on the validation set
y_pred = dummy.predict(X_valid_cs)

# Calculate and display the Dummy Classifier results
dummy_results = ml_scores("dummy", y_valid, y_pred)
print(dummy_results)

# Print the classification report for the Dummy Classifier
print(classification_report(y_valid, y_pred))

# Calculate cross-validation results for the Dummy Classifier
dummy_cv = ml_cv_results("Dummy", DummyClassifier(), X_temp, y_temp)
print(dummy_cv)

# Initialize and fit a Logistic Regression model
lg = LogisticRegression()
lg.fit(X_train_cs, y_train)

# Make predictions on the validation set
y_pred = lg.predict(X_valid_cs)

# Calculate and display the Logistic Regression results
lg_results = ml_scores("Logistic Regression", y_valid, y_pred)
print(lg_results)

# Print the classification report for the Logistic Regression model
print(classification_report(y_valid, y_pred))

# Calculate cross-validation results for the Logistic Regression model
lg_cv = ml_cv_results("Logistic Regression", LogisticRegression(), X_temp_cs, y_temp)
print(lg_cv)

# Initialize and fit a K Nearest Neighbors model
knn = KNeighborsClassifier()
knn.fit(X_train_cs, y_train)

# Make predictions on the validation set
y_pred = knn.predict(X_valid_cs)

# Calculate and display the K Nearest Neighbors results
knn_results = ml_scores("K Nearest Neighbors", y_valid, y_pred)
print(knn_results)

# Print the classification report for the K Nearest Neighbors model
print(classification_report(y_valid, y_pred))

# Calculate cross-validation results for the K Nearest Neighbors model
knn_cv = ml_cv_results("K Nearest Neighbors", KNeighborsClassifier(), X_temp_cs, y_temp)
print(knn_cv)

# Initialize and fit a Support Vector Machine model
svm = SVC()
svm.fit(X_train_cs, y_train)

# Make predictions on the validation set
y_pred = svm.predict(X_valid_cs)

# Calculate and display the SVM results
svm_results = ml_scores("SVM", y_valid, y_pred)
print(svm_results)

# Print the classification report for the SVM model
print(classification_report(y_valid, y_pred))

# Calculate cross-validation results for the SVM model
svm_cv = ml_cv_results("SVM", SVC(), X_temp_cs, y_temp)
print(svm_cv)

# Initialize and fit a Random Forest model with balanced class weights
rf = RandomForestClassifier(class_weight="balanced")
rf.fit(X_train_cs, y_train)

# Make predictions on the validation set
y_pred = rf.predict(X_valid_cs)

# Calculate and display the Random Forest results
rf_results = ml_scores("Random Forest", y_valid, y_pred)
print(rf_results)

# Print the classification report for the Random Forest model
print(classification_report(y_valid, y_pred))

# Calculate cross-validation results for the Random Forest model
rf_cv = ml_cv_results("Random Forest", RandomForestClassifier(), X_temp_cs, y_temp)
print(rf_cv)

# Initialize and fit an XGBoost model
xgb = XGBClassifier()
xgb.fit(X_train_cs, y_train)

# Make predictions on the validation set
y_pred = xgb.predict(X_valid_cs)

# Calculate and display the XGBoost results
xgb_results = ml_scores("XGBoost", y_valid, y_pred)
print(xgb_results)

# Print the classification report for the XGBoost model
print(classification_report(y_valid, y_pred))

# Calculate cross-validation results for the XGBoost model
xgb_cv = ml_cv_results("XGBoost", XGBClassifier(), X_temp_cs, y_temp)
print(xgb_cv)

# Initialize and fit a LightGBM model
lightgbm = LGBMClassifier()
lightgbm.fit(X_train_cs, y_train)

# Make predictions on the validation set
y_pred = lightgbm.predict(X_valid_cs)

# Calculate and display the LightGBM results
lightgbm_results = ml_scores("LightGBM", y_valid, y_pred)
print(lightgbm_results)

# Print the classification report for the LightGBM model
print(classification_report(y_valid, y_pred))

# Calculate cross-validation results for the LightGBM model
lightgbm_cv = ml_cv_results("LightGBM", LGBMClassifier(), X_temp_cs, y_temp)
print(lightgbm_cv)

# 7.1 COMPARING MODEL'S PERFORMANCE
# Concatenate the results of all models
modeling_performance = pd.concat(
    [
        dummy_results,
        lg_results,
        knn_results,
        rf_results,
        xgb_results,
        lightgbm_results,
        svm_results,
    ]
)

# Sort the results by F1 score in ascending order
modeling_performance.sort_values(by="F1", ascending=True)

# Concatenate the cross-validation results of all models
modeling_performance_cv = pd.concat(
    [dummy_cv, lg_cv, knn_cv, rf_cv, xgb_cv, lightgbm_cv, svm_cv]
)

# Sort the cross-validation results by F1 score in ascending order
modeling_performance_cv.sort_values(by="F1", ascending=True)

# 8. HYPERPARAMETER FINE TUNING
# Define a custom scoring function for F1 score
f1 = make_scorer(f1_score)

# Define the parameter grid for GridSearchCV
params = {
    "booster": ["gbtree", "gblinear", "dart"],
    "eta": [0.3, 0.1, 0.01],
    "scale_pos_weight": [1, 774, 508, 99],
}

# Perform GridSearchCV with XGBClassifier using F1 score as the scoring metric
gs = GridSearchCV(
    XGBClassifier(), param_grid=params, scoring=f1, cv=StratifiedKFold(n_splits=5)
)

# Fit the GridSearchCV object on the data
gs.fit(X_params_cs, y_temp)

# Get the best parameters from the GridSearchCV
best_params = gs.best_params_
print(best_params)

# Best parameters
best_params = {"booster": "gbtree", "eta": 0.3, "scale_pos_weight": 1}

# Best F1 score
gs.best_score_

# Initialize and fit an XGBoost model with the best parameters from GridSearchCV
xgb_gs = XGBClassifier(
    booster=best_params["booster"],
    eta=best_params["eta"],
    scale_pos_weight=best_params["scale_pos_weight"],
)
xgb_gs.fit(X_train_cs, y_train)

# Make predictions on the validation set using the GridSearchCV optimized XGBoost model
y_pred = xgb_gs.predict(X_valid_cs)

# Calculate the scores for the GridSearchCV optimized XGBoost model
xgb_gs_results = ml_scores("XGBoost GS", y_valid, y_pred)
print(xgb_gs_results)

# Calculate cross-validation results for the GridSearchCV optimized XGBoost model
xgb_gs_cv = ml_cv_results("XGBoost GS", xgb_gs, X_temp_cs, y_temp)
print(xgb_gs_cv)

# 9. FINAL MODEL
# Initialize and fit the final XGBoost model with the best parameters from GridSearchCV using all data
final_model = XGBClassifier(
    booster=best_params["booster"],
    eta=best_params["eta"],
    scale_pos_weight=best_params["scale_pos_weight"],
)
final_model.fit(X_params_cs, y_temp)

# Make predictions on the test set using the final model
y_pred = final_model.predict(X_test_cs)

# Calculate scores for the final model on the unseen test data
unseen_scores = ml_scores("unseen", y_test, y_pred)
print(unseen_scores)

# 10. TESTING
# Create a DataFrame with test data and predictions
df_test = df5.loc[X_test.index, :]
df_test["predictions"] = y_pred

# Calculate the potential amount the company can receive by detecting fraud transactions
aux1 = df_test[(df_test["is_fraud"] == "yes") & (df_test["predictions"] == 1)]
receives = aux1["amount"].sum() * 0.25
print("The company can receive %.2f detecting fraud transactions." % (receives))

# Calculate the potential amount the company can receive for wrong decisions
aux1 = df_test[(df_test["is_fraud"] == "no") & (df_test["predictions"] == 1)]
receives = aux1["amount"].sum() * 0.05
print("For wrong decisions, the company can receive %.2f." % (receives))

# Calculate the amount the company must return for incorrect predictions
aux1 = df_test[(df_test["is_fraud"] == "yes") & (df_test["predictions"] == 0)]
returns = aux1["amount"].sum()
print("However, the company must return the amount of %.2f." % (returns))

# Print the performance metrics for unseen data
print(
    "For unseen data, the values of balanced accuracy is equal %.2f and precision is equal %.2f."
    % (unseen_scores["Balanced Accuracy"], unseen_scores["Precision"])
)
print(
    "The model can detect 0.851 +/- 0.023 of the fraud. However it detected 0.84 of the frauds from unseen data."
)

# Calculate the potential revenue for the company
aux1 = df_test[(df_test["is_fraud"] == "yes") & (df_test["predictions"] == 1)]
receives = aux1["amount"].sum() * 0.25

aux2 = df_test[(df_test["is_fraud"] == "no") & (df_test["predictions"] == 1)]
receives2 = aux2["amount"].sum() * 0.05

print(
    "Using the model, the company can generate revenue of %.2f."
    % (receives + receives2)
)

# Calculate the potential revenue for the company using the current method
aux3 = df_test[(df_test["is_fraud"] == "yes") & (df_test["is_flagged_fraud"] == "yes")]
curr_receives = aux3["amount"].sum() * 0.25

aux4 = df_test[(df_test["is_fraud"] == "no") & (df_test["is_flagged_fraud"] == "yes")]
curr_receives2 = aux4["amount"].sum() * 0.05

print(
    "However, with the current method, the revenue is %.2f."
    % (curr_receives + curr_receives2)
)

# Calculate the potential loss for the company from wrong classifications
aux1 = df_test[(df_test["is_fraud"] == "yes") & (df_test["predictions"] == 0)]
loss = aux1["amount"].sum()

print("For wrong classifications, the company must return the amount of %.2f." % (loss))

# Calculate the potential loss for the company from wrong classifications using the current method
aux1 = df_test[(df_test["is_fraud"] == "yes") & (df_test["is_flagged_fraud"] == "no")]
curr_loss = aux1["amount"].sum()

print(
    "For wrong classifications using the current method, the company must return the amount of %.2f."
    % (curr_loss)
)

# Calculate the expected profit for the company from using the model
expected_profit = receives + receives2 - loss

print("The company can expect a profit of %.2f." % expected_profit)

# Calculate the expected profit for the company from using the current method
current_method_profit = curr_receives + curr_receives - curr_loss

print("Using the current method, the profit is %.2f." % current_method_profit)

# 11. DEPLOYMENT
# Define the final model with the best parameters
final_model = XGBClassifier(
    booster=best_params["booster"],
    eta=best_params["eta"],
    scale_pos_weight=best_params["scale_pos_weight"],
)

# Fit the final model on the entire dataset
final_model.fit(X_params_cs, y_temp)

# Save the final model to a file
joblib.dump(final_model, "../models/model_cycle1.joblib")

# Create a MinMaxScaler instance and fit it to the data
scaler = MinMaxScaler()
scaler.fit(X_params_cs, y_temp)

# Save the fitted scaler object to a file
joblib.dump(scaler, "../parameters/minmaxscaler_cycle1.joblib")

# Save the OneHotEncoder object to a file
joblib.dump(ohe, "../parameters/onehotencoder_cycle1.joblib")
