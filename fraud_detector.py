import inflection
import joblib
import pandas as pd


class FraudDetector:
    def __init__(self):
        # Load preprocessing parameters
        self.minmaxscaler = joblib.load("minmaxscaler_cycle1.joblib")
        self.onehotencoder = joblib.load("onehotencoder_cycle1.joblib")

    def data_cleaning(self, df):
        """
        Clean column names using snake case.

        Args:
        - df (DataFrame): Input dataframe with unclean column names.

        Returns:
        - DataFrame: DataFrame with cleaned column names.
        """
        cols_old = df.columns.tolist()
        snakecase = lambda i: inflection.underscore(i)
        cols_new = list(map(snakecase, cols_old))
        df.columns = cols_new
        return df

    def feature_engineering(self, df):
        """
        Perform feature engineering on the input dataframe.

        Args:
        - df (DataFrame): Input dataframe.

        Returns:
        - DataFrame: Transformed dataframe after feature engineering.
        """
        df["step_days"] = df["step"].apply(lambda i: i / 24)
        df["step_weeks"] = df["step"].apply(lambda i: i / (24 * 7))
        df["diff_new_old_balance"] = df["newbalance_orig"] - df["oldbalance_org"]
        df["diff_new_old_destiny"] = df["newbalance_dest"] - df["oldbalance_dest"]
        df["name_orig"] = df["name_orig"].apply(lambda i: i[0])
        df["name_dest"] = df["name_dest"].apply(lambda i: i[0])
        return df.drop(
            columns=["name_orig", "name_dest", "step_weeks", "step_days"], axis=1
        )

    def data_preparation(self, df):
        """
        Prepare the data for model prediction.

        Args:
        - df (DataFrame): Input dataframe.

        Returns:
        - DataFrame: Processed dataframe ready for prediction.
        """
        df = self.onehotencoder.transform(df)
        num_columns = [
            "amount",
            "oldbalance_org",
            "newbalance_orig",
            "oldbalance_dest",
            "newbalance_dest",
            "diff_new_old_balance",
            "diff_new_old_destiny",
        ]
        df[num_columns] = self.minmaxscaler.transform(df[num_columns])
        final_columns_selected = [
            "step",
            "oldbalance_org",
            "newbalance_orig",
            "newbalance_dest",
            "diff_new_old_balance",
            "diff_new_old_destiny",
            "type_TRANSFER",
        ]
        return df[final_columns_selected]

    def get_prediction(self, model, original_data, test_data):
        """
        Get predictions from the model.

        Args:
        - model: Trained machine learning model.
        - original_data (DataFrame): Original data with predictions appended.
        - test_data (DataFrame): Dataframe containing test data.

        Returns:
        - str: JSON string containing original data with predictions.
        """
        pred = model.predict(test_data)
        original_data["prediction"] = pred
        return original_data.to_json(orient="records", date_format="iso")
