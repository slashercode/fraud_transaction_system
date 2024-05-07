import joblib
import pandas as pd
from flask import Flask, Response, request

from fraud_detector import FraudDetector

# Loading model
model = joblib.load("model_cycle1.joblib")

# Initialize API
app = Flask(__name__)


@app.route("/fraud/predict", methods=["POST"])
def fraud_predict():
    test_json = request.get_json()

    if test_json:  # There is data
        if isinstance(test_json, dict):  # Unique example
            test_raw = pd.DataFrame(test_json, index=[0])
        else:  # Multiple examples
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())

        # Instantiate Fraud class
        pipeline = FraudDetector()

        # Data cleaning
        df1 = pipeline.data_cleaning(test_raw)

        # Feature engineering
        df2 = pipeline.feature_engineering(df1)

        # Data preparation
        df3 = pipeline.data_preparation(df2)

        # Prediction
        df_response = pipeline.get_prediction(model, test_raw, df3)

        return df_response

    else:
        return Response("{}", status=200, mimetype="application/json")


if __name__ == "__main__":
    app.run("0.0.0.0")
