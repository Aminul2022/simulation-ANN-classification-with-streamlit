from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


# ---------------------------------------------------------
# Application configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("Customer Churn Prediction")
st.write(
    "Enter the customer information below to estimate the probability "
    "that the customer will churn."
)


# ---------------------------------------------------------
# File locations
# ---------------------------------------------------------
# This ensures files are loaded relative to app.py rather
# than relative to the terminal's current working directory.
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model.h5"
GENDER_ENCODER_PATH = BASE_DIR / "label_encoder_gender.pkl"
GEOGRAPHY_ENCODER_PATH = BASE_DIR / "onehot_encoder_geo.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"


# ---------------------------------------------------------
# Load trained artefacts once
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


@st.cache_resource
def load_pickle(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "rb") as file:
        return pickle.load(file)


try:
    model = load_model()

    label_encoder_gender = load_pickle(
        GENDER_ENCODER_PATH
    )

    onehot_encoder_geo = load_pickle(
        GEOGRAPHY_ENCODER_PATH
    )

    scaler = load_pickle(
        SCALER_PATH
    )

except Exception as error:
    st.error("The trained model or preprocessing files could not be loaded.")
    st.exception(error)
    st.stop()


# ---------------------------------------------------------
# User input form
# ---------------------------------------------------------
with st.form("customer_input_form"):

    geography = st.selectbox(
        "Geography",
        options=onehot_encoder_geo.categories_[0].tolist()
    )

    gender = st.selectbox(
        "Gender",
        options=label_encoder_gender.classes_.tolist()
    )

    age = st.slider(
        "Age",
        min_value=18,
        max_value=92,
        value=35
    )

    tenure = st.slider(
        "Tenure",
        min_value=0,
        max_value=10,
        value=5
    )

    balance = st.number_input(
        "Balance",
        min_value=0.0,
        value=0.0,
        step=100.0
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=650,
        step=1
    )

    estimated_salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        value=50000.0,
        step=100.0
    )

    num_of_products = st.slider(
        "Number of Products",
        min_value=1,
        max_value=4,
        value=1
    )

    has_cr_card = st.selectbox(
        "Has Credit Card",
        options=[0, 1],
        format_func=lambda value: "Yes" if value == 1 else "No"
    )

    is_active_member = st.selectbox(
        "Is Active Member",
        options=[0, 1],
        format_func=lambda value: "Yes" if value == 1 else "No"
    )

    submitted = st.form_submit_button(
        "Predict churn"
    )


# ---------------------------------------------------------
# Prepare data and make prediction
# ---------------------------------------------------------
if submitted:

    try:
        gender_encoded = int(
            label_encoder_gender.transform([gender])[0]
        )

        input_data = pd.DataFrame({
            "CreditScore": [float(credit_score)],
            "Gender": [gender_encoded],
            "Age": [float(age)],
            "Tenure": [float(tenure)],
            "Balance": [float(balance)],
            "NumOfProducts": [float(num_of_products)],
            "HasCrCard": [float(has_cr_card)],
            "IsActiveMember": [float(is_active_member)],
            "EstimatedSalary": [float(estimated_salary)]
        })

        # One-hot encode Geography
        geography_encoded = onehot_encoder_geo.transform(
            [[geography]]
        )

        # Handles both sparse and dense encoder output
        if hasattr(geography_encoded, "toarray"):
            geography_encoded = geography_encoded.toarray()

        geography_columns = (
            onehot_encoder_geo.get_feature_names_out(
                ["Geography"]
            )
        )

        geography_df = pd.DataFrame(
            geography_encoded,
            columns=geography_columns
        )

        # Combine numerical and one-hot encoded columns
        input_data = pd.concat(
            [
                input_data.reset_index(drop=True),
                geography_df.reset_index(drop=True)
            ],
            axis=1
        )

        # Enforce exactly the same feature order used
        # when the StandardScaler was trained.
        if hasattr(scaler, "feature_names_in_"):
            expected_columns = list(
                scaler.feature_names_in_
            )

            missing_columns = [
                column
                for column in expected_columns
                if column not in input_data.columns
            ]

            unexpected_columns = [
                column
                for column in input_data.columns
                if column not in expected_columns
            ]

            if missing_columns:
                raise ValueError(
                    "Missing model input columns: "
                    + ", ".join(missing_columns)
                )

            if unexpected_columns:
                input_data = input_data.drop(
                    columns=unexpected_columns
                )

            input_data = input_data[expected_columns]

        # Scale input using the fitted training scaler
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(
            input_data_scaled,
            verbose=0
        )

        prediction_probability = float(
            np.squeeze(prediction)
        )

        st.subheader("Prediction result")

        st.metric(
            label="Churn probability",
            value=f"{prediction_probability:.2%}"
        )

        st.progress(
            min(max(prediction_probability, 0.0), 1.0)
        )

        if prediction_probability > 0.5:
            st.warning(
                "The customer is likely to churn."
            )
        else:
            st.success(
                "The customer is not likely to churn."
            )

        with st.expander("View prepared model input"):
            st.dataframe(input_data)

    except Exception as error:
        st.error(
            "The prediction could not be completed. "
            "Check that the saved preprocessing objects match "
            "the preprocessing used during model training."
        )
        st.exception(error)