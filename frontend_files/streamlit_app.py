
import streamlit as st
import pandas as pd
import requests
import json
import os
from pathlib import Path

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="SuperKart Sales Forecasting",
    page_icon="🛒",
    layout="wide"
)

API_URL = os.getenv(
    "BACKEND_URL",
    "http://localhost:7860"
).rstrip("/")

# Load feature schema
@st.cache_data
def load_schema():
    schema_path = Path(__file__).resolve().parent / "schema.json"
    with open(schema_path, encoding="utf-8") as f:
        return json.load(f)

schema = load_schema()
features = schema["features"]
numeric = schema["numeric"]
options = schema["options"]
example = schema["example"]

# -------------------------------
# Header
# -------------------------------
st.title("🛒 SuperKart Sales Forecasting")
st.markdown(
    "### AI-Powered Retail Sales Prediction"
)

st.info(
    "Forecast product-store sales using a tuned "
    "Random Forest regression model."
)

st.markdown("---")

# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.header("📊 Model Information")
    st.metric("Test R²", "93.25%")
    st.metric("Test RMSE", "277.55")
    st.metric("Test MAPE", "3.75%")

    st.markdown("---")
    st.caption(
        "Model: Tuned Random Forest Regression"
    )

# -------------------------------
# Prediction tabs
# -------------------------------
single_tab, batch_tab = st.tabs([
    "🎯 Single Prediction",
    "📁 Batch Prediction"
])

# -------------------------------
# Single prediction
# -------------------------------
with single_tab:

    st.subheader("Enter Product and Store Details")

    with st.form("sales_prediction_form"):

        col1, col2 = st.columns(2)
        inputs = {}

        for index, feature in enumerate(features):

            container = col1 if index % 2 == 0 else col2

            with container:

                label = feature.replace("_", " ")

                if feature in numeric:

                    default = float(example[feature])

                    if feature == "Store_Establishment_Year":
                        inputs[feature] = st.number_input(
                            label,
                            min_value=1900,
                            max_value=2100,
                            value=int(default),
                            step=1
                        )
                    else:
                        inputs[feature] = st.number_input(
                            label,
                            value=default,
                            format="%.4f"
                        )

                else:

                    choices = options.get(feature, [])

                    if not choices:
                        choices = [str(example[feature])]

                    default_value = str(example[feature])

                    default_index = (
                        choices.index(default_value)
                        if default_value in choices else 0
                    )

                    inputs[feature] = st.selectbox(
                        label,
                        options=choices,
                        index=default_index
                    )

        submitted = st.form_submit_button(
            "🔮 Predict Sales",
            use_container_width=True
        )

    if submitted:

        try:
            response = requests.post(
                f"{API_URL}/v1/predict",
                json=inputs,
                timeout=60
            )

            response.raise_for_status()

            result = response.json()

            prediction = float(result["prediction"])

            st.success("Prediction completed successfully!")

            st.metric(
                "📈 Predicted Product-Store Sales",
                f"{prediction:,.2f}"
            )

            with st.expander("View Submitted Inputs"):
                st.json(inputs)

        except requests.RequestException as exc:
            st.error(
                f"Backend API connection failed: {exc}"
            )

        except (KeyError, TypeError, ValueError) as exc:
            st.error(
                f"Unexpected prediction response: {exc}"
            )

# -------------------------------
# Batch prediction
# -------------------------------
with batch_tab:

    st.subheader("Upload CSV for Batch Predictions")

    st.write(
        "Upload a CSV file containing the required "
        "product and store attributes."
    )

    template = pd.DataFrame(
        [example],
        columns=features
    )

    st.download_button(
        "📥 Download Sample CSV Template",
        data=template.to_csv(index=False),
        file_name="superkart_input_template.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        uploaded_bytes = uploaded_file.getvalue()

        try:
            preview_df = pd.read_csv(
                uploaded_file
            )

            st.write("### 📋 Uploaded Data Preview")
            st.dataframe(
                preview_df.head(),
                use_container_width=True
            )

            missing_columns = [
                feature for feature in features
                if feature not in preview_df.columns
            ]

            if missing_columns:
                st.error(
                    "Missing required columns: "
                    + ", ".join(missing_columns)
                )

            else:

                st.success(
                    "All required columns are present."
                )

                if st.button(
                    "🚀 Generate Batch Predictions",
                    use_container_width=True
                ):

                    response = requests.post(
                        f"{API_URL}/v1/predictbatch",
                        files={
                            "file": (
                                uploaded_file.name,
                                uploaded_bytes,
                                "text/csv"
                            )
                        },
                        timeout=180
                    )

                    response.raise_for_status()

                    result = response.json()

                    # Supports dictionary indexed by row number
                    # or a list of prediction values.
                    if isinstance(result, dict):
                        if "predictions" in result:
                            predictions = result["predictions"]
                        else:
                            predictions = [
                                result[str(i)]
                                for i in range(len(preview_df))
                            ]
                    else:
                        predictions = result

                    if len(predictions) != len(preview_df):
                        raise ValueError(
                            "Prediction count does not match CSV rows."
                        )

                    output_df = preview_df.copy()
                    output_df["Predicted_Sales"] = predictions

                    st.success(
                        f"Predictions generated for "
                        f"{len(output_df)} records!"
                    )

                    st.metric(
                        "Average Predicted Sales",
                        f"{pd.to_numeric(output_df['Predicted_Sales']).mean():,.2f}"
                    )

                    st.dataframe(
                        output_df,
                        use_container_width=True
                    )

                    st.download_button(
                        "📥 Download Prediction Results",
                        data=output_df.to_csv(index=False),
                        file_name="superkart_predictions.csv",
                        mime="text/csv"
                    )

        except (
            requests.RequestException,
            KeyError,
            ValueError,
            TypeError,
            pd.errors.ParserError
        ) as exc:

            st.error(f"Batch prediction failed: {exc}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption(
    "SuperKart Sales Forecasting | "
    "Machine Learning Model Deployment Project"
)
