import streamlit as st
import pandas as pd
import joblib

# 1. Cache models & feature list for fast reloads
@st.cache_data
def load_pipeline_and_features():
    pipeline = joblib.load("models/rf_model.pkl")
    expected_feats = joblib.load("models/expected_features.pkl")
    return pipeline, expected_feats

pipeline, expected_features = load_pipeline_and_features()

# 2. App header
st.set_page_config(page_title="Student Performance Dashboard", layout="wide")
st.title("📊 Student Performance Dashboard")

# 3. File uploader: XLSX or CSV
uploaded_file = st.file_uploader("Upload .xlsx or .csv", type=["xlsx", "csv"])
if uploaded_file:
    file_ext = uploaded_file.name.split(".")[-1].lower()
    if file_ext == "csv":
        df_input = pd.read_csv(uploaded_file)
    else:
        try:
            df_input = pd.read_excel(uploaded_file, engine="openpyxl")
        except ImportError:
            st.error("Excel uploads require the openpyxl library. Please add `openpyxl` to your requirements.")
            st.stop()

    # 4. Preview raw data
    st.subheader("📝 Input Data Preview")
    st.dataframe(df_input)

    # 5. Align features & predict
    df_feats = df_input.reindex(columns=expected_features, fill_value=0)
    preds = pipeline.predict(df_feats)
    df_input["acad_pred"], df_input["atten_pred"] = zip(*preds)

    # 6. Show predictions
    st.subheader("🎯 Predictions")
    st.dataframe(df_input)

    # 7. Download results
    csv_data = df_input.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Predictions as CSV",
        data=csv_data,
        file_name="student_predictions.csv",
        mime="text/csv"
    )
