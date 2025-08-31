import streamlit as st
import pandas as pd
from rf_pipeline import preprocess, load_pipeline

# Cache loading model + feature list
@st.cache_data
def get_pipeline():
    pipeline, features = load_pipeline()
    return pipeline, features

pipeline, expected_features = get_pipeline()

st.set_page_config(page_title="Student Performance Dashboard", layout="wide")
st.title("📊 Student Performance Dashboard")

uploaded_file = st.file_uploader("Upload .xlsx or .csv", type=["xlsx","csv"])
if not uploaded_file:
    st.info("Please upload a student survey file first.")
    st.stop()

# 1. Read file
ext = uploaded_file.name.rsplit(".",1)[-1].lower()
if ext == "csv":
    df_raw = pd.read_csv(uploaded_file)
else:
    df_raw = pd.read_excel(uploaded_file, engine="openpyxl")

st.subheader("📝 Raw Input Preview")
st.dataframe(df_raw)

# 2. Preprocess exactly as during training
df_pre = preprocess(df_raw)

# 3. Align to expected features (fill missing with 0)
df_pre = df_pre.reindex(columns=expected_features, fill_value=0)

# 4. Optional debug expander
with st.expander("🔍 Debug: Preprocessed Features", expanded=False):
    st.write("Columns fed to model:", df_pre.columns.tolist())
    st.dataframe(df_pre.describe())

# 5. Predict
with st.spinner("Running predictions…"):
    preds = pipeline.predict(df_pre)

# 6. Merge predictions back into raw DataFrame
df_raw["acad_pred"], df_raw["atten_pred"] = zip(*preds)

st.subheader("🎯 Predictions")
st.dataframe(df_raw)

# 7. Download CSV
csv = df_raw.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download with Predictions", data=csv,
                   file_name="student_predictions.csv",
                   mime="text/csv")
