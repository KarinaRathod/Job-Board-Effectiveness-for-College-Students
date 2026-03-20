import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="Job Platform AI Dashboard", layout="wide")
st.title("🎓 Job Search Platform Effectiveness + AI Prediction")

DATA_FILE = "job_search_platform_efficacy_100k.csv"

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    if not os.path.exists(DATA_FILE):
        return None
    try:
        df = pd.read_csv(DATA_FILE)
        if len(df.columns) == 1:
            df = pd.read_csv(DATA_FILE, sep="\t")
    except:
        df = pd.read_csv(DATA_FILE, delim_whitespace=True)
    return df

df = load_data()

if df is None:
    st.error(f"❌ File '{DATA_FILE}' not found in folder")
    st.stop()

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.subheader("📌 Columns")
st.sidebar.write(list(df.columns))

# -------------------------------
# CLEANING
# -------------------------------
df.fillna("Unknown", inplace=True)

def find_col(keywords):
    for col in df.columns:
        if any(k.lower() in col.lower() for k in keywords):
            return col
    return None

job_col = find_col(["platform", "job", "board"])
status_col = find_col(["status", "result", "offer", "selected"])
num_cols = df.select_dtypes(include=np.number).columns.tolist()

# -------------------------------
# KPIs
# -------------------------------
st.subheader("📊 Key Metrics")

c1, c2, c3 = st.columns(3)
c1.metric("Total Records", len(df))

if status_col:
    success = df[status_col].astype(str).str.lower().isin(["selected","offer","hired"])
    success_rate = success.mean() * 100
    c2.metric("Success Rate", f"{success_rate:.2f}%")
else:
    success_rate = 0
    c2.metric("Success Rate", "N/A")

c3.metric("Columns", len(df.columns))

# -------------------------------
# FILTER
# -------------------------------
filtered = df.copy()

if job_col:
    options = df[job_col].unique()
    selected = st.sidebar.multiselect("Platform", options, default=options)
    filtered = filtered[filtered[job_col].isin(selected)]

# -------------------------------
# PLATFORM PERFORMANCE
# -------------------------------
st.subheader("🎯 Platform Performance")

if job_col and status_col:
    perf = filtered.groupby(job_col)[status_col].apply(
        lambda x: x.astype(str).str.lower().isin(["selected","offer","hired"]).mean()
    ).reset_index()

    perf.columns = ["Platform", "Success Rate"]

    st.plotly_chart(px.bar(perf, x="Platform", y="Success Rate"),
                    use_container_width=True)

    best_platform = perf.sort_values("Success Rate", ascending=False).iloc[0]["Platform"]
    st.success(f"🏆 Best Platform: {best_platform}")

# -------------------------------
# OUTCOME
# -------------------------------
if status_col:
    st.subheader("📈 Application Outcomes")
    st.plotly_chart(px.pie(filtered, names=status_col),
                    use_container_width=True)

# -------------------------------
# SEGMENTATION
# -------------------------------
st.subheader("🧠 Student Segmentation")

if len(num_cols) >= 2:
    data = filtered[num_cols].dropna()

    if len(data) > 10:
        k = st.slider("Clusters", 2, 5, 3)

        model = KMeans(n_clusters=k, random_state=42)
        data["Cluster"] = model.fit_predict(data)

        st.plotly_chart(
            px.scatter(data, x=num_cols[0], y=num_cols[1],
                       color=data["Cluster"].astype(str)),
            use_container_width=True
        )

        st.dataframe(data.groupby("Cluster").mean())

# -------------------------------
# PLACEMENT PREDICTION (FIXED)
# -------------------------------
st.subheader("🎯 Placement Prediction")

if status_col:

    df_model = df.copy()
    df_model[status_col] = df_model[status_col].astype(str).str.lower()

    df_model["Target"] = df_model[status_col].apply(
        lambda x: 1 if x in ["selected","offer","hired"] else 0
    )

    # Encode categorical
    le = LabelEncoder()
    for col in df_model.columns:
        if df_model[col].dtype == "object":
            df_model[col] = le.fit_transform(df_model[col].astype(str))

    # Features
    X = df_model.drop(columns=[status_col, "Target"]).select_dtypes(include=np.number)
    y = df_model["Target"]

    if len(X.columns) > 1:

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        acc = model.score(X_test, y_test)
        st.success(f"✅ Model Accuracy: {acc:.2f}")

        st.subheader("🧾 Predict New Student")

        input_data = {}

        # Limit UI inputs (but still keep model correct)
        for col in X.columns[:8]:
            input_data[col] = st.number_input(col, float(X[col].mean()))

        # Fill missing columns automatically
        input_df = pd.DataFrame([input_data])

        for col in X.columns:
            if col not in input_df:
                input_df[col] = X[col].mean()

        input_df = input_df[X.columns]

        if st.button("Predict Placement"):
            pred = model.predict(input_df)[0]

            if pred == 1:
                st.success("🎉 Likely Selected")
            else:
                st.error("⚠️ Low Chance")

# -------------------------------
# CHATBOT
# -------------------------------
st.subheader("🤖 AI Assistant")

q = st.text_input("Ask about insights")

if q:
    q = q.lower()
    if "best" in q:
        st.info(f"Best platform: {best_platform}")
    elif "success" in q:
        st.info(f"Success rate: {success_rate:.2f}%")
    else:
        st.info("Try: best platform / success rate")

# -------------------------------
# DOWNLOAD
# -------------------------------
st.download_button(
    "⬇️ Download Data",
    df.to_csv(index=False).encode(),
    "job_analysis.csv"
)