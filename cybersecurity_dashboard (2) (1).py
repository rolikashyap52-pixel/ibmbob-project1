import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="CyberShield Analytics", page_icon="🛡️", layout="wide")
st.title("🛡️ CyberShield Analytics")
st.caption("Cybersecurity Threat Analytics & Risk Prediction Dashboard")

uploaded = st.sidebar.file_uploader("Upload cybersecurity CSV", type=["csv"])
default_files = list(Path(".").glob("*.csv"))

if uploaded is not None:
    df = pd.read_csv(uploaded)
elif default_files:
    df = pd.read_csv(default_files[0])
else:
    st.info("Upload your cybersecurity CSV dataset from the sidebar.")
    st.stop()

df.columns = [str(c).strip() for c in df.columns]
df = df.replace([np.inf, -np.inf], np.nan)

st.sidebar.success(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
st.sidebar.write("Columns:", list(df.columns))

# Try to identify useful columns automatically
def find_col(words):
    for c in df.columns:
        cl = c.lower().replace("_", " ").replace("-", " ")
        if any(w in cl for w in words):
            return c
    return None

attack_col = find_col(["attack type", "attack_type", "subcategory", "category", "label", "attack"])
target_col = find_col(["attack detected", "attack_detected", "target", "label", "attack"])
severity_col = find_col(["severity", "risk level", "risk_level"])
time_col = find_col(["timestamp", "datetime", "date", "time"])

tab1, tab2, tab3, tab4 = st.tabs(["Executive Overview", "Threat Analysis", "Risk Analysis", "ML Prediction"])

with tab1:
    st.subheader("Executive Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Features", f"{len(df.columns):,}")
    if target_col:
        attack_rate = pd.to_numeric(df[target_col], errors="coerce").mean() if pd.api.types.is_numeric_dtype(df[target_col]) else np.nan
        c3.metric("Target Column", target_col)
        c4.metric("Missing Cells", f"{int(df.isna().sum().sum()):,}")
    else:
        c3.metric("Missing Cells", f"{int(df.isna().sum().sum()):,}")
        c4.metric("Duplicate Rows", f"{int(df.duplicated().sum()):,}")

    st.write("### Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    if attack_col and df[attack_col].nunique(dropna=True) <= 50:
        counts = df[attack_col].astype(str).value_counts().reset_index()
        counts.columns = [attack_col, "Count"]
        fig = px.bar(counts.head(20), x=attack_col, y="Count", title="Attack / Class Distribution")
        st.plotly_chart(fig, use_container_width=True)

    if time_col:
        t = pd.to_datetime(df[time_col], errors="coerce")
        if t.notna().sum() > 0:
            trend = t.dt.date.value_counts().sort_index().reset_index()
            trend.columns = ["Date", "Records"]
            fig = px.line(trend, x="Date", y="Records", title="Traffic / Incident Trend")
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Threat & Attack Analysis")
    if attack_col:
        vc = df[attack_col].astype(str).value_counts().reset_index()
        vc.columns = ["Attack Type", "Count"]
        fig = px.pie(vc.head(15), names="Attack Type", values="Count", title="Attack Type Distribution")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(vc.head(20), use_container_width=True)
    else:
        st.warning("No obvious attack/class column was detected. Select/rename your target column to 'attack_type' or 'attack_detected'.")

    numeric = df.select_dtypes(include=np.number)
    if not numeric.empty:
        corr = numeric.corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=False, title="Numeric Feature Correlation")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Risk Analysis")
    if severity_col:
        sev = df[severity_col].astype(str).value_counts().reset_index()
        sev.columns = ["Severity", "Count"]
        fig = px.bar(sev, x="Severity", y="Count", title="Severity Distribution")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No severity column detected. The dashboard will use available attack/class features instead.")

    st.write("### Data Quality / Risk Signals")
    quality = pd.DataFrame({
        "Metric": ["Missing cells", "Duplicate rows", "Columns", "Rows"],
        "Value": [int(df.isna().sum().sum()), int(df.duplicated().sum()), len(df.columns), len(df)]
    })
    st.dataframe(quality, use_container_width=True)

    st.write("### Recommended Actions")
    st.markdown("""
    - Prioritize critical/high-risk incidents when severity information is available.
    - Investigate attack categories with unusually high frequency.
    - Monitor features strongly associated with the target or attack class.
    - Retrain the detection model periodically because cyber-threat patterns can change.
    - Review false positives and false negatives before deploying an automated alerting system.
    """)

with tab4:
    st.subheader("Machine Learning Prediction")
    if not target_col:
        st.warning("No target column was detected. Rename the target to 'attack_detected' (0/1) or 'attack_type' and reload.")
    elif df[target_col].nunique(dropna=True) < 2:
        st.warning("The target must contain at least two classes.")
    else:
        data = df.dropna(subset=[target_col]).copy()
        y = data[target_col]
        # Remove obvious identifiers and very high-cardinality text columns
        X = data.drop(columns=[target_col])
        drop_cols = [c for c in X.columns if X[c].nunique(dropna=True) > min(500, max(50, len(X)*0.5))]
        X = X.drop(columns=drop_cols, errors="ignore")

        if len(X) < 50:
            st.warning("Not enough usable records for a reliable demonstration model.")
        else:
            cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
            num_cols = X.select_dtypes(include=np.number).columns.tolist()
            pre = ColumnTransformer([
                ("num", SimpleImputer(strategy="median"), num_cols),
                ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                  ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
            ])
            model = Pipeline([("preprocess", pre),
                              ("model", RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1, class_weight="balanced"))])
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if y.value_counts().min() >= 2 else None
            )
            with st.spinner("Training Random Forest model..."):
                model.fit(X_train, y_train)
                pred = model.predict(X_test)

            a = accuracy_score(y_test, pred)
            p = precision_score(y_test, pred, average="weighted", zero_division=0)
            r = recall_score(y_test, pred, average="weighted", zero_division=0)
            f = f1_score(y_test, pred, average="weighted", zero_division=0)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{a:.2%}")
            m2.metric("Precision", f"{p:.2%}")
            m3.metric("Recall", f"{r:.2%}")
            m4.metric("F1 Score", f"{f:.2%}")

            cm = confusion_matrix(y_test, pred)
            fig = px.imshow(cm, text_auto=True, title="Confusion Matrix")
            st.plotly_chart(fig, use_container_width=True)

            st.caption("Metrics are based on the uploaded dataset and should be interpreted with class balance, leakage, and deployment context in mind.")

st.divider()
st.caption("CyberShield Analytics | B.Tech CSE Cybersecurity + Data Analytics Project")
