"""
app.py — Streamlit Demo
-----------------------
Multi-page income risk scoring demo built on the responsible-mlops-risk-engine pipeline.

Pages:
    1. Overview       — project summary, dataset, model progression
    2. Prediction     — live inference via SageMaker endpoint
    3. Fairness Audit — per-group PPR and AUC from evaluate.py
    4. Model Metrics  — full performance breakdown and feature importance

Run:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib
import boto3
import json
from glob import glob
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Income Risk Engine",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #1e2130;
}
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

/* Main background */
.main { background: #0a0c14; }
.block-container { padding: 2rem 3rem; }

/* Headers */
h1 { font-family: 'IBM Plex Mono', monospace !important; color: #7ee8a2 !important; letter-spacing: -0.5px; }
h2 { font-family: 'IBM Plex Mono', monospace !important; color: #a8b4c8 !important; font-size: 1.1rem !important; text-transform: uppercase; letter-spacing: 2px; }
h3 { color: #e2e8f0 !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #141824;
    border: 1px solid #1e2a3a;
    border-radius: 8px;
    padding: 1rem;
}
[data-testid="metric-container"] label { color: #64748b !important; font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 1px; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #7ee8a2 !important; font-family: 'IBM Plex Mono', monospace; font-size: 1.8rem !important; }

/* Buttons */
.stButton > button {
    background: #7ee8a2;
    color: #0a0c14;
    border: none;
    border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    padding: 0.6rem 2rem;
    letter-spacing: 0.5px;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: #5fd48a;
    transform: translateY(-1px);
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border: 1px solid #1e2a3a;
    border-radius: 8px;
}

/* Input widgets */
.stSelectbox > div, .stSlider > div {
    color: #e2e8f0;
}

/* Info/success/warning boxes */
.stAlert { border-radius: 8px; }

/* Divider */
hr { border-color: #1e2a3a; }

/* Tag badges */
.tag {
    display: inline-block;
    background: #1a2332;
    border: 1px solid #2d4a6e;
    color: #7eb8f7;
    padding: 2px 10px;
    border-radius: 20px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    margin: 2px;
}

/* Prediction result */
.pred-high {
    background: linear-gradient(135deg, #0d2818, #0a1f12);
    border: 1px solid #7ee8a2;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
}
.pred-low {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #4a5568;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
}
.pred-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 3rem;
    font-weight: 600;
    color: #7ee8a2;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
ENDPOINT_NAME = "responsible-risk-engine-prod-v1"
AWS_REGION = "us-east-1"
MODELS_DIR = "models"
DATA_DIR = "data/processed"

# Label maps for encoded categoricals
EDUCATION_LABELS = {
    1: "No schooling", 2: "Nursery school", 3: "Grade 1", 4: "Grade 2",
    5: "Grade 3", 6: "Grade 4", 7: "Grade 5", 8: "Grade 6",
    9: "Grade 7", 10: "Grade 8", 11: "Grade 9", 12: "Grade 10",
    13: "Grade 11", 14: "Grade 12 (no diploma)", 15: "High school diploma",
    16: "GED", 17: "Some college (<1yr)", 18: "Some college (>1yr)",
    19: "Associate degree", 20: "Bachelor's degree",
    21: "Master's degree", 22: "Professional degree", 23: "Doctorate",
    24: "No schooling (older code)",
}
OCCUPATION_SAMPLE = {
    10: "Management", 100: "Business Operations", 200: "Finance",
    300: "Computer & Math", 400: "Architecture & Engineering",
    500: "Science", 600: "Social Services", 700: "Legal",
    800: "Education", 900: "Arts & Entertainment",
    1000: "Healthcare (Practitioner)", 1100: "Healthcare (Support)",
    1200: "Protective Services", 1300: "Food Prep",
    1400: "Building & Grounds", 1500: "Personal Care",
    1600: "Sales", 1700: "Office & Admin",
    1800: "Farming", 1900: "Construction",
    2000: "Extraction", 2100: "Installation & Repair",
    2200: "Production", 2300: "Transportation",
    2400: "Material Moving",
}
COW_LABELS = {
    1: "Private for-profit (employee)", 2: "Private non-profit (employee)",
    3: "Local government", 4: "State government",
    5: "Federal government", 6: "Self-employed (not inc.)",
    7: "Self-employed (inc.)", 8: "Family business (unpaid)",
    9: "Unemployed",
}
MAR_LABELS = {
    1: "Married", 2: "Widowed", 3: "Divorced",
    4: "Separated", 5: "Never married",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource
def load_pipeline():
    """Load full sklearn Pipeline — preprocessor + model in one artifact."""
    import sys
    sys.path.insert(0, ".")
    from src.data.preprocess import ACSPreprocessor  # noqa: F401 — needed for joblib
    files = sorted(glob(f"{MODELS_DIR}/full_pipeline_*.joblib"))
    if files:
        return joblib.load(files[-1])
    return None


def predict_sagemaker(csv_input: str) -> float:
    """Run inference via live SageMaker endpoint — accepts preprocessed CSV string."""
    try:
        runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="text/csv",
            Body=csv_input,
        )
        result = response["Body"].read().decode("utf-8").strip()
        return float(result.split("\n")[0])
    except Exception as e:
        st.warning(f"Endpoint unavailable: {e}. Using local model.")
        return None


def predict_local(raw_input: dict) -> float:
    """Local inference using full pipeline — no preprocessing needed."""
    pipeline = load_pipeline()
    if pipeline:
        import pandas as pd
        df = pd.DataFrame([raw_input])
        return float(pipeline.predict_proba(df)[0][1])
    return None


def build_raw_input(age, education_code, occupation_code,
                    hours_per_week, cow_code, mar_code) -> dict:
    """
    Build raw input dict for the full pipeline.
    The pipeline handles all preprocessing — no encoding needed here.
    """
    return {
        "age": age,
        "education": str(education_code),
        "occupation": str(occupation_code),
        "hours_per_week": hours_per_week,
        "class_of_worker": str(cow_code),
        "marital_status": str(mar_code),
    }


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ Income Risk Engine")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Overview", "Prediction", "Fairness Audit", "Model Metrics"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
    <div style='font-family: IBM Plex Mono; font-size: 0.7rem; color: #4a5568; line-height: 1.8;'>
    Dataset: ACS PUMS 2023<br>
    State: Virginia (FIPS 51)<br>
    Records: 88,928<br>
    Threshold: $75,000<br>
    Model: XGBoost v2<br>
    AUC-ROC: 0.9506<br>
    Fairness: PASSED
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("responsible-mlops-risk-engine")
    st.markdown("""
    <p style='color: #64748b; font-size: 1.1rem; margin-bottom: 2rem;'>
    Production-grade income risk scoring on 2023 U.S. Census Bureau data.
    NIST AI RMF 1.0 aligned. Demographic fairness audits enforced in CI/CD.
    </p>
    """, unsafe_allow_html=True)

    # Tag badges
    tags = ["ACS PUMS 2023", "XGBoost", "Optuna", "MLflow",
            "Evidently AI", "SageMaker", "Terraform", "NIST AI RMF"]
    st.markdown(" ".join([f'<span class="tag">{t}</span>' for t in tags]),
                unsafe_allow_html=True)

    st.markdown("---")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AUC-ROC", "0.9506", "+0.0398 vs baseline")
    col2.metric("F1 Score", "0.7633", "+0.1125 vs baseline")
    col3.metric("Training Records", "57,628", "Virginia 2023")
    col4.metric("Fairness Gate", "PASSED", "10/10 groups")

    st.markdown("---")

    # Model progression
    st.markdown("## Model Progression")
    progression = pd.DataFrame({
        "Model": ["Logistic Regression", "Ridge (L2)", "XGBoost + Optuna"],
        "AUC-ROC": [0.9108, 0.9108, 0.9506],
        "F1": [0.6508, 0.6507, 0.7633],
        "Notes": [
            "Strong interpretable baseline",
            "CV selected C=100 — no regularization benefit",
            "Non-linear signal in occupation confirmed",
        ]
    })
    st.dataframe(progression, width="stretch", hide_index=True)

    st.markdown("---")

    # Pipeline stages
    st.markdown("## Pipeline")
    stages = {
        "Data": ["ingest.py — ACS Census API pull",
                 "preprocess.py — encoding, scaling, split"],
        "Training": ["baseline.py → ridge.py → train_xgboost.py",
                     "evaluate.py — metrics + fairness gate",
                     "register.py — MLflow model registry"],
        "Infrastructure": ["infrastructure/main.tf — S3, IAM, CloudWatch",
                           "Terraform applied — all resources provisioned"],
        "Serving": ["deploy.py — SageMaker real-time endpoint",
                    "Native XGBoost JSON format — no custom script"],
        "Monitoring": ["drift_monitor.py — Evidently AI + CloudWatch",
                       "0/6 features drifted on Virginia baseline"],
    }
    cols = st.columns(len(stages))
    for col, (stage, items) in zip(cols, stages.items()):
        with col:
            st.markdown(f"**{stage}**")
            for item in items:
                st.markdown(
                    f"<div style='font-size:0.8rem; color:#64748b; "
                    f"margin-bottom:4px;'>✓ {item}</div>",
                    unsafe_allow_html=True
                )

    st.markdown("---")
    st.markdown("""
    <div style='color: #4a5568; font-size: 0.85rem;'>
    Dataset: American Community Survey PUMS 2023 — U.S. Census Bureau |
    Income threshold: $75,000 (2023 US median household income) |
    Development scope: Virginia (FIPS 51) |
    Production path: STATE_CODE="*" — one config change
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Prediction":
    st.title("Income Risk Prediction")
    st.markdown("""
    <p style='color: #64748b;'>
    Live inference via SageMaker endpoint. Enter individual characteristics
    to get a probability score for annual income ≥ $75,000.
    </p>
    """, unsafe_allow_html=True)

    st.info(
        "⚡ Predictions served by SageMaker endpoint "
        "`responsible-risk-engine-prod-v1` (us-east-1). "
        "Falls back to local model if endpoint is offline.",
        icon="🔌"
    )

    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Individual Characteristics")
        age = st.slider("Age", min_value=18, max_value=95, value=35)
        hours_per_week = st.slider(
            "Hours worked per week", min_value=1, max_value=99, value=40
        )
        education_code = st.selectbox(
            "Education level",
            options=list(EDUCATION_LABELS.keys()),
            format_func=lambda x: EDUCATION_LABELS[x],
            index=list(EDUCATION_LABELS.keys()).index(20),
        )
        occupation_code = st.selectbox(
            "Occupation",
            options=list(OCCUPATION_SAMPLE.keys()),
            format_func=lambda x: OCCUPATION_SAMPLE[x],
            index=3,
        )
        cow_code = st.selectbox(
            "Class of worker",
            options=list(COW_LABELS.keys()),
            format_func=lambda x: COW_LABELS[x],
            index=0,
        )
        mar_code = st.selectbox(
            "Marital status",
            options=list(MAR_LABELS.keys()),
            format_func=lambda x: MAR_LABELS[x],
            index=0,
        )

    with col2:
        st.markdown("### Prediction")
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Run Prediction", width="stretch"):
            with st.spinner("Calling SageMaker endpoint..."):
                raw_input = build_raw_input(
                    age, education_code, occupation_code,
                    hours_per_week, cow_code, mar_code
                )

                # SageMaker endpoint receives preprocessed CSV
                # built from the raw input via the full pipeline
                pipeline = load_pipeline()
                if pipeline:
                    processed = pipeline.named_steps["preprocessor"].transform(
                        pd.DataFrame([raw_input])
                    )
                    csv_input = ",".join(map(str, processed[0]))
                    prob = predict_sagemaker(csv_input)
                else:
                    prob = None

                if prob is None:
                    prob = predict_local(raw_input)
                    source = "local pipeline"
                else:
                    source = "SageMaker endpoint"

            if prob is not None:
                high_income = prob >= 0.5
                css_class = "pred-high" if high_income else "pred-low"
                label = "High Income" if high_income else "Below Threshold"
                color = "#7ee8a2" if high_income else "#94a3b8"

                st.markdown(f"""
                <div class="{css_class}">
                    <div class="pred-value">{prob:.1%}</div>
                    <div style='color: {color}; font-size: 1.2rem;
                         font-family: IBM Plex Mono; margin-top: 0.5rem;'>
                        {label}
                    </div>
                    <div style='color: #4a5568; font-size: 0.75rem;
                         margin-top: 1rem; font-family: IBM Plex Mono;'>
                        via {source}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Feature summary
                st.markdown("**Input summary:**")
                summary = pd.DataFrame({
                    "Feature": ["age", "education", "occupation",
                                "hours_per_week", "class_of_worker",
                                "marital_status"],
                    "Value": [
                        age,
                        EDUCATION_LABELS[education_code],
                        OCCUPATION_SAMPLE[occupation_code],
                        hours_per_week,
                        COW_LABELS[cow_code],
                        MAR_LABELS[mar_code],
                    ]
                })
                st.dataframe(summary, width="stretch",
                             hide_index=True)
            else:
                st.error("Prediction failed — model artifact not found locally.")

        else:
            st.markdown("""
            <div style='background: #141824; border: 1px dashed #1e2a3a;
                 border-radius: 12px; padding: 3rem; text-align: center;
                 color: #4a5568; font-family: IBM Plex Mono;'>
                Configure inputs and click<br>Run Prediction
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='color: #4a5568; font-size: 0.8rem;'>
    Note: Sensitive features (race, sex, nativity) are not inputs to this model.
    They are used exclusively for post-prediction fairness auditing.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — FAIRNESS AUDIT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Fairness Audit":
    st.title("Demographic Fairness Audit")
    st.markdown("""
    <p style='color: #64748b;'>
    Post-prediction fairness analysis across race, sex, and nativity groups.
    Sensitive features are never model inputs — audited after prediction only.
    Virginia (FIPS 51) development data. Overall PPR: 0.2686.
    </p>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Fairness Gate", "PASSED")
    col2.metric("Groups Audited", "10")
    col3.metric("Threshold", "±0.20 PPR")

    st.markdown("---")

    # Race results
    st.markdown("## Race")
    race_data = pd.DataFrame({
        "Group": ["White alone", "Black or African American alone",
                  "American Indian alone", "Asian alone",
                  "Some other race alone", "Two or more races"],
        "N": [9685, 2086, 68, 1059, 448, 1044],
        "PPR": [0.279, 0.171, 0.353, 0.413, 0.167, 0.263],
        "Delta": [0.010, 0.098, 0.084, 0.144, 0.101, 0.005],
        "AUC-ROC": [0.954, 0.920, 0.909, 0.952, 0.941, 0.948],
        "Gate": ["✅ Pass", "✅ Pass", "✅ Pass",
                 "✅ Pass", "✅ Pass", "✅ Pass"],
    })
    st.dataframe(race_data, width="stretch", hide_index=True)

    st.markdown("""
    <div style='background: #141824; border-left: 3px solid #f59e0b;
         padding: 1rem; border-radius: 4px; margin: 1rem 0;
         font-size: 0.85rem; color: #94a3b8;'>
    ⚠️ <strong style='color: #f59e0b;'>Notable:</strong>
    Largest inter-group gap — Black or African American (PPR 0.171) vs
    Asian (PPR 0.413), absolute difference 0.242. Both pass the gate
    measured against overall PPR. Reflects income distribution in Virginia
    2023 ACS data. American Indian group (n=68) flagged — metrics unreliable
    at this sample size.
    </div>
    """, unsafe_allow_html=True)

    # Sex results
    st.markdown("## Sex")
    sex_data = pd.DataFrame({
        "Group": ["Male", "Female"],
        "N": [6929, 7478],
        "PPR": [0.321, 0.220],
        "Delta": [0.053, 0.049],
        "AUC-ROC": [0.946, 0.954],
        "Gate": ["✅ Pass", "✅ Pass"],
    })
    st.dataframe(sex_data, width="stretch", hide_index=True)

    # Nativity results
    st.markdown("## Nativity")
    nativity_data = pd.DataFrame({
        "Group": ["Native born", "Foreign born"],
        "N": [12445, 1962],
        "PPR": [0.260, 0.324],
        "Delta": [0.009, 0.055],
        "AUC-ROC": [0.951, 0.949],
        "Gate": ["✅ Pass", "✅ Pass"],
    })
    st.dataframe(nativity_data, width="stretch", hide_index=True)

    st.markdown("---")
    st.markdown("""
    <div style='color: #4a5568; font-size: 0.85rem;'>
    Fairness gate enforced in CI/CD — evaluate.py exits with code 1 if any
    group exceeds ±0.20 PPR threshold, blocking all downstream deployment steps.
    Full findings and risk response commitments in docs/fairness_report.md.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL METRICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Metrics":
    st.title("Model Performance")
    st.markdown("""
    <p style='color: #64748b;'>
    XGBoost production model — evaluated on held-out test set (14,407 records).
    Trained on Virginia ACS PUMS 2023 data.
    </p>
    """, unsafe_allow_html=True)

    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AUC-ROC", "0.9506")
    col2.metric("F1 Score", "0.7633")
    col3.metric("Precision", "0.6896")
    col4.metric("Recall", "0.8546")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # Feature importance
        st.markdown("## Feature Importance")
        importance = pd.DataFrame({
            "Feature": ["hours_per_week", "occupation", "education",
                        "class_of_worker", "age", "marital_status"],
            "Importance": [0.4397, 0.1719, 0.1403, 0.1132, 0.0713, 0.0636],
        }).sort_values("Importance", ascending=True)

        # Simple bar chart using streamlit
        st.bar_chart(
            importance.set_index("Feature")["Importance"],
            width="stretch",
            color="#7ee8a2",
        )
        st.markdown("""
        <div style='font-size: 0.8rem; color: #4a5568;'>
        occupation and class_of_worker had near-zero linear signal in
        Logistic Regression but rank 2nd and 4th in XGBoost — confirming
        non-linear interactions with education and hours worked.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Model progression
        st.markdown("## Model Progression")
        progression = pd.DataFrame({
            "Model": ["Logistic Regression", "Ridge (L2)", "XGBoost"],
            "AUC-ROC": [0.9108, 0.9108, 0.9506],
            "F1": [0.6508, 0.6507, 0.7633],
            "Delta AUC": ["—", "+0.0000", "+0.0398"],
            "Delta F1": ["—", "-0.0001", "+0.1125"],
        })
        st.dataframe(progression, width="stretch", hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Best params
        st.markdown("## XGBoost Best Parameters")
        params = pd.DataFrame({
            "Parameter": ["n_estimators", "max_depth", "learning_rate",
                          "scale_pos_weight", "gamma", "subsample",
                          "colsample_bytree"],
            "Value": [403, 5, 0.0432, 2.43, 0.438, 0.691, 0.792],
        })
        st.dataframe(params, width="stretch", hide_index=True)
        st.markdown("""
        <div style='font-size: 0.8rem; color: #4a5568;'>
        Tuned via Optuna — 30 trials, 5-fold stratified CV.
        scale_pos_weight 2.43 handles class imbalance (21.7% positive rate).
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # MLflow registry info
    st.markdown("## MLflow Registry")
    col1, col2, col3 = st.columns(3)
    col1.metric("Model Name", "income-risk-xgboost")
    col2.metric("Version", "v2")
    col3.metric("Alias", "staging")

    st.markdown("""
    <div style='font-size: 0.8rem; color: #4a5568; margin-top: 0.5rem;'>
    Start MLflow UI: <code style='background: #141824; padding: 2px 6px;
    border-radius: 4px; color: #7ee8a2;'>mlflow ui</code> →
    http://localhost:5000
    </div>
    """, unsafe_allow_html=True)
