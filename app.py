import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import os
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_shap import st_shap

from aif360.datasets import StandardDataset
from aif360.metrics import ClassificationMetric

# === SETTINGS ===
MODEL_PATH = "models/best_rf_pipeline.pkl"
THRESHOLD_PATH = "models/best_rf_threshold.txt"
SHAP_VALUES_PATH = "models/rf_shap_values.npy"
X_TEST_PATH = "final_datasets/X_test_final.csv"
Y_TEST_PATH = "final_datasets/y_test.csv"
FAIRNESS_TEST_PATH = "final_datasets/df_test_fairness.csv"
FAIRNESS_METRICS_PATH = "fairness_metrics_summary.csv"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# === LOAD & SYNC DATA ===
@st.cache_resource
def load_and_sync():
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).iloc[:, 0]
    df_fair = pd.read_csv(FAIRNESS_TEST_PATH)

    common_idx = X_test.index.intersection(y_test.index).intersection(df_fair.index)
    X_sync = X_test.loc[common_idx].reset_index(drop=True)
    y_sync = y_test.loc[common_idx].reset_index(drop=True)
    df_sync = df_fair.loc[common_idx].reset_index(drop=True)

    X_sync.to_csv("final_datasets/X_test_final_sync.csv", index=False)
    y_sync.to_csv("final_datasets/y_test_sync.csv", index=False)
    df_sync.to_csv("final_datasets/df_test_fairness_sync.csv", index=False)
    return X_sync, y_sync, df_sync

X_test_final, y_test, df_test_fairness = load_and_sync()

# === LOAD MODEL AND THRESHOLD ===
@st.cache_resource
def load_model():
    rf_pipeline = joblib.load(MODEL_PATH)
    with open(THRESHOLD_PATH) as f:
        best_threshold = float(f.read())
    return rf_pipeline, best_threshold

rf_pipeline, best_threshold = load_model()

# === SHAP values (precomputed or compute on first run) ===
@st.cache_resource
def get_shap_values(_pipeline, _X):
    def get_estimator(pipeline):
        for name in ['model', 'clf', 'estimator', 'final_estimator']:
            if hasattr(pipeline, 'named_steps') and name in pipeline.named_steps:
                return pipeline.named_steps[name]
            if hasattr(pipeline, name):
                return getattr(pipeline, name)
        if hasattr(pipeline, 'predict'):
            return pipeline
        raise ValueError("Cannot find estimator in pipeline.")
    estimator = get_estimator(_pipeline)
    if os.path.exists(SHAP_VALUES_PATH):
        shap_values = np.load(SHAP_VALUES_PATH, allow_pickle=True)
        explainer = shap.TreeExplainer(estimator)
    else:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(_X)
        np.save(SHAP_VALUES_PATH, shap_values)
    return explainer, shap_values

explainer_rf, shap_values_rf = get_shap_values(rf_pipeline, X_test_final)

# === PREDICTIONS (with threshold tuning) ===
y_prob = rf_pipeline.predict_proba(X_test_final)
y_pred_default = rf_pipeline.predict(X_test_final)
y_pred_threshold = y_pred_default.copy()
high_risk_mask = (y_prob[:, 2] > best_threshold)
y_pred_threshold[high_risk_mask] = 2

# === STREAMLIT UI SETUP ===
st.set_page_config(page_title="Credit Risk SHAP & Fairness Dashboard", layout="wide")
st.title("Credit Risk Model - Explainability & Fairness (Random Forest + SHAP + AIF360)")

st.markdown("""
This dashboard allows you to explore model predictions, fairness metrics (Gender, Age, Education), and SHAP explanations for individual clients.
""")

# === SIDEBAR NAVIGATION ===
st.sidebar.header("Navigation")
nav = st.sidebar.radio("Select page:", [
    "Fairness Overview",
    "Client SHAP & Fairness",
    "Global Feature Importance",
    "Feature Distribution",
    "Download Reports & Plots"
])

def show_legend():
    st.markdown("""
    <div style="background:#f9f9f9;padding:8px;border-radius:8px;">
    <b>Protected Attribute Coding (0 / 1):</b>
    <ul>
        <li><b>Gender</b>: <span style="color: #3366cc">0 = Female, 1 = Male</span></li>
        <li><b>Age_Privileged</b>: <span style="color: #3366cc">0 = Age < 42.6, 1 = Age ≥ 42.6</span></li>
        <li><b>Edu_Privileged</b>: <span style="color: #3366cc">0 = Other, 1 = Higher/Academic</span></li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# === FAIRNESS METRICS PAGE ===
if nav == "Fairness Overview":
    st.header("1. Fairness Metrics Overview")
    show_legend()
    st.markdown("**Precomputed Fairness Summary:**")
    if os.path.exists(FAIRNESS_METRICS_PATH):
        fairness_df = pd.read_csv(FAIRNESS_METRICS_PATH)
        st.dataframe(fairness_df, use_container_width=True)
    else:
        st.warning("fairness_metrics_summary.csv not found.")

    st.markdown("**Live Fairness Metrics (recomputed):**")
    prot_attrs = [
        ('Gender', 'Gender', 1),
        ('Age', 'Age_Privileged', 1),
        ('Education', 'Edu_Privileged', 1)
    ]
    metrics = {}
    for attr_name, col, priv_val in prot_attrs:
        if col in df_test_fairness.columns:
            aif_ds = StandardDataset(
                df=df_test_fairness,
                label_name='Risk_Level',
                favorable_classes=[0],
                protected_attribute_names=[col],
                privileged_classes=[[priv_val]]
            )
            aif_ds_pred = aif_ds.copy(deepcopy=True)
            aif_ds_pred.labels = y_pred_threshold.reshape(-1, 1)
            unpriv, priv = [{col: 0}], [{col: 1}]
            metric = ClassificationMetric(aif_ds, aif_ds_pred, unprivileged_groups=unpriv, privileged_groups=priv)
            metrics[attr_name] = {
                'SPD': metric.statistical_parity_difference(),
                'DI': metric.disparate_impact(),
                'EOD': metric.equal_opportunity_difference(),
                'AOD': metric.average_odds_difference()
            }
    fairness_metrics_df = pd.DataFrame(metrics).T
    st.dataframe(fairness_metrics_df, use_container_width=True)

    # === BARPLOT + HEATMAP ===
    st.markdown("**Barplot and Heatmap of Fairness Metrics**")
    fig, ax = plt.subplots(figsize=(8, 4))
    fairness_metrics_df.plot(kind="bar", ax=ax)
    plt.ylabel("Metric Value")
    plt.title("Fairness Metrics by Protected Attribute")
    plt.tight_layout()
    barplot_path = os.path.join(PLOTS_DIR, "fairness_metrics_barplot.png")
    plt.savefig(barplot_path)
    st.pyplot(fig)
    with open(barplot_path, "rb") as f:
        st.download_button("Download barplot", f, "fairness_metrics_barplot.png", "image/png")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.heatmap(fairness_metrics_df, annot=True, cmap="coolwarm", fmt=".3f", ax=ax2)
    plt.title("Fairness Metrics Heatmap")
    plt.tight_layout()
    heatmap_path = os.path.join(PLOTS_DIR, "fairness_metrics_heatmap.png")
    plt.savefig(heatmap_path)
    st.pyplot(fig2)
    with open(heatmap_path, "rb") as f:
        st.download_button("Download heatmap", f, "fairness_metrics_heatmap.png", "image/png")
    plt.close(fig2)

    st.info("Metrics: SPD = Statistical Parity Difference, DI = Disparate Impact, EOD = Equal Opportunity Difference, AOD = Average Odds Difference")

# === INDIVIDUAL CLIENT SHAP & FAIRNESS PAGE ===
if nav == "Client SHAP & Fairness":
    st.header("2. Individual Client Explanation & Fairness")
    show_legend()
    st.markdown(f"""
    Enter a <b>Client Index</b> (0 - {len(X_test_final) - 1}) to view detailed SHAP explanations and model prediction/fairness.
    Index corresponds to row in test set.
    """, unsafe_allow_html=True)
    client_idx = st.number_input("Client Index (row number in test set)", min_value=0, max_value=len(X_test_final) - 1, value=0, step=1)

    # Show raw info
    st.subheader("Client Raw Features")
    st.dataframe(df_test_fairness.iloc[[client_idx]])

    # Show model prediction
    pred = y_pred_threshold[client_idx]
    st.markdown(f"**Model Prediction:** Risk Level = `{pred}` (0=Good, 1=Low Risk, 2=High Risk)")

    # SHAP explanation (robust, with st_shap)
    st.subheader("SHAP Force Plot (Client-Level)")
    try:
        force_plot = shap.force_plot(
            explainer_rf.expected_value[pred],
            shap_values_rf[client_idx, :, pred],
            X_test_final.iloc[client_idx],
            feature_names=X_test_final.columns,
            show=False,
            matplotlib=False
        )
        st_shap(force_plot, height=400)
    except Exception as e:
        st.error(f"Could not generate SHAP force plot: {e}")

    # Show SHAP feature importance for this client
    st.markdown("**Top Features for this Prediction:**")
    try:
        feat_imp = pd.Series(shap_values_rf[client_idx, :, pred], index=X_test_final.columns).abs().sort_values(ascending=False)
        st.dataframe(feat_imp.head(10))
    except Exception as e:
        st.warning("Could not show top features: " + str(e))

    # Fairness for this client (with legend!)
    st.subheader("Protected Attributes for this Client")
    prot_show = df_test_fairness.loc[client_idx, ['Gender', 'Age_Privileged', 'Edu_Privileged']]
    prot_map = {
        'Gender': {1: "Male", 0: "Female"},
        'Age_Privileged': {1: "Age ≥ 42.6", 0: "Age < 42.6"},
        'Edu_Privileged': {1: "Higher/Academic", 0: "Other"}
    }
    prot_legenda = {k: prot_map[k][v] for k, v in prot_show.items()}
    st.table(pd.DataFrame({"Value": prot_show, "Label": prot_legenda}))

    st.markdown("**Interpretation:**")
    st.write("For this client, you can compare the prediction and feature attributions with their protected attribute values. If the prediction changes significantly only when the protected attribute (e.g., Gender) is switched, this can be a sign of individual unfairness.")

# === GLOBAL SHAP PAGE ===
if nav == "Global Feature Importance":
    st.header("3. Global SHAP Feature Importance")
    st.markdown("**Global Bar Plot (SHAP):**")
    try:
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values_rf,
            X_test_final,
            feature_names=X_test_final.columns,
            plot_type="bar",
            show=False
        )
        plt.title("SHAP Feature Importance (Bar Plot)")
        plt.tight_layout()
        barplot_fp = os.path.join(PLOTS_DIR, "shap_rf_feature_importance_bar.png")
        plt.savefig(barplot_fp)
        st.pyplot(plt.gcf())
        plt.close()
    except Exception as e:
        st.warning(f"Could not display global SHAP bar plot: {e}")

    st.markdown("**SHAP Beeswarm (Swarm) Plot:**")
    try:
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
        shap_values_rf,
            X_test_final,
            feature_names=X_test_final.columns,
            plot_type="dot",
            show=False
        )
        plt.title("SHAP Feature Importance (Swarm Plot)", pad=30)
        plt.tight_layout()
        plt.savefig("plots/shap_rf_feature_importance_swarm.png")
        plt.close()
    except Exception as e:
        st.warning(f"Could not display global SHAP swarm plot: {e}")

# === FEATURE DISTRIBUTION PAGE ===
if nav == "Feature Distribution":
    st.header("4. Feature Distribution vs. Risk Level")
    st.markdown("You can quickly visualize the distribution of any feature grouped by predicted Risk Level on the test set.")
    selected_feat = st.selectbox("Select feature for distribution plot:", X_test_final.columns)
    try:
        data = pd.concat([X_test_final[[selected_feat]], pd.Series(y_pred_threshold, name="Risk_Level")], axis=1)
        fig, ax = plt.subplots(figsize=(8, 4))
        if pd.api.types.is_numeric_dtype(X_test_final[selected_feat]):
            sns.boxplot(x="Risk_Level", y=selected_feat, data=data, ax=ax)
            plt.title(f"{selected_feat} distribution by Predicted Risk Level")
        else:
            sns.countplot(x=selected_feat, hue="Risk_Level", data=data, ax=ax)
            plt.title(f"{selected_feat} count by Predicted Risk Level")
        plt.tight_layout()
        dist_fp = os.path.join(PLOTS_DIR, f"dist_{selected_feat}.png")
        plt.savefig(dist_fp)
        st.pyplot(fig)
        plt.close()
        with open(dist_fp, "rb") as f:
            st.download_button("Download Plot", f, f"dist_{selected_feat}.png", "image/png")
    except Exception as e:
        st.warning(f"Could not generate distribution plot: {e}")

# === DOWNLOAD PAGE ===
if nav == "Download Reports & Plots":
    st.header("5. Download Reports & All Plots")
    for fname in os.listdir(PLOTS_DIR):
        if fname.endswith(".png"):
            st.image(os.path.join(PLOTS_DIR, fname), caption=fname)
            with open(os.path.join(PLOTS_DIR, fname), "rb") as f:
                st.download_button(
                    label=f"Download {fname}",
                    data=f,
                    file_name=fname,
                    mime="image/png"
                )
    # Download fairness_metrics_summary.csv
    if os.path.exists(FAIRNESS_METRICS_PATH):
        with open(FAIRNESS_METRICS_PATH, "rb") as f:
            st.download_button("Download Fairness Metrics CSV", f, "fairness_metrics_summary.csv", "text/csv")

    # Download SHAP values
    if os.path.exists(SHAP_VALUES_PATH):
        with open(SHAP_VALUES_PATH, "rb") as f:
            st.download_button("Download SHAP Values NPY", f, "rf_shap_values.npy", "application/octet-stream")

    # Download df_test_fairness
    if os.path.exists(FAIRNESS_TEST_PATH):
        with open(FAIRNESS_TEST_PATH, "rb") as f:
            st.download_button("Download df_test_fairness CSV", f, "df_test_fairness.csv", "text/csv")

st.sidebar.info("For questions, contact Victor Pavel")
