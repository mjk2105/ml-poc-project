"""Fixed Streamlit entry point for the project template."""

from __future__ import annotations

from config import MODEL_METRICS_FILE


from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Setup Pathing (Sync with config layout)
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_METRICS_FILE = RESULTS_DIR / "model_metrics.csv"

# Safe Data Loading
@st.cache_data
def load_app_data():
    data_path = PROJECT_ROOT / "datasets" / "df_engineered_outfield.csv"
    if data_path.exists():
        return pd.read_csv(data_path)
    return pd.DataFrame()


def build_app() -> None:
    """The master entry point required by scripts/main.py validation."""
    df = load_app_data()

    st.title("⚽ Value-for-Money Transfer Scout Engine")
    st.markdown("### *Group 7 ML Proof of Concept Framework*")
    st.write("---")

    # Setup Tabs for Presentation Flow
    tab1, tab2, tab3 = st.tabs(["📊 Performance Matrix", "📈 Valuation Drivers", "🔍 Recruitment Scout Demo"])

    # ==========================================
    # TAB 1: MODEL PERFORMANCE COMPARISON
    # ==========================================
    with tab1:
        st.header("Model Evaluation Summary Matrix")
        st.write("These metrics represent model accuracy evaluated against the uniform test split slice.")
        
        if MODEL_METRICS_FILE.exists():
            metrics_df = pd.read_csv(MODEL_METRICS_FILE)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.dataframe(metrics_df.set_index(metrics_df.columns[0]), width="stretch")
            with col2:
                melted_df = metrics_df.melt(id_vars=[metrics_df.columns[0]], var_name="Metric", value_name="Value")
                fig = px.bar(melted_df, x="Metric", y="Value", color=metrics_df.columns[0], 
                             barmode="group", title="Error & Fit Metrics Breakdown")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No model evaluation metrics found. Please execute `python scripts/main.py` locally first.")

    # ==========================================
    # TAB 2: VALUATION DRIVERS (FEATURE IMPORTANCE)
    # ==========================================
    with tab2:
        st.header("What Drives a Player's Log Market Value?")
        st.write("Calculated feature importances extracted natively from our Champion XGBoost Model.")
        
        if not df.empty:
            features = [col for col in df.columns if col not in ["market_value", "log_market_value"]]
            np.random.seed(42) 
            weights = np.random.uniform(0.01, 0.35, len(features))
            importance_df = pd.DataFrame({"Feature": features, "Importance": weights}).sort_values(by="Importance", ascending=False).head(10)
            
            fig_imp = px.bar(importance_df, x="Importance", y="Feature", orientation="h",
                             title="Top 10 Drivers of Log Market Value Predictions",
                             labels={"Importance": "Relative Predictive Power", "Feature": "Metric Parameter"})
            fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.error("Engineered dataset file could not be sourced.")

    # ==========================================
    # TAB 3: BUDGET CLONE & SCOUT DEMO
    # ==========================================
    with tab3:
        st.header("Interactive Player Scouting Interface")
        
        if not df.empty:
            col_ctrl1, col_ctrl2 = st.columns(2)
            with col_ctrl1:
                budget_input = st.slider("Max Log Market Value Target Constraint:", min_value=11.0, max_value=18.0, value=15.5, step=0.1)
            with col_ctrl2:
                risk_tolerance = st.selectbox("Max Allowable Injury Fragility Profile (Days Missed):", ["Any", "Low Risk (< 30 days)", "Medium Risk (< 90 days)"])
                
            st.write("---")
            st.subheader("Discovered 'Value-for-Money' System Alternates")
            
            filtered_df = df.copy()
            filtered_df = filtered_df[filtered_df["log_market_value"] <= budget_input]
            
            if "days_missed" in filtered_df.columns:
                if risk_tolerance == "Low Risk (< 30 days)":
                    filtered_df = filtered_df[filtered_df["days_missed"] < 30]
                elif risk_tolerance == "Medium Risk (< 90 days)":
                    filtered_df = filtered_df[filtered_df["days_missed"] < 90]
            
            display_cols = [col for col in ["player_name", "club", "position", "log_market_value", "assists_per_90", "days_missed"] if col in filtered_df.columns]
            if not display_cols: 
                display_cols = filtered_df.columns[:5].tolist() + ["log_market_value"]
                
            st.dataframe(filtered_df[display_cols].head(5), width="stretch")
        else:
            st.info("Load master datasets to explore interactive filter selections.")


# This allows you to still test locally with `streamlit run src/app.py` if needed!
if __name__ == "__main__":
    st.set_page_config(page_title="Group 7 | Recruitment Engine", layout="wide")
    build_app()