"""Fixed Streamlit entry point for the project template."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config import MODEL_METRICS_FILE


def build_app() -> None:
    """Render the project Streamlit application.

    Students should replace the placeholder sections with their own visualizations,
    explanations, and prediction workflow. The function name and file location are
    fixed because ``scripts/main.py`` launches Streamlit with this module.
    """

    st.set_page_config(page_title="Group 7 - Transfer Scout", layout="wide")

    st.title("🔍 The 'Value-for-Money' Transfer Scout")
    st.write(
        "Welcome to the Group 7 ML Proof of Concept. This application serves as a "
        "predictive scouting engine designed to identify undervalued football talents "
        "and suggest optimal budget alternatives using advanced machine learning."
    )

    st.subheader("Core Prototype Features")
    st.markdown(
        """
        - **Budget Clone Finder:** Input a high-profile player to discover 3 budget-friendly alternatives with matching profiles.
        - **Scout Search:** Filter by target position and club budget constraints to surface undervalued market gems.
        - **Player Risk Profiles:** Evaluate underlying athletic performance metrics versus physical fragility and injury risk trends.
        """
    )

    st.subheader("📊 Latest Model Evaluation Results")
    if MODEL_METRICS_FILE.exists():
        metrics_df = pd.read_csv(MODEL_METRICS_FILE)
        st.dataframe(metrics_df, use_container_width=True)


if __name__ == "__main__":
    build_app()
