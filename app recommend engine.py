"""
app.py -- minimal VS Code test harness for Part 4 (Analytics & Recommendations).

This file stands in for Parts 1-3 only so Part 4 can be run on its own.
Replace it with the platform's real upload/quality/validation flow later.
"""

# streamlit renders the page; pandas reads the uploaded file.
import pandas as pd
import streamlit as st

# The two Part 4 entry points.
from analytics import render_analytics
from recommendations_engine import render_recommendations

# Configure the browser tab; must be the first Streamlit call.
st.set_page_config(page_title="Data Quality & Analytics", layout="wide")

# File uploader so the user supplies a real dataset (CSV or Excel).
uploaded = st.sidebar.file_uploader("Upload a dataset", type=["csv", "xlsx"])
# Radio button choosing which Part 4 section to display.
page = st.sidebar.radio("Section", ["Analytics & Charts", "Recommendations"])

if uploaded is None:
    # Prompt shown until a dataset is uploaded.
    st.info("Upload a CSV or Excel file in the sidebar to begin.")
else:
    try:
        # Read the file into a DataFrame based on its extension.
        data = (pd.read_csv(uploaded) if uploaded.name.endswith(".csv")
                else pd.read_excel(uploaded))
    except Exception as error:
        # Show unreadable-file problems instead of crashing.
        st.error(f"The file could not be read: {error}")
    else:
        # Hand the DataFrame to the selected Part 4 module.
        if page == "Analytics & Charts":
            render_analytics(data)
        else:
            render_recommendations(data)
