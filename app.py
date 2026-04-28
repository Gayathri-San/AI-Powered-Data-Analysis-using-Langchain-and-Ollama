import streamlit as st
import pandas as pd

from agent import run_query
from charts import plot_chart
from insight import generate_insights


st.title("AI Data Analyst")

file = st.file_uploader("Upload CSV", type=["csv"])

# ✅ Initialize session state
if "final_df" not in st.session_state:
    st.session_state.final_df = None

if "query" not in st.session_state:
    st.session_state.query = ""

if file:
    df = pd.read_csv(file)

    st.write("### Data Preview")
    st.dataframe(df.head())

    query = st.text_input("Enter Query")

    analyze = st.button("Analyze")

    # ✅ Run query ONLY when button clicked
    if analyze and query:
        st.session_state.final_df = run_query(df, query)
        st.session_state.query = query

    # ✅ If result exists, show everything
    if st.session_state.final_df is not None:

        st.write("### Result DataFrame")
        st.dataframe(st.session_state.final_df)

        # ✅ Chart dropdown (this will NOT rerun query)
        chart_type = st.selectbox(
            "Select Chart Type",
            ["auto", "bar", "line", "pie", "histogram", "clustered"]
        )

        st.write("### Chart")
        plot_chart(st.session_state.final_df, chart_type)

        st.write("### Insights")
        st.write(generate_insights(st.session_state.final_df, st.session_state.query))
