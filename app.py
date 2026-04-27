import streamlit as st
import pandas as pd

from agent import run_query
from charts import plot_chart
from insight import generate_insights


st.title("AI Data Analyst")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:

    df = pd.read_csv(file)

    st.write("### Data Preview")
    st.dataframe(df.head())

    query = st.text_input("Enter Query")

    analyze = st.button("Analyze")

    if analyze and query:

        # ✅ FINAL DATAFRAME FROM AGENT
        final_df = run_query(df, query)

        st.write("### Result DataFrame")
        st.dataframe(final_df)

        st.write("### Chart")
        plot_chart(final_df, "auto")

        st.write("### Insights")
        st.write(generate_insights(final_df, query))