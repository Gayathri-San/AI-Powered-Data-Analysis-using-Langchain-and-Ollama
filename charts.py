import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

def plot_chart(df, chart_type="auto"):
    if df is None or df.empty:
        print("No data to plot")
        return

    cols = df.columns.tolist()

    # =========================================================
    # 🔥 AUTO DETECT CHART TYPE
    # =========================================================


    if chart_type == "auto":
        if len(cols) == 3:
            chart_type = "clustered"
        elif len(cols) == 2:
            if any(k in cols[0].lower() for k in ["date", "time", "month", "year", "day"]):
                chart_type = "line"
            else:
                 chart_type = "bar"
        elif len(cols) == 1:
            chart_type = "histogram"
        else:
            chart_type = "bar"  # fallback


    # =========================================================
    # 📈 TIME SERIES DETECTION (SMART FIXED VERSION)
    # =========================================================
    time_keywords = ["date", "time", "month", "year", "day"]

    time_col = next(
        (c for c in cols if any(k in c.lower() for k in time_keywords)),
        None
    )

    if time_col and len(cols) >= 2 and chart_type in ["auto", "line"]:
        y_col = [c for c in cols if c != time_col][0]

        plt.figure()
        plt.plot(df[time_col], df[y_col], marker="o")

        plt.title(f"{y_col} over {time_col}")
        plt.xlabel(time_col)
        plt.ylabel(y_col)

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(plt)
        return

    # =========================================================
    # 📊 BAR CHART
    # =========================================================
    if chart_type == "bar":
        x, y = cols[0], cols[1]

        plt.figure()
        plt.bar(df[x], df[y])

        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(f"{y} by {x}")

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(plt)
        return

    # =========================================================
    # 📈 LINE CHART
    # =========================================================
    if chart_type == "line":
        x, y = cols[0], cols[1]

        plt.figure()
        plt.plot(df[x], df[y], marker='o')

        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(f"{y} by {x}")

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(plt)
        return

    # =========================================================
    # 🥧 PIE CHART
    # =========================================================
    if chart_type == "pie":
        x, y = cols[0], cols[1]

        plt.figure()

        try:
            plt.pie(df[y], labels=df[x], autopct='%1.1f%%')
            plt.title(f"{y} distribution")
            st.pyplot(plt)
        except:
            print("Pie chart requires numeric values")
        return

    # =========================================================
    # 🔥 CLUSTERED BAR CHART
    # =========================================================
    if chart_type == "clustered":
        if len(cols) < 3:
            print("Clustered chart needs 3 columns")
            return

        x_col, sub_col, y_col = cols[0], cols[1], cols[2]

        try:
            pivot_df = df.pivot(index=x_col, columns=sub_col, values=y_col).fillna(0)
        except Exception as e:
            print("Pivot error:", e)
            return

        x = np.arange(len(pivot_df.index))
        width = 0.8 / len(pivot_df.columns)

        plt.figure()

        for i, col in enumerate(pivot_df.columns):
            plt.bar(x + i * width, pivot_df[col], width, label=str(col))

        plt.xticks(
            x + width * (len(pivot_df.columns) / 2),
            pivot_df.index,
            rotation=45,
            ha='right'
        )

        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"{y_col} by {x_col} and {sub_col}")
        plt.legend()

        plt.tight_layout()
        st.pyplot(plt)
        return

    # =========================================================
    # 📊 HISTOGRAM
    # =========================================================
    if chart_type == "histogram":
        numeric_df = df.select_dtypes(include=['number'])

        if numeric_df.empty:
            print("No numeric column found")
            return

        y = numeric_df.columns[0]

        plt.figure()
        plt.hist(df[y])

        plt.xlabel(y)
        plt.title(f"Distribution of {y}")

        plt.tight_layout()
        st.pyplot(plt)
        return

    # =========================================================
    # ❌ FALLBACK
    # =========================================================
    print("Unsupported chart type or dataframe structure")