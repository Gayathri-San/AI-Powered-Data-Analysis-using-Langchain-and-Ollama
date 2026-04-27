import pandas as pd
import json
import re

from langchain_community.llms import Ollama   # ✅ CHANGED


def run_query(df, query):

    df = df.copy()  # safety

    # -------- PROMPT --------
    columns = df.columns.tolist()

    prompt = f"""
    Return ONLY JSON.

    Rules:
    - group_by can be single OR multiple columns
    - metric must be column name ONLY
    - aggregation: count, sum, mean, min, max
    - Use ONLY columns from this list: {columns}

    Format:
    {{
      "group_by": "",
      "metric": "",
      "aggregation": "",
      "top_n": null
    }}

    Query: {query}
    """

    # -------- LANGCHAIN OLLAMA --------
    llm = Ollama(
        model="llama3",   # ✅ using llama3:8b
        temperature=0
    )

    raw_output = llm.invoke(prompt)
    print("\nRaw LLM Output:\n", raw_output)

    # -------- EXTRACT JSON --------
    def extract_json(text):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        else:
            raise ValueError("❌ No valid JSON found")

    config = extract_json(raw_output)

    # -------- EXTRACT TOP N --------
    def extract_top_n(query):
        match = re.search(r"top\s*(\d+)", query.lower())
        if match:
            return int(match.group(1))
        return None

    top_n = extract_top_n(query)

    # -------- FIX COUNT --------
    metric = config.get("metric")
    agg = config.get("aggregation")

    if isinstance(metric, str) and "count(" in metric.lower():
        metric = re.findall(r"count\((.*?)\)", metric.lower())[0]
        agg = "count"

    if any(word in query.lower() for word in ["count", "number", "how many"]):
        agg = "count"

    config["metric"] = metric
    config["aggregation"] = agg

    # -------- TIME DETECTION --------
    def detect_time_grain(query):
        q = query.lower()
        if "month" in q:
            return "month"
        elif "year" in q:
            return "year"
        elif "day" in q or "date" in q:
            return "day"
        return None

    time_grain = detect_time_grain(query)

    # detect date column
    date_col = None
    for col in df.columns:
        if "date" in col:
            date_col = col
            break

    # -------- COLUMN MATCH --------
    def match_column(name, df_columns):
        if name is None:
            return None

        name = str(name).lower().replace("_", "").replace(" ", "")

        for col in df_columns:
            col_clean = col.lower().replace("_", "").replace(" ", "")
            if name in col_clean or col_clean in name:
                return col

        return None

    # =========================================================
    # 🔥 FIXED TIME GROUPING (MONTH ORDER CORRECT)
    # =========================================================
    if time_grain and date_col:

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        if time_grain == "month":
            df["month"] = df[date_col].dt.to_period("M")

            result = (
                df.groupby("month")[metric]
                .agg(agg)
                .reset_index()
                .sort_values("month")
            )

            result["month"] = result["month"].astype(str)
            group_cols = ["month"]

        elif time_grain == "year":
            df["year"] = df[date_col].dt.year

            result = (
                df.groupby("year")[metric]
                .agg(agg)
                .reset_index()
                .sort_values("year")
            )

            group_cols = ["year"]

        elif time_grain == "day":
            df["day"] = df[date_col].dt.date

            result = (
                df.groupby("day")[metric]
                .agg(agg)
                .reset_index()
                .sort_values("day")
            )

            group_cols = ["day"]

    else:
        # normal grouping
        group_raw = config.get("group_by")

        if isinstance(group_raw, str) and "," in group_raw:
            group_raw = [g.strip() for g in group_raw.split(",")]
        elif not isinstance(group_raw, list):
            group_raw = [group_raw]

        group_cols = []
        for col in group_raw:
            matched = match_column(col, df.columns)
            if matched:
                group_cols.append(matched)

        if not group_cols:
            raise ValueError(f"❌ No valid group columns found: {group_raw}")

        if agg == "count":
            result = df.groupby(group_cols).size().reset_index(name="count")

        else:
            df[metric] = pd.to_numeric(df[metric], errors="coerce")

            result = (
                df.groupby(group_cols)[metric]
                .agg(agg)
                .reset_index()
            )

    # =========================================================
    # 🔥 FINAL SORT
    # =========================================================
    time_cols = ["month", "year", "day"]

    if any(col in result.columns for col in time_cols):
        for t in time_cols:
            if t in result.columns:
                result = result.sort_values(by=t)
                break
    else:
        value_col = result.columns[-1]
        result = result.sort_values(by=value_col, ascending=False)

    # -------- TOP N --------
    if top_n is not None:
        result = result.head(top_n)

    # -------- DEBUG --------
    print("\n--- DEBUG INFO ---")
    print("Query:", query)
    print("Parsed JSON:", config)
    print("Group Columns:", group_cols)
    print("Metric Column:", metric)

    return result