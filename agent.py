import pandas as pd
import json
import re
from langchain_community.llms import Ollama


def run_query(df, query):

    df = df.copy()

    # -------- PROMPT --------
    columns = df.columns.tolist()

    prompt = f"""
    Return ONLY JSON.

    Rules:
    - group_by can be single OR multiple columns
    - metric must be column name ONLY (never 'count')
    - aggregation: count, sum, mean, min, max
    - Avoid ID-like columns for grouping
    - Prefer categorical columns
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

    llm = Ollama(model="llama3", temperature=0)

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

    # -------- TOP N --------
    def extract_top_n(query):
        match = re.search(r"top\s*(\d+)", query.lower())
        return int(match.group(1)) if match else None

    top_n = extract_top_n(query)

    # -------- FIX COUNT --------
    metric = config.get("metric")
    agg = config.get("aggregation")

    if isinstance(metric, str) and "count(" in metric.lower():
        metric = re.findall(r"count\((.*?)\)", metric.lower())[0]
        agg = "count"

    if any(word in query.lower() for word in ["count", "number", "how many"]):
        agg = "count"

    # -------- 🚨 FIX INVALID METRIC --------
    if metric not in df.columns:
        metric = None

    config["metric"] = metric
    config["aggregation"] = agg

    # -------- MATCH COLUMN --------
    def match_column(name):
        if not name:
            return None
        name = str(name).lower().replace("_", "").replace(" ", "")
        for col in df.columns:
            col_clean = col.lower().replace("_", "").replace(" ", "")
            if name in col_clean or col_clean in name:
                return col
        return None

    # =========================================================
    # GROUPING
    # =========================================================
    group_raw = config.get("group_by")

    if isinstance(group_raw, str) and "," in group_raw:
        group_raw = [g.strip() for g in group_raw.split(",")]
    elif not isinstance(group_raw, list):
        group_raw = [group_raw]

    group_cols = []
    for col in group_raw:
        matched = match_column(col)
        if matched:
            group_cols.append(matched)

    # -------- REMOVE HIGH CARDINALITY + ID --------
    filtered = []
    for col in group_cols:
        if df[col].nunique() < len(df) * 0.3 and "id" not in col:
            filtered.append(col)

    if filtered:
        group_cols = filtered

    # -------- AUTO ADD SECOND COLUMN --------
    if len(group_cols) == 1:
        for col in df.columns:
            if col not in group_cols:
                ratio = df[col].nunique() / len(df)
                if 0.01 < ratio < 0.5:
                    group_cols.append(col)
                    break

    if not group_cols:
        raise ValueError("❌ No valid group columns")

    # =========================================================
    # COUNT / NUNIQUE (FIXED)
    # =========================================================
    working_df = df

    use_nunique = False

    if agg == "count":
        if "unique" in query.lower():
            use_nunique = True
        elif metric and "id" in str(metric).lower():
            use_nunique = True
        elif any(word in query.lower() for word in ["customer", "user", "product"]):
            use_nunique = True

    if agg == "count":

        # ✅ USE NUNIQUE ONLY IF VALID METRIC EXISTS
        if use_nunique and metric:
            result = (
                working_df.groupby(group_cols)[metric]
                .nunique()
                .reset_index(name="count")
            )

        # ✅ SAFE FALLBACK (NO METRIC)
        else:
            result = (
                working_df.groupby(group_cols)
                .size()
                .reset_index(name="count")
            )

    else:
        if metric is None:
            raise ValueError("❌ Metric required for aggregation")

        working_df[metric] = pd.to_numeric(working_df[metric], errors="coerce")

        result = (
            working_df.groupby(group_cols)[metric]
            .agg(agg)
            .reset_index()
        )

    # =========================================================
    # PIVOT CONTROL
    # =========================================================
    by_count = query.lower().count("by")

    if len(group_cols) == 2 and by_count < 2:
        try:
            result = result.pivot(
                index=group_cols[0],
                columns=group_cols[1],
                values=result.columns[-1]
            ).fillna(0).reset_index()
        except:
            pass

    # =========================================================
    # FORCE COLUMN ORDER
    # =========================================================
    if "category" in result.columns:
        cols = ["category"] + [c for c in result.columns if c != "category"]
        result = result[cols]

    elif len(group_cols) >= 2:
        cols = group_cols[:2] + [c for c in result.columns if c not in group_cols[:2]]
        result = result[cols]

    # -------- SORT --------
    if result.shape[1] > 1:
        result = result.sort_values(by=result.columns[-1], ascending=False)

    # -------- TOP N --------
    if top_n:
        result = result.head(top_n)

    return result
