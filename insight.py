import json
from langchain_community.llms import Ollama   # ✅ CHANGED


def df_to_summary(df):
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "sample": df.head(5).to_dict(orient="records"),
        "stats": df.describe(include="all").fillna("").to_dict()
    }


def generate_insights(final_df, query):

    summary = df_to_summary(final_df)

    prompt = f"""
You are a data analyst.

User Query: {query}

Dataset Summary:
{json.dumps(summary, indent=2)}

STRICT RULES:
- Output EXACTLY 3 bullet points
- Each bullet MUST be one line
- Each bullet must be a meaningful insight (trend, highest, lowest, comparison)
- DO NOT print raw values alone
- DO NOT print column names alone
- DO NOT explain anything
- DO NOT add extra text
- Each line must start with "- "

GOOD EXAMPLE:
- Brand A has the highest sales
- Sales increased over time
- Category X contributes most revenue

BAD EXAMPLE:
- sales: 1000
- brand: A

Now generate insights:
"""

    # -------- OLLAMA MODEL --------
    llm = Ollama(
        model="llama3",   # llama3:8b
        temperature=0
    )

    response = llm.invoke(prompt)

    return response