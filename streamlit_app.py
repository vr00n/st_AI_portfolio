import streamlit as st
import openai
import datetime
import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
import traceback

st.set_page_config(page_title="AI Portfolio Generator", layout="wide")
st.title("📊 AI-Powered Investment Portfolio Simulator")

# User input fields
investment_thesis = st.text_area("Describe your investment thesis")
openai_api_key = st.text_input("OpenAI API Key", type="password")
fmp_api_key = st.text_input("FMP API Key", type="password")

timeframe = st.selectbox("Performance View Range", ["MTD", "QTD", "YTD", "1Y", "5Y", "Since Custom Date"])
custom_start_date = None
if timeframe == "Since Custom Date":
    custom_start_date = st.date_input("Select custom start date", value=datetime.date.today() - datetime.timedelta(days=365))

if st.button("Generate Portfolio"):
    if not investment_thesis or not openai_api_key or not fmp_api_key:
        st.error("Please provide all required fields.")
    else:
        with st.spinner("Generating portfolio..."):
            client = openai.OpenAI(api_key=openai_api_key)

            prompt = f"""
Please respond with a json object only. Do not include any introductory or concluding text, just the raw json. You are a financial advisor. Based on the following investment thesis, generate a diversified investment portfolio suitable for a starting capital of $10,000.
The portfolio should consist of 10-15 assets, with approximately 70% allocation to ETFs and 30% to individual stocks or bonds.
For each asset, provide its ticker symbol, a proposed percentage allocation (summing to 100%), and a brief justification for its inclusion based on the investment thesis.
Additionally, provide an overall justification for the portfolio strategy.
The json object should have two top-level keys: 'portfolio' (an array of asset objects) and 'overallJustification' (a string).
Each asset object in the 'portfolio' array should have 'symbol' (string), 'name' (string), 'allocation' (number), and 'justification' (string).

Investment Thesis: "{investment_thesis}"
"""

            try:
                # Debugging log for payload
                st.code(prompt, language="markdown")
                st.info("Sending request to OpenAI API...")

                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    tool_choice="none",
                    extra_headers={"OpenAI-Beta": "assistants=v1"},
                    response_format="json_object"
                )

                st.success("OpenAI response received successfully.")
                st.json(response.model_dump())

                content = response.choices[0].message.content
                st.code(content, language="json")

                portfolio_data = json.loads(content)
                portfolio = portfolio_data['portfolio']
                justification = portfolio_data['overallJustification']

                st.subheader("📌 Overall Strategy Justification")
                st.write(justification)

                df = pd.DataFrame(portfolio)
                total_alloc = df['allocation'].sum()
                if abs(total_alloc - 100) > 0.1:
                    st.warning(f"Total allocation is {total_alloc:.2f}%. Adjusting.")
                    df['allocation'] = df['allocation'] / total_alloc * 100

                st.subheader("📈 Portfolio Composition")
                st.dataframe(df[['symbol', 'name', 'allocation', 'justification']])

                st.subheader("📊 Simulated Performance vs SPY")
                num_days = {
                    "MTD": (datetime.date.today() - datetime.date.today().replace(day=1)).days,
                    "QTD": (datetime.date.today() - datetime.date(datetime.date.today().year, 3 * ((datetime.date.today().month - 1) // 3) + 1, 1)).days,
                    "YTD": (datetime.date.today() - datetime.date(datetime.date.today().year, 1, 1)).days,
                    "1Y": 365,
                    "5Y": 365 * 5
                }.get(timeframe, (datetime.date.today() - custom_start_date).days if custom_start_date else 365)

                dates = pd.date_range(end=datetime.date.today(), periods=num_days)
                perf_df = pd.DataFrame({
                    "Date": dates,
                    "Portfolio": 10000 + np.cumsum(np.random.randn(num_days) * 10),
                    "SPY": 10000 + np.cumsum(np.random.randn(num_days) * 8)
                })

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(perf_df["Date"], perf_df["Portfolio"], label="Portfolio", color="blue")
                ax.plot(perf_df["Date"], perf_df["SPY"], label="S&P 500", color="orange")
                ax.set_xlabel("Date")
                ax.set_ylabel("Value ($)")
                ax.set_title("Performance Comparison")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.exception(traceback.format_exc())
