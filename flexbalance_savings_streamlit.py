import streamlit as st
import pandas as pd
import numpy as np
import datetime

# --- Function for activation decision with errors ---
def simulate_activation_with_errors(prices, above_threshold, below_threshold, false_negative_rate, false_positive_rate):
    """
    Simulates resource activation decisions with errors based on false negatives and false positives.
    Returns an integer (0, 1, or 2) for each price:
    - 0: no activation
    - 1: activate up
    - 2: activate down
    """
    decisions = []

    # Initial perfect model activation logic
    for price in prices:
        if price > above_threshold:
            decisions.append(1)  # activate_up
        elif price < below_threshold:
            decisions.append(2)  # activate_down
        else:
            decisions.append(0)  # no_activation

    # Apply False Negatives (FN)
    # FN: Change some 1 or 2 to 0 based on false_negative_rate
    decisions_with_fn = [
        0 if np.random.rand() < (false_negative_rate / 100) and decision != 0 else decision
        for decision in decisions
    ]

    # Apply False Positives (FP)
    # FP: Change some 0 to 1 or 2 based on false_positive_rate
    decisions_with_fp = [
        np.random.choice([1, 2]) if decision == 0 and np.random.rand() < (false_positive_rate / 100) else decision
        for decision in decisions_with_fn
    ]

    return np.array(decisions_with_fp)


# --- Streamlit UI Setup ---
st.title("Energy Flexibility Savings Calculator")
st.markdown("Upload a CSV file with an 'imbalance_price' column")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# Sidebar inputs
st.sidebar.header("Model Parameters")
delay = st.sidebar.number_input("Delay in resource activation (PTUs)", min_value=0, value=0)
ptu_duration = st.sidebar.number_input("Price-time-unit duration (minutes)", min_value=1, value=15)
activation_cost = st.sidebar.number_input("Resource activation cost (EUR/MWh)", min_value=0.0, value=0.0)
exchange_rate = st.sidebar.number_input("Exchange rate to EUR (set 1 if already in EUR)", min_value=0.0001, value=1.0)
accuracy = st.sidebar.slider("Classification model accuracy (%)", min_value=0, max_value=100, value=100)
start_of_first_ptu = st.sidebar.text_input("Start of first PTU (e.g., 2024-01-01 00:00)", value="2024-01-01 00:00")

# New logic inputs with default values
activate_above = st.sidebar.number_input("Activate when price is ABOVE (EUR/MWh)", value=1200.0)
activate_below = st.sidebar.number_input("Activate when price is BELOW (EUR/MWh)", value=-500.0)
false_negative_rate = st.sidebar.number_input("False Negative Rate (%)", min_value=0, max_value=100, value=0)
false_positive_rate = st.sidebar.number_input("False Positive Rate (%)", min_value=0, max_value=100, value=0)

# --- Main App Logic ---
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "imbalance_price" not in df.columns:
        st.error("CSV must contain a column named 'imbalance_price'")
    else:
        # Apply exchange rate
        df["imbalance_price_eur"] = df["imbalance_price"] / exchange_rate

        # Simulate activation decision with errors (based on FN and FP rates)
        df["resource_activated"] = simulate_activation_with_errors(
            df["imbalance_price_eur"], activate_above, activate_below, false_negative_rate, false_positive_rate
        )

        # Placeholder savings logic
        df["savings"] = df["resource_activated"].apply(lambda x: 100 if x != 0 else 0)

        # Generate timestamps for PTUs
        try:
            start_time = pd.to_datetime(start_of_first_ptu)
            df["timestamp"] = [start_time + pd.Timedelta(minutes=ptu_duration * i) for i in range(len(df))]
        except Exception as e:
            st.error(f"Error parsing start time: {e}")
            st.stop()

        # Aggregate daily savings
        df["date"] = df["timestamp"].dt.date
        daily_savings = df.groupby("date")["savings"].sum().reset_index()

        # Add day number for x-axis
        daily_savings["day_number"] = range(1, len(daily_savings) + 1)

        # Calculate total savings per MW per year
        total_savings = df["savings"].sum()
        total_hours = (len(df) * ptu_duration) / 60
        annualized_savings = (total_savings / total_hours) * 8760 if total_hours else 0

        st.metric("Potential Savings per MW per Year (EUR)", f"{annualized_savings:,.2f}")

        st.download_button(
            label="Download Results as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="savings_results.csv",
            mime="text/csv",
        )

        st.write("🔍 Preview of processed data:", df[["timestamp", "imbalance_price_eur", "resource_activated", "savings"]].head())

        # Show daily savings chart if data is available
        if not daily_savings.empty:
            st.subheader("📈 Daily Savings (EUR)")
            st.line_chart(daily_savings.set_index("day_number")["savings"])
        else:
            st.warning("No daily savings data available to plot.")
