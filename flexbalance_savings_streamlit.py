import streamlit as st
import pandas as pd
import numpy as np
import csv
import io

# --- Functions ---
def generate_activation_decisions(prices, activate_above, activate_below, false_neg_rate, false_pos_rate):
    decisions = []
    for price in prices:
        if price > activate_above:
            decisions.append(1)  # activate_up
        elif price < activate_below:
            decisions.append(2)  # activate_down
        else:
            decisions.append(0)  # no activation
    decisions = np.array(decisions)

    # Apply false negatives (turn 1 or 2 into 0)
    is_activation = (decisions == 1) | (decisions == 2)
    false_negatives = np.random.rand(len(decisions)) < (false_neg_rate / 100)
    decisions[is_activation & false_negatives] = 0

    # Apply false positives (turn 0 into 1 or 2)
    is_no_activation = decisions == 0
    false_positives = np.random.rand(len(decisions)) < (false_pos_rate / 100)
    flip = np.random.choice([1, 2], size=len(decisions))
    decisions[is_no_activation & false_positives] = flip[is_no_activation & false_positives]

    return decisions

def calculate_savings(decisions, prices, activation_cost_up_eur, activation_cost_down_eur, delay_minutes, ptu_duration):
    savings = []
    efficiency = max(0, 1 - delay_minutes / ptu_duration)
    for decision, price in zip(decisions, prices):
        # Calculate the duration of activation in hours
        duration_in_hours = (ptu_duration) / 60
        if decision == 1:
            saving = (price - activation_cost_up_eur) * efficiency * duration_in_hours
        elif decision == 2:
            saving = (-price - activation_cost_down_eur) * efficiency * duration_in_hours
        else:
            saving = 0
        savings.append(saving)
    return savings

# --- Streamlit App ---
st.title("Energy Flexibility Savings Calculator")
st.markdown("Upload a CSV file containing imbalance prices and calculate your potential savings in EUR.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# Sidebar inputs
st.sidebar.header("Model Parameters")
delay_minutes = st.sidebar.number_input("Delay in resource activation (minutes)", min_value=0, value=0)
ptu_duration = st.sidebar.number_input("Price-time-unit duration (minutes)", min_value=1, value=15)
if delay_minutes >= ptu_duration:
    st.error("Delay must be smaller than the PTU duration.")
    st.stop()

activation_cost_up_eur = st.sidebar.number_input("Activation cost for positive imbalance (EUR/MW)", min_value=0.0, value=800.0)
activation_cost_down_eur = st.sidebar.number_input("Activation cost for negative imbalance (EUR/MW)", min_value=0.0, value=100.0)
exchange_rate = st.sidebar.number_input("Exchange rate: 1 unit of input currency = ? EUR", min_value=0.0001, value=1.0)
activate_above = st.sidebar.number_input("Activate above price (input currency per MW)", value=1200.0)
activate_below = st.sidebar.number_input("Activate below price (input currency per MW)", value=-500.0)
false_negatives = st.sidebar.slider("False Negatives (%)", min_value=0, max_value=100, value=0)
false_positives = st.sidebar.slider("False Positives (%)", min_value=0, max_value=100, value=0)
start_of_first_ptu = st.sidebar.text_input("Start of first PTU (e.g., 2024-01-01 00:00)", value="2024-01-01 00:00")

if uploaded_file:
    # Detect delimiter
    content = uploaded_file.getvalue().decode("utf-8")
    sniffer = csv.Sniffer()
    dialect = sniffer.sniff(content.splitlines()[0])
    delimiter = dialect.delimiter

    # Read CSV with detected delimiter
    df = pd.read_csv(io.StringIO(content), delimiter=delimiter)

    # Let user select the price column
    selected_column = st.selectbox("Select the column containing imbalance prices (input currency per MW)", df.columns)

    # Ensure numeric
    try:
        df[selected_column] = df[selected_column].astype(float)
    except ValueError:
        st.error(f"The selected column '{selected_column}' contains non-numeric values. Please select a numeric column.")
        st.stop()

    # Convert to EUR (only imbalance prices, not activation costs)
    df['price_eur'] = df[selected_column] * exchange_rate

    # Generate timestamps
    try:
        start_time = pd.to_datetime(start_of_first_ptu)
        df['timestamp'] = [start_time + pd.Timedelta(minutes=ptu_duration * i) for i in range(len(df))]
    except Exception as e:
        st.error(f"Error parsing start time: {e}")
        st.stop()

    # Activation decisions and apply delay
    raw_decisions = generate_activation_decisions(
        df['price_eur'], activate_above, activate_below,
        false_negatives, false_positives
    )
    delay_ptus = delay_minutes // ptu_duration
    df['resource_activated'] = 0
    if delay_ptus > 0:
        df.loc[df.index[delay_ptus]:, 'resource_activated'] = raw_decisions[:-delay_ptus]
    else:
        df['resource_activated'] = raw_decisions

    # Calculate savings
    df['savings'] = calculate_savings(
        df['resource_activated'], df['price_eur'], activation_cost_up_eur,
        activation_cost_down_eur, delay_minutes, ptu_duration
    )

    # Calculate activation costs
    df['activation_cost'] = df['resource_activated'].apply(lambda x: activation_cost_up_eur if x == 1 else (activation_cost_down_eur if x == 2 else 0))

    # Calculate decision per PTU (already in the data)
    df['decision'] = df['resource_activated']

    # Calculate day from start
    df['day_from_start'] = (df.index // (60 / ptu_duration * 24)).astype(int) + 1

    # Select columns to export
    df_export = df[['timestamp', selected_column, 'decision', 'activation_cost', 'savings', 'day_from_start']]

    # Daily aggregation
    df['day'] = (df.index // (60 / ptu_duration * 24)).astype(int) + 1
    daily_savings = df.groupby('day')['savings'].sum().reset_index()

    # Annualized metric
    total_savings = df['savings'].sum()
    total_hours = (len(df) * ptu_duration) / 60
    annualized_savings = (total_savings / total_hours) * 8760

    st.metric("Potential Savings per MW per Year (EUR)", f"{annualized_savings:,.2f}")

    st.download_button(
        label="Download Results as CSV",
        data=df_export.to_csv(index=False).encode('utf-8'),
        file_name='savings_results.csv',
        mime='text/csv'
    )

    st.subheader("Daily Savings (EUR)")
    st.line_chart(daily_savings.set_index('day'))
