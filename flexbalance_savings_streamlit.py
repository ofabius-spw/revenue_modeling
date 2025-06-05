import streamlit as st
import pandas as pd
import numpy as np
import csv
import io
import altair as alt
import matplotlib.pyplot as plt

# --- Functions ---
def generate_activation_decisions(prices, activate_above, activate_below, false_neg_rate, false_pos_rate, system_imbalances, min_abs_system_imbalance, max_activations_per_day=None):
    decisions = []
    for price, imbalance in zip(prices, system_imbalances):
        decision = 0
        if abs(imbalance) > min_abs_system_imbalance:
            if price > activate_above:
                decision = 1 # activate_up
            elif price < activate_below:
                decision = 2  # activate_down
        decisions.append(decision)  # no activation
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
    savings_net = []
    savings_gross = []
    efficiency = max(0, 1 - delay_minutes / ptu_duration)
    for decision, price in zip(decisions, prices):
        duration_in_hours = ptu_duration / 60
        if decision == 1:
            saving_gross = price * efficiency * duration_in_hours
            saving_net = (price - activation_cost_up_eur) * efficiency * duration_in_hours
        elif decision == 2:
            saving_gross = -price * efficiency * duration_in_hours
            saving_net = (-price - activation_cost_down_eur) * efficiency * duration_in_hours
        else:
            saving_net = 0
            saving_gross = 0
        savings_net.append(saving_net)
        savings_gross.append(saving_gross)
    return savings_net, savings_gross

def limit_daily_activations(decisions, max_decisions, availability_mask):
    # Create a copy to avoid modifying the original array
    decisions = decisions.copy()
    
    # Process the array in chunks of 96
    n = len(decisions)
    for start in range(0, n, 96):

        # End index is either the next 96 elements or end of array
        end = min(start + 96, n)
        
        # Get the current chunk (a full or partial day)
        day_slice = decisions[start:end]
        if len(day_slice) == len(availability_mask):
            day_slice = day_slice * availability_mask
        else:
            # If the length of the day_slice is less than 96, apply the mask only to the available part
            day_slice = day_slice * availability_mask[:len(day_slice)]

        # Find indices where decisions are nonzero
        nonzero_indices = np.nonzero(day_slice)[0]
        decisions[start:end] = day_slice
        # If there are more nonzero decisions than allowed, set the extras to 0
        if len(nonzero_indices) > max_decisions:
            indices_to_zero = nonzero_indices[max_decisions:]
            # Apply the change to the appropriate slice of the original array
            decisions[start + indices_to_zero] = 0
            
    return decisions


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
activation_cost_down_eur = st.sidebar.number_input("Activation cost for negative imbalance (EUR/MW)", value=100.0)
max_activations_per_day = st.sidebar.number_input("Maximum number of asset activations per day", value=96, min_value=1)

st.sidebar.markdown("**NB Availability only works if the first PTU in the dataset is at midnight**")
availability_start = st.sidebar.number_input("Availability start (hours)", value=0, min_value=0)
availability_end = st.sidebar.number_input("Availability end (hours)", value=24, min_value=0)
if availability_start >= availability_end:
    st.error("Availability start must be smaller than the end time.")
    st.stop()
activate_above = st.sidebar.number_input("Activate above price (input currency per MW)", value=1200.0)
activate_below = st.sidebar.number_input("Activate below price (input currency per MW)", value=-500.0)
min_abs_sys_imbalance = st.sidebar.number_input("Minimum absolute system imbalance (MW). NB if you use a MWh number (instead of in MW) from the inout csv for this, adjust this value accordingly. For example for 15-minute-ptus, use 40 (MWh) instead of 10 (MW) ", min_value=0.00, value=0.00)

exchange_rate = st.sidebar.number_input("Exchange rate: 1 unit of input currency = ? EUR", min_value=0.0001, value=1.0)

false_negatives = st.sidebar.number_input("False negatives (%)", min_value=0.00, max_value=100.00, value=0.00)
false_positives = st.sidebar.number_input("False positives (%)", min_value=0.00, max_value=100.00, value=0.00)
start_of_first_ptu = st.sidebar.text_input("Start of first PTU (e.g., 2024-01-01 00:00)", value="2024-01-01 00:00")

if uploaded_file:
    # Detect delimiter
    content = uploaded_file.getvalue().decode("utf-8")
    sniffer = csv.Sniffer()
    dialect = sniffer.sniff(content.splitlines()[0])
    delimiter = dialect.delimiter

    # Read CSV
    df = pd.read_csv(io.StringIO(content), delimiter=delimiter)

    # Select actual imbalance price column
    selected_column = st.selectbox("Select the column with **actual imbalance prices** (used for savings calculation)", df.columns)

    # Optional: forecast column for decision-making
    forecast_column = st.selectbox(
        "Optional: Select a column with **forecast imbalance prices** (used only for activation decisions)",
        ["<Use actual prices>"] + ["synthetic_prediction"]+list(df.columns),
        index=0
    )
    system_imbalance_column = st.selectbox("Select the column with *system imbalance** (used for minimum required system imbalance)", ["<No minimum>"] + list(df.columns))

    try:
        df[selected_column] = df[selected_column].astype(float)
        if forecast_column not in ["<Use actual prices>", "synthetic_prediction"]:
            df[forecast_column] = df[forecast_column].astype(float)
        if system_imbalance_column != "<No minimum>":
            df[system_imbalance_column] = df[system_imbalance_column].astype(float)
    except ValueError:
        st.error("Selected column(s) contain non-numeric values.")
        st.stop()

    df['price_eur'] = df[selected_column] * exchange_rate

    ##################### create synthetic data #####################
    if forecast_column == "synthetic_prediction":
        np.random.seed(42)  # For reproducibility
        st.subheader("Create Synthetic Data for Forecast Imbalance Prices")
        default_std = float(df['price_eur'].std())

        # === PARAMETERS ===
        with st.expander("Synthetic Prediction Settings", expanded=True):
            sqrt_of_dataset = np.sqrt(df['price_eur'] - np.mean(df['price_eur']))
            st.write("The square root of the dataset is {}. Going above this is not recommended.".format(sqrt_of_dataset.mean()))
            scalar = st.number_input("Scalar multiplier (applied after sqrt)", value=1.0, max_value=sqrt_of_dataset.max(), min_value=0.0)
            noise_mean = np.mean(df['price_eur']) # st.number_input("Gaussian noise mean", value=0.0)
            noise_std = st.number_input("Gaussian noise std", value=np.sqrt(default_std))
            last_n = st.number_input("Plot last N datapoints", min_value=2, max_value=len(df), value=500)

        # === GENERATE SYNTHETIC DATA ===
        base_prediction = (
            np.sqrt(np.abs(df['price_eur']-np.mean(df['price_eur']))) * scalar * np.sign(df['price_eur']-np.mean(df['price_eur'])) + np.mean(df['price_eur']) 
        )
        noise = np.random.normal(loc=noise_mean, scale=noise_std, size=len(df))
        df["synthetic_prediction"] = base_prediction + noise

        # === PLOT ===
        st.subheader("Real vs Synthetic (Last N Points)")
        fig, ax = plt.subplots(figsize=(10, 5))
        last_indices = df.index[-last_n:]

        ax.plot(last_indices, df['price_eur'].iloc[-last_n:], label="Real Value", color="blue", alpha=0.7)
        ax.plot(last_indices, df["synthetic_prediction"].iloc[-last_n:], label="Synthetic Prediction", color="orange", alpha=0.7)

        ax.set_xlabel("Index")
        ax.set_ylabel("Value (EUR/MWh)")
        ax.set_title(f"Real vs Synthetic Prediction (Last {last_n} Points)")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

    if forecast_column == "<Use actual prices>":
        decision_prices = df['price_eur']
    elif forecast_column == "synthetic_prediction":
        decision_prices = df['synthetic_prediction']
    else:
        decision_prices = df[forecast_column] * exchange_rate
    # decision_prices = df[forecast_column] * exchange_rate if forecast_column != "<Use actual prices>" else df['price_eur']
    system_imbalance = df[system_imbalance_column] if system_imbalance_column != "<No minimum>" else np.ones_like(df['price_eur'])*10000

        # # === DOWNLOAD UPDATED CSV ===
        # csv = df.to_csv(index=False).encode("utf-8")
        # st.download_button(
        #     label="Download updated CSV",
        #     data=csv,
        #     file_name="with_synthetic_predictions.csv",
        #     mime="text/csv"
        # )
    #####################################################################


    # Generate timestamps
    try:
        start_time = pd.to_datetime(start_of_first_ptu)
        df['timestamp'] = [start_time + pd.Timedelta(minutes=ptu_duration * i) for i in range(len(df))]
    except Exception as e:
        st.error(f"Error parsing start time: {e}")
        st.stop()

    # Generate activation decisions based on forecast or actual
    decisions = generate_activation_decisions(
        decision_prices, activate_above, activate_below, false_negatives, false_positives, system_imbalance, min_abs_sys_imbalance
    )
    
    # decision = 0 for unavailable hours
    availability_mask = np.zeros(96, dtype=int)

    # Convert hours to quarter-hour indices
    start_index = availability_start * 4
    end_index = availability_end * 4  

    # Set mask to 1 for selected range
    availability_mask[start_index:end_index] = 1
    decisions = limit_daily_activations(decisions, max_activations_per_day, availability_mask)
    df['resource_activated'] = decisions
    # Calculate savings using actual prices
    df['savings_net'], df['savings_gross'] = calculate_savings(
        df['resource_activated'], df['price_eur'], activation_cost_up_eur,
        activation_cost_down_eur, delay_minutes, ptu_duration
    )

    # Activation cost and metadata
    # df['activation_cost'] = df['resource_activated'].apply(lambda x: activation_cost_up_eur if x == 1 else (activation_cost_down_eur if x == 2 else 0))
    # df['decision'] = df['resource_activated']
    df['day_from_start'] = (np.arange(len(df)) // (60 / ptu_duration * 24)).astype(int) + 1

    # Export dataframe
    df_export = df[['timestamp', selected_column, 'resource_activated', 'savings_gross', 'savings_net', 'day_from_start']]

    # Daily savings aggregation
    df['day'] = (np.arange(len(df)) // (60 / ptu_duration * 24)).astype(int) + 1
    daily_savings = df.groupby('day')['savings_net'].sum().reset_index()

    # Annualized metrics
    total_net_savings = df['savings_net'].sum()
    total_gross_savings = df['savings_gross'].sum()
    total_hours = (len(df) * ptu_duration) / 60

    gross_annualized = (total_gross_savings / total_hours) * 8760
    net_annualized = (total_net_savings / total_hours) * 8760

    st.metric("Gross Revenues per MW per Year (EUR)", f"{gross_annualized:,.2f}")
    st.metric("Net Revenues per MW per Year (EUR)", f"{net_annualized:,.2f}")

    st.subheader("Daily Savings (EUR)")
    st.line_chart(daily_savings.set_index('day'))

    st.subheader("Inspect PTU-Level Savings for a Specific Day Number")

    max_day = df['day'].max()
    selected_day = st.selectbox("Select day number to inspect PTU-level costs", options=[None] + list(range(1, max_day + 1)), index=0)

    if selected_day is not None:
        filtered_df = df[df['day'] == selected_day].copy()
        filtered_decision_prices = decision_prices.loc[filtered_df.index]

        ptu_chart_df = pd.DataFrame({
            'timestamp': filtered_df['timestamp'],
            'Net Savings (EUR)': filtered_df['savings_net'],
            'Gross Savings (EUR)': filtered_df['savings_gross'],
            'Imbalance Price (EUR/MW)': filtered_df['price_eur'],
            'Forecast Price (EUR/MW)': filtered_decision_prices
        })

        line_order = [
            ('Forecast Price (EUR/MW)', 'green'),
            ('Imbalance Price (EUR/MW)', 'red'),
            ('Gross Savings (EUR)', '#1f77b4'),
            ('Net Savings (EUR)', '#9467bd')
        ]

        # Convert to long format for Altair legend support
        ptu_chart_df_long = ptu_chart_df.melt(
            id_vars=['timestamp'],
            value_vars=[metric for metric, _ in line_order],
            var_name='Metric', value_name='Value'
        )

        color_scale = alt.Scale(
            domain=[metric for metric, _ in line_order],
            range=[color for _, color in line_order]
        )

        ptu_chart_with_legend = alt.Chart(ptu_chart_df_long).mark_line().encode(
            x=alt.X('timestamp:T', title='Timestamp'),
            y=alt.Y('Value:Q', title='EUR'),
            color=alt.Color('Metric:N', scale=color_scale, legend=alt.Legend(title="Metrics")),
            tooltip=['timestamp:T', 'Metric', 'Value']
        ).properties(height=350)

        st.altair_chart(ptu_chart_with_legend, use_container_width=True)
    else:
        st.info("Select a day number above to display PTU-level savings and prices.")


    # Activation statistics section (toggle)
    if st.checkbox("Show Activation Statistics"):
        st.subheader("Activation Statistics")

        total_activations = (df['resource_activated'] != 0).sum()
        total_days = (df['timestamp'].max() - df['timestamp'].min()).days + 1
        avg_activations_per_day = total_activations / total_days if total_days > 0 else total_activations

        loss_mask = (df['resource_activated'] != 0) & (df['savings_net'] < 0)

        percent_loss_activations = 100 * loss_mask.sum() / total_activations if total_activations > 0 else 0

        price_based_loss = df.apply(lambda row: (
            row['price_eur'] < 0 if row['resource_activated'] == 1
            else row['price_eur'] > 0 if row['resource_activated'] == 2
            else False
        ), axis=1)
        percent_price_loss_activations = 100 * price_based_loss.sum() / total_activations if total_activations > 0 else 0

        activations_per_day = df[df['resource_activated'] != 0].groupby('day')['resource_activated'].count()
        max_activations_in_day = activations_per_day.max() if not activations_per_day.empty else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Total activations", f"{total_activations}")
        col1.metric("Avg activations/day", f"{avg_activations_per_day:.1f}")
        col2.metric("% PTUs activated with net loss", f"{percent_loss_activations:.1f}%")
        col2.metric("% PTUs gross loss (wrong direction)", f"{percent_price_loss_activations:.1f}%")
        col3.metric("Max activations/day", f"{max_activations_in_day}")

        st.subheader("Daily Activation Count")
        daily_activations = df[df['resource_activated'] != 0].groupby('timestamp').size().resample('D').sum().reset_index(name='count')
        daily_chart = alt.Chart(daily_activations).mark_line().encode(
            x='timestamp:T',
            y=alt.Y('count:Q', title='Activations'),
            tooltip=['timestamp:T', 'count:Q']
        ).properties(height=300)
        st.altair_chart(daily_chart, use_container_width=True)

    st.download_button(
        label="Download Results as CSV",
        data=df_export.to_csv(index=False).encode('utf-8'),
        file_name='savings_results.csv',
        mime='text/csv'
    )