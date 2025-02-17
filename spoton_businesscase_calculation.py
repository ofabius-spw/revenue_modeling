import pandas as pd
import numpy as np

def schedule_day_ahead(df, Palt=50, Hs=0, Hw=0):
    """
    Schedule e-boiler operation based on dayahead market prices.
    """
    df = df.copy()
    df['boiler_active_dayahead'] = 0  # Default: Boiler off

    # Determine minimum hours for season (simplified: assuming winter if month >=10 or <=3)
    df['month'] = df['utc_timestamp'].dt.month
    min_hours = np.where(df['month'].isin([10, 11, 12, 1, 2, 3]), Hw, Hs)
    
    # Select the cheapest min_hours slots
    cheapest_hours = df.nsmallest(min_hours[0], 'Day-ahead Energy Price')['utc_timestamp']
    df.loc[df['utc_timestamp'].isin(cheapest_hours), 'boiler_active_dayahead'] = 1
    
    # Sort remaining hours where boiler is off by price
    remaining_off_hours = df[df['boiler_active_dayahead'] == 0].sort_values(by='Day-ahead Energy Price')
    
    # Turn on boiler if dayahead price < alternative heating price
    for idx in remaining_off_hours.index:
        if df.loc[idx, 'Day-ahead Energy Price'] < Palt:
            df.loc[idx, 'boiler_active_dayahead'] = 1

    return df

def bid_balancing_market(df):
    """
    Place bids in the balancing market based on mFRR prices.
    For now, we always do a bid in up OR down direction.
    """
    df2 = df.copy()
    # initialize with no bids
    df2['Bid_UP'] = 0
    df2['Bid_DOWN'] = 0

    # step 1: set all hours where Act == 1 to bid on UP direction #####
    # additional constraint: Customer revenues for mFRR up: 80% (capacity + energy activation) - 50€/MWh > 0
    revenue_up = 0.8 * (df2['up_cap_price'] + df2['up_energy_price']) - 50
    df2.loc[(df['boiler_active_dayahead'] == 1) & (revenue_up > 0), 'Bid_UP'] = 1

    # set all hours where Act == 0 and Dayahead energy price is below 50 to bid on DOWN direction #####
    df2.loc[(df['boiler_active_dayahead'] == 0) & (df['Day-ahead Energy Price']<50), 'Bid_DOWN'] = 1
    

    return df2


def calculate_bid_acceptance(df):
    """
    Calculate if the bid is accepted in the balancing market.
    Inputs: 
    df: the dataframe with the data

    outputs:
    
    """


    # TODO unpack variables from df
    Pe_up = df['up_energy_price']
    Pe_down = df['down_energy_price']

    clear_price_up = 0
    clear_price_down = 0
    Pc_up = df['up_cap_price']
    Pc_down = df['down_cap_price']
    Ep_up = df['up_energy_price']
    Ep_down = df['down_energy_price']
    
    # calculate bid price equivalent
    Eq_bid_price_up = Pc_up + Pe_up * df['Bid_UP']
    Eq_bid_price_down = Pc_down - Pe_down * df['Bid_DOWN']

    # check for acceptance
    accept_capacity_down = Eq_bid_price_down < Ep_down
    accept_capacity_up   = Eq_bid_price_up   < Ep_up
    accept_energy_down = (Eq_bid_price_down < Ep_down) & (Pe_down < clear_price_down)
    accept_energy_up   = (Eq_bid_price_up   < Ep_up)   & (Pe_up   < clear_price_up)

    return accept_capacity_down, accept_energy_down, accept_capacity_up, accept_energy_up

def calculate_revenue(df, Palt):
    """
    Calculate financial benefits of e-boiler operation and balancing market participation.

    inputs:
    df: the dataframe with the data. As implemented this will be data for one full day

    outputs:
    df: the dataframe with the data for same moments (rows) as the input, updated with the calculated revenues
    """
    df = df.copy()

    # Calculate bid acceptance for each hour in the dataframe of the day
    accept_capacity_down, accept_energy_down, accept_capacity_up, accept_energy_up = calculate_bid_acceptance(df)

    for idx in df.index:
        if df.loc[idx, 'boiler_active_dayahead'] == 1: # if boiler activated
            df.loc[idx, 'V_dayahead'] = Palt - df.loc[idx, 'Day-ahead Energy Price']
        # if we bid in UP direction
        if df.loc[idx, 'Bid_UP'] == 1:

            df.at[idx, 'V_bal_up_capacity'] += df.at[idx,'up_cap_price'] * accept_capacity_up[idx].astype(float)
            df.loc[idx, 'V_bal_up_energy'] += df.loc[idx,'up_energy_price'] * accept_energy_up[idx].astype(float)
            df.loc[idx, 'V_energyvalue_balancing_up'] += -Palt

        elif df.loc[idx, 'Bid_DOWN'] == 1:
            df.loc[idx, 'V_bal_down_energy'] = df.loc[idx, 'down_energy_price'] * df.loc[idx, 'bid_acceptance_probability']
            # TODO: check this next line
            # to check: revenues_down = + down_cap_price - down_energy_price
            df.loc[idx, 'V_bal_down_capacity'] += df.loc[idx, 'down_cap_price'] - df.loc[idx, 'down_energy_price']
            df.loc[idx, 'V_energyvalue_balancing_down'] += Palt

    return df

def convert_to_float_and_fill(df, columns):
    """
    Converts specified columns in the dataframe to float and fills NaN values with the column mean.
    
    Parameters:
    df (pd.DataFrame): The dataframe to process.
    columns (list): A list of column names to convert to float.
    
    Returns:
    pd.DataFrame: The updated dataframe with NaN values filled with the mean.
    """
    for column in columns:
        # Convert column to float (invalid entries will become NaN)
        df[column] = pd.to_numeric(df[column], errors='coerce')  
        # Fill NaN values with the column mean
        df[column].fillna(df[column].mean(), inplace=True)
    return df

def merge_overlapping_timeframes(df1, df2):
    """
    Merge two dataframes on their overlapping datetime values, accounting for missing timestamps by reindexing.
    
    Assumes both dataframes have a column named 'utc_timestamp'.
    
    Parameters:
    - df1: First dataframe with 'utc_timestamp' column.
    - df2: Second dataframe with 'utc_timestamp' column.
    
    Returns:
    - Merged dataframe with only the overlapping time range.
    """
    # Ensure datetime values are present in both dataframes
    if 'utc_timestamp' not in df1.columns or 'utc_timestamp' not in df2.columns:
        raise ValueError("Both dataframes must have 'utc_timestamp' column.")

    # Merge the dataframes on 'utc_timestamp' column (inner join to retain overlapping periods)
    merged_df = pd.merge(df1, df2, on='utc_timestamp', how='inner')

    return merged_df

def main(dayahead_file='dayahead_prices.csv', balancing_file='Finland - mFRR 2024 - Export for bc model.csv', Palt=50, Hs=0, Hw=0, output_file='results_eboiler.csv'):
    """
    Main function to process dayahead scheduling, balancing market bidding, and calculate revenues.
    """
    # Read dayahead prices
    df_da = pd.read_csv(dayahead_file, delimiter=',')
    df_da = df_da.assign(utc_timestamp=pd.to_datetime(df_da['utc_timestamp']))
    # NB we impute with mean values
    df_da = convert_to_float_and_fill(df_da, ['Day-ahead Energy Price']) 
 
    # Read balancing market data
    df_bal = pd.read_csv(balancing_file, parse_dates=['startTime'])
    df_bal.rename(columns={'startTime': 'utc_timestamp'}, inplace=True)
    df_bal['utc_timestamp'] = df_bal['utc_timestamp'].dt.tz_localize(None)
    # NB we impute with mean values
    df_bal = convert_to_float_and_fill(df_bal, ['up_energy_price', 'down_energy_price', 'bid_acceptance_probability'])
    
    # Merge dataframes on overlapping timeframes
    df = merge_overlapping_timeframes(df_da, df_bal)

    # Assuming df is already loaded and 'utc_timestamp' is a datetime object
    # Set Day 0 based on the first complete day
    first_complete_day = df[df['utc_timestamp'].dt.hour == 0].iloc[0]['utc_timestamp'].date()

    # add output columns to fill
    df['V_dayahead'] = 0.
    df['V_bal_up_energy'] = 0.
    df['V_bal_up_capacity'] = 0.
    df['V_bal_down_energy'] = 0.
    df['V_bal_down_capacity'] = 0.
    df['V_energyvalue_balancing_up'] = 0.
    df['V_energyvalue_balancing_down'] = 0.
    
    # Iterate over slices of the dataframe corresponding to full days
    processed_days = []
    for day_and_time in pd.date_range(start=first_complete_day, end=df['utc_timestamp'].dt.date.max(), freq='D'):
        day = day_and_time.date() 
        day_slice = df[df['utc_timestamp'].dt.date == day]
       
        # Check if the slice contains 24 rows (one for each hour)
        if len(day_slice) == 24:
            # Call functions with the selected day slice
            day_slice = schedule_day_ahead(day_slice, Palt=50, Hs=2, Hw=4)
            day_slice = bid_balancing_market(day_slice)
            day_slice = calculate_revenue(day_slice, Palt=50)
            
            # Append the processed day slice to the list
            processed_days.append(day_slice)
            
            print(f'Processed day: {day}')
            
        else:
            print(f'Warning: Missing data for day {day}, skipping.')
    # Concatenate all processed day slices into a single dataframe
    df_processed = pd.concat(processed_days)
    df_processed.reset_index(drop=True, inplace=True)
    print(f'All days processed')

    return df_processed

if __name__ == '__main__':
    # Run the main function with default parameters
    df_results = main(dayahead_file='dayahead_prices_finland_2024.csv', balancing_file='Finland - mFRR 2024 - Export for bc model.csv', Palt=50, Hs=2, Hw=4)
    df_results.to_csv('spoton_model_results.csv')