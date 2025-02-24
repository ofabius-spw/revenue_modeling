import pandas as pd
import numpy as np

def schedule_day_ahead(df, max_hours_per_month=None, Palt=50, ):
    """
    Schedule e-boiler operation based on dayahead market prices.
    """
    df = df.copy()
    df['boiler_active_dayahead'] = 0  # Default: Boiler off

    # Determine minimum hours for season (simplified: assuming winter if month >=10 or <=3)
    if max_hours_per_month is not None:
        month = df['utc_timestamp'].dt.month.iloc[0]
        max_hours = max_hours_per_month[month]
    else:
        max_hours=24

    # Select the cheapest max_hours slots
    cheapest_hours = df.nsmallest(max_hours, 'Day-ahead Energy Price')['utc_timestamp']
    # print(cheapest_hours, 'cheapest hours')
    df.loc[df['utc_timestamp'].isin(cheapest_hours), 'boiler_active_dayahead'] = 1
    # print(df.loc[:, 'boiler_active_dayahead'], 'boiler active dayahead')
    # Sort remaining hours where boiler is off by price
    on_hours = df[df['boiler_active_dayahead'] == 1]
    
    # Turn off boiler if dayahead price > alternative heating price
    for idx in on_hours.index:
        if df.loc[idx, 'Day-ahead Energy Price'] > Palt:
            df.loc[idx, 'boiler_active_dayahead'] = 0

    df['max_dayahead_hours'] = max_hours # for output reporting only

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

    # set all hours where boiler is ON to bid on UP direction #####
    # additional constraint: Customer revenues for mFRR up: 80% (capacity + energy activation) - 50€/MWh > 0
    revenue_up = 0.8 * (df2['up_capacity_price'] + df2['up_energy_price']) + df['Day-ahead Energy Price'] - 50
    # set bid in up direction if revenue is positive
    df2.loc[(df['boiler_active_dayahead'] == 1) & (revenue_up > 0), 'Bid_UP'] = 1

    # set all hours where boiler is OFF and Dayahead energy price is below 50 to bid on DOWN direction #####    
    revenue_down = 0.8 * (df2['down_capacity_price'] - df2['down_energy_price']) - df['Day-ahead Energy Price']
    df2.loc[(df['boiler_active_dayahead'] == 0) & (revenue_down > 0), 'Bid_DOWN'] = 1


    # for idx in range(1, 24):

        # if df.loc[idx, 'boiler_active_dayahead'] == 1:
            # print('boiler status', df.loc[idx, 'boiler_active_dayahead'])
            # print('up price:', df.loc[idx, 'up_energy_price'])
            # print('capacity price:', df.loc[idx, 'up_capacity_price'])
            # print('dayahead price:', df.loc[idx, 'Day-ahead Energy Price'])
            # print('revenue_up:', revenue_up[idx])
            # print('active:', df.loc[idx, 'boiler_active_dayahead'])
            # print('bid up:', df2.loc[idx, 'Bid_UP'])
            # print('boiler status after:', df2.loc[idx, 'boiler_active_dayahead'])
            # input('Press Enter to continue...')

    return df2

def initialize_output_columns(df):
    
    df['V_dayahead'] = 0.
    df['V_bal_up_energy'] = 0.
    df['V_bal_up_capacity'] = 0.
    df['V_bal_down_energy'] = 0.
    df['V_bal_down_capacity'] = 0.
    df['V_energyvalue_balancing_up'] = 0.
    df['V_energyvalue_balancing_down'] = 0.
    df['max_dayahead_hours'] = 0 # used only for output reporting

    return df

def calculate_bid_acceptance(df, bpp):
    """
    Calculate if the bid is accepted in the balancing market.
    Inputs: 
    df: the dataframe with the data

    outputs:
    accept_capacity_down: boolean array, True if the bid is accepted for capacity in the down direction
    accept_energy_down: boolean array, True if the bid is accepted for energy in the down direction
    accept_capacity_up: boolean array, True if the bid is accepted for capacity in the up direction
    accept_energy_up: boolean array, True if the bid is accepted for energy in the up direction
    """
    # market values (cleared prices):
    cleared_price_energy_up     = df['up_energy_price']
    cleared_price_energy_down   = df['down_energy_price']
    cleared_price_capacity_up   = df['up_capacity_price']
    cleared_price_capacity_down = df['down_capacity_price']
    equivalent_cleared_price_up    = df['up_equivalent_price']
    equivalent_cleared_price_down  = df['down_equivalent_price']

    # bidding values
    # TO DO: add the correct column name for each variable
    bid_price_energy_up   = bpp['bid_price_energy_up']
    bid_price_energy_down = bpp['bid_price_energy_down']
    bid_price_capacity_up = bpp['bid_price_capacity_up']
    bid_price_capacity_down = bpp['bid_price_capacity_down']
    activation_derating_factor_down  = bpp['activation_derating_factor_down']
    activation_derating_factor_up  = bpp['activation_derating_factor_up']

    
    equivalent_bid_price_up = bid_price_capacity_up + activation_derating_factor_up*bid_price_energy_up
    equivalent_bid_price_down = bid_price_capacity_down - activation_derating_factor_down*bid_price_energy_down
 
    # check for acceptance
    accept_capacity_down = equivalent_bid_price_down < equivalent_cleared_price_down
    accept_capacity_up   = equivalent_bid_price_up < equivalent_cleared_price_up
    accept_energy_down   = (equivalent_bid_price_down < equivalent_cleared_price_down) & (bid_price_energy_down < cleared_price_energy_down)
    accept_energy_up     = (equivalent_bid_price_up < equivalent_cleared_price_up) & (bid_price_energy_up < cleared_price_energy_up)

    return accept_capacity_down, accept_energy_down, accept_capacity_up, accept_energy_up

def calculate_revenue(df, bpp):
    """
    Calculate financial benefits of e-boiler operation and balancing market participation.

    inputs:
    df: the dataframe with the data. As implemented this will be data for one full day

    outputs:
    df: the dataframe with the data for same moments (rows) as the input, updated with the calculated revenues
    """
    df2 = df.copy()

    Palt = bpp['price_alternative_energy']
    # Calculate bid acceptance for each hour in the dataframe of the day
    accept_capacity_down, accept_energy_down, accept_capacity_up, accept_energy_up = calculate_bid_acceptance(df, bpp)

    # calculate value in dayahead market for all active hours, including the ones where we bid in the balancing market
    df2['V_dayahead'] = df['boiler_active_dayahead'] * (Palt - df['Day-ahead Energy Price'])
    for idx in df.index:
        # removed: we do not calculate valiue based on dayahead price if our bid on the market is accepted; we do not get the dayahead price then
        # if df.loc[idx, 'boiler_active_dayahead'] == 1: # if boiler activated
        #     df.loc[idx, 'V_dayahead'] = Palt - df.loc[idx, 'Day-ahead Energy Price']
        # if we bid in UP direction

        if df.loc[idx, 'Bid_UP'] == 1:

            df2.at[idx, 'V_bal_up_capacity'] += df.at[idx,'up_capacity_price'] * accept_capacity_up[idx].astype(float)
            df2.loc[idx, 'V_bal_up_energy'] += df.loc[idx,'up_energy_price'] * accept_energy_up[idx].astype(float)
            df2.loc[idx, 'V_energyvalue_balancing_up'] += -Palt

        elif df.loc[idx, 'Bid_DOWN'] == 1:
            df2.loc[idx, 'V_bal_down_energy'] = df.loc[idx, 'down_energy_price'] * df.loc[idx, 'bid_acceptance_probability']
            df2.loc[idx, 'V_bal_down_capacity'] += df.loc[idx, 'down_capacity_price'] - df.loc[idx, 'down_energy_price']
            df2.loc[idx, 'V_energyvalue_balancing_down'] += Palt

    return df2

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

def main(bpp, dayahead_file='dayahead_prices.csv', balancing_file='Finland - mFRR 2024 - Export for bc model.csv', alternative_energy_price=50, max_hours_per_month=None, output_file='results_eboiler.csv'):
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
    df_bal = convert_to_float_and_fill(df_bal, ['up_energy_price', 'down_energy_price','up_equivalent_price', 'down_equivalent_price','bid_acceptance_probability'])
    
    # Merge dataframes on overlapping timeframes
    df = merge_overlapping_timeframes(df_bal, df_da)

    # Assuming df is already loaded and 'utc_timestamp' is a datetime object
    # Set Day 0 based on the first complete day
    first_complete_day = df[df['utc_timestamp'].dt.hour == 0].iloc[0]['utc_timestamp'].date()

    # add output columns to fill
    df = initialize_output_columns(df)
    
    # Iterate over slices of the dataframe corresponding to full days
    processed_days = []
    for day_and_time in pd.date_range(start=first_complete_day, end=df['utc_timestamp'].dt.date.max(), freq='D'):
        day = day_and_time.date() 
        day_slice = df[df['utc_timestamp'].dt.date == day]
       
        # Check if the slice contains 24 rows (one for each hour)
        if len(day_slice) == 24:
            # Call functions with the selected day slice
            day_slice = schedule_day_ahead(day_slice, max_hours_per_month=max_hours_per_month, Palt=alternative_energy_price)
            day_slice = bid_balancing_market(day_slice)
            day_slice = calculate_revenue(day_slice, bpp)
            
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
    # set bid price parameters (float)
    bid_price_parameters = {
    'bid_price_energy_up':              50. ,
    'bid_price_energy_down':            8. ,
    'bid_price_capacity_up':            4. ,
    'bid_price_capacity_down':          10. ,
    'activation_derating_factor_down':  0.2 ,
    'activation_derating_factor_up':    0.3 ,
    'price_alternative_energy':         50.
    }
    # set maximum operational hours for each month
    max_hours_per_month = {1: 24, 2: 24, 3: 16, 4: 16, 5: 8, 6: 2, 7: 2, 8: 2, 9: 8, 10: 16, 11: 16, 12: 24}
    # optional: set all months to 24 by uncommenting next line
    # max_hours_per_month = {i: 24 for i in range(1, 13)}

    # Run the main function
    df_results = main(bid_price_parameters, dayahead_file='dayahead_prices_finland_2024.csv', balancing_file='Finland - mFRR 2024_CAP+ENE_new.xlsx - export_for_optimize_model.csv', max_hours_per_month=max_hours_per_month)
    df_results.to_csv('spoton_model_results.csv')
