import pandas as pd
import numpy as np

def calculate_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame): DataFrame containing coordinates (latitude and longitude).

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Assuming df has 'latitude' and 'longitude' columns for calculating distances
    distances = np.zeros((len(df), len(df)))

    for i in range(len(df)):
        for j in range(len(df)):
            if i != j:
                lat1, lon1 = df.iloc[i]['latitude'], df.iloc[i]['longitude']
                lat2, lon2 = df.iloc[j]['latitude'], df.iloc[j]['longitude']
                # Calculate Haversine distance (or other distance metrics as needed)
                distance = haversine(lat1, lon1, lat2, lon2)  # Placeholder for distance calculation
                distances[i][j] = distance

    return pd.DataFrame(distances, columns=df['id'], index=df['id'])


def unroll_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame): Distance matrix.

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    unrolled = []

    for i in range(len(df)):
        for j in range(len(df)):
            if i != j:  # Exclude self-distance
                unrolled.append({
                    'id_start': df.columns[i],
                    'id_end': df.columns[j],
                    'distance': df.iloc[i, j]
                })

    return pd.DataFrame(unrolled)


def find_ids_within_ten_percentage_threshold(df: pd.DataFrame, reference_id: int) -> pd.DataFrame:
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame): Unrolled DataFrame with distances.
        reference_id (int): ID to compare against.

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Calculate average distances for each ID
    avg_distances = df.groupby('id_start')['distance'].mean()
    
    if reference_id not in avg_distances:
        return pd.DataFrame()  # Return empty if reference_id not found

    reference_distance = avg_distances[reference_id]
    lower_bound = reference_distance * 0.9
    upper_bound = reference_distance * 1.1

    # Find IDs within the specified threshold
    filtered_ids = avg_distances[(avg_distances >= lower_bound) & (avg_distances <= upper_bound)].index
    return pd.DataFrame({'id': filtered_ids, 'average_distance': avg_distances[filtered_ids]})


def calculate_toll_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame): Unrolled DataFrame with distances.

    Returns:
        pandas.DataFrame: DataFrame with toll rates.
    """
    # Assuming a simplified toll calculation based on distance
    rate_per_km = 0.1  # Example rate
    df['toll_rate'] = df['distance'] * rate_per_km
    return df[['id_start', 'id_end', 'toll_rate']]


def calculate_time_based_toll_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame): DataFrame containing time of day.

    Returns:
        pandas.DataFrame: DataFrame with time-based toll rates.
    """
    # Example time-based rate logic
    def get_toll_rate(row):
        hour = row['time'].hour
        if 6 <= hour < 9:  # Peak hours
            return 1.5  # Higher toll during peak hours
        elif 9 <= hour < 18:  # Off-peak hours
            return 1.0  # Normal toll
        else:  # Late night
            return 0.5  # Reduced toll

    df['toll_rate'] = df.apply(get_toll_rate, axis=1)
    return df[['id_start', 'id_end', 'toll_rate']]


# Q9 ANS 
import pandas as pd

def calculate_distance_matrix(file_path: str) -> pd.DataFrame:
    """
    Calculate a distance matrix based on the dataset in the provided CSV file.

    Args:
        file_path (str): Path to the dataset-2.csv.

    Returns:
        pandas.DataFrame: Symmetric distance matrix with cumulative distances.
    """
    # Load the dataset
    df = pd.read_csv(file_path)

    # Create a list of unique IDs
    unique_ids = pd.concat([df['id_start'], df['id_end']]).unique()

    # Initialize the distance matrix with infinity
    distance_matrix = pd.DataFrame(float('inf'), index=unique_ids, columns=unique_ids)

    # Set the diagonal values to 0
    for id in unique_ids:
        distance_matrix.loc[id, id] = 0

    # Fill the distance matrix with direct distances
    for _, row in df.iterrows():
        start_id = row['id_start']
        end_id = row['id_end']
        distance = row['distance']

        distance_matrix.loc[start_id, end_id] = distance
        distance_matrix.loc[end_id, start_id] = distance  # Ensure symmetry

    # Calculate cumulative distances
    for k in unique_ids:
        for i in unique_ids:
            for j in unique_ids:
                if distance_matrix.loc[i, j] > distance_matrix.loc[i, k] + distance_matrix.loc[k, j]:
                    distance_matrix.loc[i, j] = distance_matrix.loc[i, k] + distance_matrix.loc[k, j]

    return distance_matrix

# Usage Example
# distance_matrix = calculate_distance_matrix('path/to/dataset-2.csv')
# print(distance_matrix)


# Q10 ANS 

import pandas as pd

def unroll_distance_matrix(distance_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        distance_matrix (pandas.DataFrame): The input distance matrix.

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Create an empty list to hold the rows
    unrolled_data = []

    # Iterate over the index (id_start) and columns (id_end) of the distance matrix
    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            # Exclude the pairs where id_start is the same as id_end
            if id_start != id_end:
                distance = distance_matrix.loc[id_start, id_end]
                # Append the data to the list
                unrolled_data.append({
                    'id_start': id_start,
                    'id_end': id_end,
                    'distance': distance
                })

    # Convert the list of dictionaries to a DataFrame
    unrolled_df = pd.DataFrame(unrolled_data)

    return unrolled_df

# Example usage
# Assuming distance_matrix is the DataFrame obtained from calculate_distance_matrix function
# unrolled_df = unroll_distance_matrix(distance_matrix)
# print(unrolled_df)
# Calculate the distance matrix first
distance_matrix = calculate_distance_matrix('path/to/dataset-2.csv')

# Then unroll it
unrolled_df = unroll_distance_matrix(distance_matrix)

# Display the unrolled DataFrame
print(unrolled_df)

# Q11 ANS 

import pandas as pd

def find_ids_within_ten_percentage_threshold(unrolled_df: pd.DataFrame, reference_id: str) -> pd.DataFrame:
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        unrolled_df (pandas.DataFrame): The unrolled distance DataFrame.
        reference_id (str): The ID for which the average distance will be calculated.

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Calculate the average distance for the reference ID
    reference_avg_distance = unrolled_df[unrolled_df['id_start'] == reference_id]['distance'].mean()

    # Calculate the lower and upper bounds for the threshold
    lower_bound = reference_avg_distance * 0.9
    upper_bound = reference_avg_distance * 1.1

    # Find IDs within the threshold
    ids_within_threshold = (
        unrolled_df.groupby('id_start')['distance']
        .mean()
        .reset_index()
    )

    # Filter based on the calculated bounds
    filtered_ids = ids_within_threshold[
        (ids_within_threshold['distance'] >= lower_bound) & 
        (ids_within_threshold['distance'] <= upper_bound)
    ]

    # Return the sorted DataFrame with the filtered IDs
    return filtered_ids.sort_values(by='id_start')

# Example usage
# Assuming unrolled_df is the DataFrame obtained from the unroll_distance_matrix function
# reference_id = 'A'
# result_df = find_ids_within_ten_percentage_threshold(unrolled_df, reference_id)
# print(result_df)
# Calculate the unrolled DataFrame first
unrolled_df = unroll_distance_matrix(distance_matrix)

# Define the reference ID
reference_id = 'A'

# Find IDs within the 10% threshold
result_df = find_ids_within_ten_percentage_threshold(unrolled_df, reference_id)

# Display the result DataFrame
print(result_df)

# Q 12 ANS 

import pandas as pd

def calculate_toll_rate(unrolled_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        unrolled_df (pandas.DataFrame): The unrolled distance DataFrame.

    Returns:
        pandas.DataFrame: The updated DataFrame with toll rate columns.
    """
    # Define the rate coefficients for each vehicle type
    rates = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    # Calculate toll rates for each vehicle type and add them to the DataFrame
    for vehicle_type, rate in rates.items():
        unrolled_df[vehicle_type] = unrolled_df['distance'] * rate

    return unrolled_df

# Example usage
# Assuming unrolled_df is the DataFrame obtained from the unroll_distance_matrix function
# result_df = calculate_toll_rate(unrolled_df)
# print(result_df)
# Calculate the unrolled DataFrame first
unrolled_df = unroll_distance_matrix(distance_matrix)

# Calculate the toll rates
result_df = calculate_toll_rate(unrolled_df)

# Display the result DataFrame
print(result_df)


# Q13 ANS 

import pandas as pd
from datetime import time

def calculate_time_based_toll_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame): The DataFrame with initial toll rates.

    Returns:
        pandas.DataFrame: The updated DataFrame with time-based toll rates.
    """
    # Define day names and initialize new columns
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df['start_day'] = ''
    df['end_day'] = ''
    df['start_time'] = time(0, 0)  # default to 00:00:00
    df['end_time'] = time(23, 59, 59)  # default to 23:59:59

    # Define the discount factors based on the time ranges
    for index, row in df.iterrows():
        # Assign start_day and end_day based on id_start and id_end for simplicity
        row['start_day'] = days[(index % 7)]  # Cycling through the days
        row['end_day'] = days[(index % 7)]  # Same day for simplicity in this example
        
        # Get start and end times
        start_time = row['start_time']
        end_time = row['end_time']
        
        # Determine the discount factor based on the day and time
        if row['start_day'] in days[:5]:  # Weekday: Monday to Friday
            if time(0, 0) <= start_time < time(10, 0):
                discount_factor = 0.8
            elif time(10, 0) <= start_time < time(18, 0):
                discount_factor = 1.2
            elif time(18, 0) <= start_time < time(23, 59, 59):
                discount_factor = 0.8
            else:
                discount_factor = 1.0  # Should not happen
        else:  # Weekend: Saturday and Sunday
            discount_factor = 0.7
        
        # Apply the discount factor to each vehicle type
        for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
            row[vehicle] *= discount_factor
            
        # Update the row in the DataFrame
        df.loc[index] = row

    return df

# Example usage
# Assuming df is the DataFrame obtained from the calculate_toll_rate function
# result_df = calculate_time_based_toll_rates(df)
# print(result_df)

# Calculate the time-based toll rates
result_df = calculate_time_based_toll_rates(df)

# Display the result DataFrame
print(result_df)
