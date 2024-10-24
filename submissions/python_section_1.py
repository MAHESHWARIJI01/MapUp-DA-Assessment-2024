from typing import Dict, List
import pandas as pd


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    
    Args:
    - lst: List of integers to reverse
    - n: Number of elements in each group
    
    Returns:
    - A new list with elements reversed in groups of n.
    """
    result = []
    for i in range(0, len(lst), n):
        result.extend(lst[i:i+n][::-1])
    return result


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    
    Args:
    - lst: List of strings to group
    
    Returns:
    - Dictionary where the key is the length of the string, and the value is a list of strings with that length.
    """
    length_dict = {}
    for s in lst:
        length_dict.setdefault(len(s), []).append(s)
    return length_dict


def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    Args:
    - nested_dict: The dictionary object to flatten
    - sep: The separator to use between parent and child keys (defaults to '.')
    
    Returns:
    - A flattened dictionary
    """
    def _flatten(d, parent_key=''):
        items = {}
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, dict):
                items.update(_flatten(v, new_key))
            else:
                items[new_key] = v
        return items
    
    return _flatten(nested_dict)


def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    Args:
    - nums: List of integers (may contain duplicates)
    
    Returns:
    - List of unique permutations
    """
    from itertools import permutations
    return list(map(list, set(permutations(nums))))


def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    import re
    date_patterns = [
        r'\b\d{2}-\d{2}-\d{4}\b',  # dd-mm-yyyy
        r'\b\d{2}/\d{2}/\d{4}\b',  # mm/dd/yyyy
        r'\b\d{4}\.\d{2}\.\d{2}\b'  # yyyy.mm.dd
    ]
    dates = []
    for pattern in date_patterns:
        dates.extend(re.findall(pattern, text))
    return dates


def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
    - polyline_str (str): The encoded polyline string.
    
    Returns:
    - pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    # Placeholder, decoding polyline requires a third-party library like polyline
    data = {
        "latitude": [],
        "longitude": [],
        "distance": []
    }
    return pd.DataFrame(data)


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    n = len(matrix)
    # Rotate 90 degrees clockwise
    rotated = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]
    # Multiply each element by the sum of its original row and column index
    for i in range(n):
        for j in range(n):
            rotated[i][j] *= (i + j)
    return rotated


def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7-day period.

    Args:
    - df (pandas.DataFrame): DataFrame with columns `id`, `id_2`, and `timestamp`.

    Returns:
    - pd.Series: return a boolean series indicating whether each pair covers the required time span.
    """
    # Placeholder logic for example purposes
    result = df.groupby(['id', 'id_2']).apply(
        lambda group: (group['timestamp'].max() - group['timestamp'].min()).days >= 7
    )
    return result


#q1 ans

from typing import List

def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements without using slicing or reverse functions.
    
    Args:
    - lst: List of integers to reverse
    - n: Number of elements in each group
    
    Returns:
    - A new list with elements reversed in groups of n.
    """
    result = []
    length = len(lst)
    
    # Loop through the list in steps of n
    for i in range(0, length, n):
        group = []
        
        # Reverse elements manually within the current group of size n
        for j in range(min(n, length - i)):  # Handle case when fewer than n elements are left
            group.append(lst[i + j])
        
        # Manually reverse the group and add to the result
        for j in range(len(group) - 1, -1, -1):
            result.append(group[j])
    
    return result

# Test cases
print(reverse_by_n_elements([1, 2, 3, 4, 5, 6, 7, 8], 3))  # Output: [3, 2, 1, 6, 5, 4, 8, 7]
print(reverse_by_n_elements([1, 2, 3, 4, 5], 2))  # Output: [2, 1, 4, 3, 5]
print(reverse_by_n_elements([10, 20, 30, 40, 50, 60, 70], 4))  # Output: [40, 30, 20, 10, 70, 60, 50]

# q2 ans 

from typing import List, Dict

def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary sorted by the length of strings.
    
    Args:
    - lst: List of strings to group
    
    Returns:
    - Dictionary where the key is the length of the string, and the value is a list of strings with that length.
    """
    length_dict = {}
    
    # Group strings by their length
    for s in lst:
        length = len(s)
        if length not in length_dict:
            length_dict[length] = []
        length_dict[length].append(s)
    
    # Sort dictionary by keys (length)
    sorted_length_dict = dict(sorted(length_dict.items()))
    
    return sorted_length_dict

# Test cases
print(group_by_length(["apple", "bat", "car", "elephant", "dog", "bear"]))
# Output: {3: ['bat', 'car', 'dog'], 4: ['bear'], 5: ['apple'], 8: ['elephant']}

print(group_by_length(["one", "two", "three", "four"]))
# Output: {3: ['one', 'two'], 4: ['four'], 5: ['three']}

#q3 ans

from typing import Dict, Any

def flatten_dict(nested_dict: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    List elements are referenced by their index, enclosed in square brackets.
    
    Args:
    - nested_dict: The dictionary object to flatten
    - parent_key: A string representing the base key for recursion
    - sep: The separator to use between parent and child keys (defaults to '.')
    
    Returns:
    - A flattened dictionary
    """
    items = {}

    for k, v in nested_dict.items():
        # Create new key by concatenating the parent key and the current key
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            # If the value is a dictionary, recursively flatten it
            items.update(flatten_dict(v, new_key, sep))
        elif isinstance(v, list):
            # If the value is a list, flatten each element by its index
            for i, item in enumerate(v):
                items.update(flatten_dict({f"{k}[{i}]": item}, parent_key, sep))
        else:
            # If it's neither a dict nor a list, add it as a flat entry
            items[new_key] = v
    
    return items

# Test case
nested_dict = {
    "road": {
        "name": "Highway 1",
        "length": 350,
        "sections": [
            {
                "id": 1,
                "condition": {
                    "pavement": "good",
                    "traffic": "moderate"
                }
            }
        ]
    }
}

flattened = flatten_dict(nested_dict)
print(flattened)

# q4 ans 

from typing import List

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    Args:
    - nums: List of integers (may contain duplicates)
    
    Returns:
    - A list of unique permutations (no duplicates).
    """
    def backtrack(path, used):
        # Base case: if the current path contains all elements, it's a valid permutation
        if len(path) == len(nums):
            result.append(path[:])  # Append a copy of the path to the result
            return
        
        # Iterate through the list of numbers
        for i in range(len(nums)):
            # Skip used numbers or duplicates (skip duplicates if the previous number is the same and hasn't been used)
            if used[i] or (i > 0 and nums[i] == nums[i - 1] and not used[i - 1]):
                continue
            
            # Mark the number as used and add it to the current path
            used[i] = True
            path.append(nums[i])
            
            # Continue exploring with the current number included
            backtrack(path, used)
            
            # Backtrack: remove the last element and mark the number as unused
            path.pop()
            used[i] = False
    
    # Sort the list to easily skip duplicates
    nums.sort()
    
    result = []
    used = [False] * len(nums)  # Keep track of which elements are used in the current permutation
    backtrack([], used)  # Start the backtracking process
    
    return result

# Test case
print(unique_permutations([1, 1, 2]))
# Output: [[1, 1, 2], [1, 2, 1], [2, 1, 1]]

# q5 ans 
import re
from typing import List

def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Args:
    - text: A string containing dates in various formats
    
    Returns:
    - A list of valid dates found in the string
    """
    # Regular expression patterns for the 3 date formats
    date_patterns = [
        r'\b\d{2}-\d{2}-\d{4}\b',  # Matches dd-mm-yyyy
        r'\b\d{2}/\d{2}/\d{4}\b',  # Matches mm/dd/yyyy
        r'\b\d{4}\.\d{2}\.\d{2}\b' # Matches yyyy.mm.dd
    ]
    
    # Combine all patterns into one
    combined_pattern = '|'.join(date_patterns)
    
    # Find all matches in the text
    matches = re.findall(combined_pattern, text)
    
    return matches

# Test case
text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
print(find_all_dates(text))
# Output: ["23-08-1994", "08/23/1994", "1994.08.23"]

#q6 ans

import polyline  # To decode polyline strings
import pandas as pd
import math

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two points on the Earth (in meters).
    The Haversine formula calculates the shortest distance over the earth's surface.
    """
    R = 6371000  # Radius of the Earth in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c  # Distance in meters

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Decodes a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str: The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance (in meters).
    """
    # Decode polyline into a list of (latitude, longitude) pairs
    coordinates = polyline.decode(polyline_str)
    
    # Create a DataFrame with latitude and longitude columns
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    
    # Initialize the distance column with 0 for the first point
    df['distance'] = 0.0
    
    # Calculate the Haversine distance for successive points
    for i in range(1, len(df)):
        lat1, lon1 = df.at[i-1, 'latitude'], df.at[i-1, 'longitude']
        lat2, lon2 = df.at[i, 'latitude'], df.at[i, 'longitude']
        df.at[i, 'distance'] = haversine(lat1, lon1, lat2, lon2)
    
    return df

# Test case (encoded polyline string)
polyline_str = "u{~vFvyys@fBlA"  # Example polyline string
df = polyline_to_dataframe(polyline_str)
print(df)

# q7 ans 
from typing import List

def rotate_matrix_90_degrees(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise.
    
    Args:
    - matrix: A square matrix (n x n)
    
    Returns:
    - A new matrix rotated by 90 degrees clockwise
    """
    n = len(matrix)
    rotated_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - 1 - i] = matrix[i][j]
    
    return rotated_matrix

def transform_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    After rotating the matrix by 90 degrees, for each element in the rotated matrix,
    replace it with the sum of all elements in the same row and column, excluding itself.
    
    Args:
    - matrix: A square matrix (n x n)
    
    Returns:
    - The transformed matrix
    """
    # Rotate the matrix by 90 degrees
    rotated_matrix = rotate_matrix_90_degrees(matrix)
    n = len(rotated_matrix)
    
    # Create the final transformed matrix
    transformed_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            # Calculate the sum of all elements in the same row
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]
            # Calculate the sum of all elements in the same column
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) - rotated_matrix[i][j]
            # Set the current element as the sum of row_sum and col_sum
            transformed_matrix[i][j] = row_sum + col_sum
    
    return transformed_matrix

# Example usage
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

final_matrix = transform_matrix(matrix)
for row in final_matrix:
    print(row)

# Output:
# [22, 19, 16]
# [23, 20, 17]
# [24, 21, 18]

# q8 ans 

import pandas as pd

def time_check(df: pd.DataFrame) -> pd.Series:
    """
    Verify completeness of the time data for each unique (id, id_2) pair, ensuring:
    1. The timestamps cover a full 24-hour period (12:00:00 AM to 11:59:59 PM).
    2. The timestamps span all 7 days of the week (Monday to Sunday).
    
    Args:
    - df (pd.DataFrame): The dataset containing columns id, id_2, startDay, startTime, endDay, endTime.
    
    Returns:
    - pd.Series: A boolean series indicating whether each (id, id_2) pair has complete timestamps. 
                 The series should have a multi-index (id, id_2).
    """
    # Convert the DataFrame to ensure timestamp columns are in datetime format
    df['startTime'] = pd.to_datetime(df['startTime'], format='%H:%M:%S').dt.time
    df['endTime'] = pd.to_datetime(df['endTime'], format='%H:%M:%S').dt.time

    # Initialize an empty dictionary to store completeness for each (id, id_2) pair
    completeness_check = {}

    # Group by id and id_2
    grouped = df.groupby(['id', 'id_2'])

    for (id_val, id_2_val), group in grouped:
        # Check if all days of the week are covered (Monday=0, Sunday=6)
        days_covered = set(group['startDay'].unique())
        full_week = set(range(7))  # Set of all days from 0 (Monday) to 6 (Sunday)
        
        # Ensure that both start and end times cover a full 24-hour period for each day
        full_day_coverage = True
        for day in full_week:
            day_records = group[group['startDay'] == day]
            if not day_records.empty:
                start_times = day_records['startTime'].min()
                end_times = day_records['endTime'].max()
                if start_times != pd.Timestamp('00:00:00').time() or end_times != pd.Timestamp('23:59:59').time():
                    full_day_coverage = False
                    break
        
        # Both conditions (full week and full day) should be met
        completeness_check[(id_val, id_2_val)] = (days_covered == full_week) and full_day_coverage
  
    # Convert the dictionary to a Pandas Series with a MultiIndex (id, id_2)
    completeness_series = pd.Series(completeness_check)

    return completeness_series

# Example usage
# Assuming dataset-1.csv has been loaded into a DataFrame df
# df = pd.read_csv('dataset-1.csv')
# result = time_check(df)
# print(result)

