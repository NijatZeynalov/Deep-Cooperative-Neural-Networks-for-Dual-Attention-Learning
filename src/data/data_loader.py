import pandas as pd
import json


def load_data(file_path):
    """
    Load data from a JSON file and preprocess it into lists of user reviews, item descriptions, and ratings.

    Parameters:
    - file_path (str): Path to the JSON file containing the data.

    Returns:
    - df (pd.DataFrame): DataFrame containing user reviews, item descriptions, and ratings.
    """
    # Load the JSON data
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Convert the JSON data to a DataFrame
    df = pd.DataFrame(data)

    return df

