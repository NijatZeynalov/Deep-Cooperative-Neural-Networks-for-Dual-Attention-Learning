import pandas as pd
from sklearn.model_selection import train_test_split
from data_loader import load_data


def preprocess_data(df):
    """
    Preprocesses the raw data by cleaning and splitting into training, validation, and test sets.

    Parameters:
    - df (pd.DataFrame): DataFrame containing user reviews, item descriptions, and ratings.

    Returns:
    - train_data (pd.DataFrame): DataFrame containing the training set.
    - val_data (pd.DataFrame): DataFrame containing the validation set.
    - test_data (pd.DataFrame): DataFrame containing the test set.
    """

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Optional: Add more text preprocessing here (e.g., removing special characters, lowercasing)
    df['user_reviews'] = df['reviewText'].str.lower()
    df['item_descriptions'] = df['item_description'].str.lower()

    # Normalize ratings to be between 0 and 1
    df['ratings'] = df['overall'] / df['overall'].max()

    # Select the relevant columns
    df = df[['user_reviews', 'item_descriptions', 'ratings']]

    # Split the data into training, validation, and test sets
    train_data, temp_data = train_test_split(df, test_size=0.4, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    return train_data, val_data, test_data


def save_processed_data(train_data, val_data, test_data, train_path, val_path, test_path):
    """
    Save the preprocessed data to CSV files.

    Parameters:
    - train_data (pd.DataFrame): DataFrame containing the training set.
    - val_data (pd.DataFrame): DataFrame containing the validation set.
    - test_data (pd.DataFrame): DataFrame containing the test set.
    - train_path (str): File path to save the training data.
    - val_path (str): File path to save the validation data.
    - test_path (str): File path to save the test data.
    """
    train_data.to_csv(train_path, index=False)
    val_data.to_csv(val_path, index=False)
    test_data.to_csv(test_path, index=False)


if __name__ == "__main__":
    # Example usage
    file_path = 'data/raw/amazon_reviews_sample.json'
    df = load_data(file_path)

    train_data, val_data, test_data = preprocess_data(df)

    save_processed_data(train_data, val_data, test_data,
                        'data/processed/train.csv',
                        'data/processed/val.csv',
                        'data/processed/test.csv')

    print("Data preprocessing complete. Processed files saved.")