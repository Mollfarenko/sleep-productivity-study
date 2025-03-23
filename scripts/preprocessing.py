import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import os

# Define file paths
RAW_DATA_PATH = "data/raw/sleep_cycle_productivity.csv"
PROCESSED_DATA_PATH = "data/processed/sleep_productivity_clean.csv"

def load_data(filepath):
    """Load raw dataset from CSV file."""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Preprocess dataset: handle categorical variables, date features"""

    # Convert 'Date' column to datetime and extract useful features
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df.drop(columns=["Date"], inplace=True)  # Drop original Date column

    # One-Hot Encode 'Gender' column
    encoder = OneHotEncoder(drop="first", sparse_output=False)
    gender_encoded = encoder.fit_transform(df[["Gender"]])
    gender_df = pd.DataFrame(gender_encoded, columns=encoder.get_feature_names_out(["Gender"]))

    # Merge and drop original 'Gender' column
    df = pd.concat([df, gender_df], axis=1).drop(columns=["Gender"])

    # Drop Person ID
    df.drop(columns=["Person_ID"], inplace=True)

    return df

def save_data(df, output_path):
    """Save the processed dataset to a CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Processed data saved to {output_path}")

if __name__ == "__main__":
    # Load, preprocess, and save data
    df = load_data(RAW_DATA_PATH)
    df_clean = preprocess_data(df)
    save_data(df_clean, PROCESSED_DATA_PATH)
