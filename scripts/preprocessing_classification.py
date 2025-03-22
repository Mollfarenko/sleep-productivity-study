import pandas as pd
import os

# Define file paths
REGRESS_DATA_PATH = "data/processed/sleep_productivity_clean.csv"
CLASS_DATA_PATH = "data/processed/sleep_productivity_classification.csv"

def load_data(filepath):
    """Load preprocessed dataset from CSV file."""
    return pd.read_csv(filepath)

def preprocess_data(df):
    # Convert Productivity Score into binary classification target
    df["Productivity Class"] = (df["Productivity Score"] >= 7).astype(int)

    # Drop Productivity Score
    df.drop(columns=["Productivity Score"], inplace=True)

    return df

def save_data(df, output_path):
    """Save the processed dataset to a CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Processed data for classification saved to {output_path}")

if __name__ == "__main__":
    # Load, preprocess, and save data
    df = load_data(REGRESS_DATA_PATH)
    df_clean = preprocess_data(df)
    save_data(df_clean, CLASS_DATA_PATH)
