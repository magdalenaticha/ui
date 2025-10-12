import pandas as pd
import numpy as np
from typing import Dict

def load_clinical_data(csv_file_path: str) -> pd.DataFrame:
    """
    Load clinical data from CSV file and clean/standardize the data
    
    Args:
        csv_file_path: Path to the CSV file
        
    Returns:
        pandas DataFrame with properly structured clinical data
    """
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file_path)
        
        print(f"Original data shape: {df.shape}")
        print(f"Original columns: {list(df.columns)}")
        
        # Display the first few rows to understand structure
        print("\nFirst few rows of raw data:")
        print(df.head())
        
        return df
        
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return pd.DataFrame()

def clean_clinical_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and structure the clinical data based on the pattern we observed
    """
    if df.empty:
        return df
    
    # Create a clean copy
    clean_df = df.copy()
    
    # Based on your data structure, we need to parse the complex format
    # The data appears to be in a nested format within cells
    
    # Example cleaning steps (adjust based on actual CSV structure):
    
    # 1. Convert byte strings to regular strings if needed
    for col in clean_df.columns:
        if clean_df[col].dtype == 'object':
            # Clean byte strings like b'5-FU/FA'
            clean_df[col] = clean_df[col].apply(
                lambda x: x.decode('utf-8') if isinstance(x, bytes) else str(x)
            )
    
    # 2. Extract patient IDs and other features
    # This will depend on how the data is actually structured in the CSV
    
    print(f"Cleaned data shape: {clean_df.shape}")
    return clean_df

def get_clinical_data_summary(df: pd.DataFrame) -> Dict:
    """
    Generate summary statistics for the clinical dataset
    """
    if df.empty:
        return {}
    
    summary = {
        'total_patients': len(df),
        'columns': list(df.columns),
        'data_types': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'basic_stats': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
    }
    
    return summary

def explore_data_structure(df: pd.DataFrame):
    """
    Explore the structure of the data to understand how to parse it
    """
    print("\n=== DATA EXPLORATION ===")
    print(f"DataFrame shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nFirst 3 rows:")
    for i in range(min(3, len(df))):
        print(f"\nRow {i}:")
        for col in df.columns:
            print(f"  {col}: {df.iloc[i][col]} (type: {type(df.iloc[i][col])})")

# Main function to process the clinical data
def process_clinical_data(csv_file_path: str) -> tuple[pd.DataFrame, Dict]:
    """
    Main function to load and process clinical data
    
    Returns:
        tuple: (cleaned_dataframe, summary_statistics)
    """
    # Load raw data
    raw_df = load_clinical_data(csv_file_path)
    
    if raw_df.empty:
        print("No data loaded!")
        return pd.DataFrame(), {}
    
    # Explore the structure first
    explore_data_structure(raw_df)
    
    # Clean the data
    cleaned_df = clean_clinical_data(raw_df)
    
    # Get summary statistics
    summary = get_clinical_data_summary(cleaned_df)
    
    return cleaned_df, summary

# Example usage
if __name__ == "__main__":
    # Replace with your actual CSV file path
    csv_path = "./ui/h5otvoreny.csv"
    
    # Process the data
    clinical_df, summary = process_clinical_data(csv_path)
    
    if not clinical_df.empty:
        print(f"\n=== PROCESSING COMPLETE ===")
        print(f"Successfully processed data for {len(clinical_df)} patients")
        print(f"Available columns: {list(clinical_df.columns)}")
        
        # Save cleaned data
        clinical_df.to_csv('cleaned_clinical_data.csv', index=False)
        print("Cleaned data saved to 'cleaned_clinical_data.csv'")
        
        # Display summary
        print(f"\nDataset Summary:")
        print(f"Total patients: {summary['total_patients']}")
        print(f"Columns: {summary['columns']}")