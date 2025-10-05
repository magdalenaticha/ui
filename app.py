import streamlit as st
import pandas as pd
from analysis_engine import explore_h5_file, load_h5_data

st.title("🔬 Gene Data Explorer")
st.write("Exploring the HDF5 data file...")

# Update this path to match your actual file location
H5_FILE_PATH = "./gene_expression.h5"  # Change this!

try:
    # Explore the file structure
    explore_h5_file(H5_FILE_PATH)
    
    st.divider()
    
    # Load the actual data
    st.header("📈 Loaded Data")
    data_dict = load_h5_data(H5_FILE_PATH)
    
    # Display each loaded dataset
    for key, data in data_dict.items():
        st.subheader(f"Dataset: {key}")
        
        if isinstance(data, pd.DataFrame):
            st.write(f"Shape: {data.shape}")
            st.dataframe(data.head())
            
            # Show basic stats
            st.write("Basic statistics:")
            st.dataframe(data.describe())
            
        else:
            st.write(f"Data type: {type(data)}, Shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")
            st.write(f"First 10 values: {data[:10] if hasattr(data, '__getitem__') else data}")

except FileNotFoundError:
    st.error(f"File not found at: {H5_FILE_PATH}")
    st.info("Please update the H5_FILE_PATH in app.py to point to your .h5 file")
    
except Exception as e:
    st.error(f"Error loading file: {e}")