# analysis_engine.py
import h5py
import pandas as pd
import streamlit as st

def explore_h5_file(file_path):
    """Explore the structure of the HDF5 file"""
    
    with h5py.File(file_path, 'r') as f:
        st.write("## HDF5 File Structure")
        
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                st.write(f"📊 Dataset: {name} | Shape: {obj.shape} | Type: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                st.write(f"📁 Group: {name}")
        
        f.visititems(print_structure)

def load_h5_data(file_path):
    """Load data from HDF5 file and convert to pandas DataFrame"""
    
    with h5py.File(file_path, 'r') as f:
        st.write("Available keys in the file:")
        for key in f.keys():
            st.write(f"- {key}")
        
        data_dict = {}
        
        for key in f.keys():
            try:
                if isinstance(f[key], h5py.Dataset):
                    data = f[key][:]
                    if len(data.shape) == 2:
                        data_dict[key] = pd.DataFrame(data)
                    else:
                        data_dict[key] = data
                elif isinstance(f[key], h5py.Group):
                    st.write(f"Group {key} contains: {list(f[key].keys())}")
            except Exception as e:
                st.write(f"Could not load {key}: {e}")
        
        return data_dict