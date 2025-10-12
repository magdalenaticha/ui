import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from lifelines import KaplanMeierFitter
import numpy as np

# Load data from CSV file
@st.cache_data
def load_data():
    try:
        # Adjust the file path as needed
        df = pd.read_csv('clinical_data.csv')  # or whatever your file is named
        
        # Convert boolean columns to integers (1 for True, 0 for False)
        bool_columns = ['os.event', 'rfs.event']
        for col in bool_columns:
            if col in df.columns:
                # Convert to integer: True -> 1, False -> 0
                df[col] = df[col].astype(int)
        
        return df
    except FileNotFoundError:
        st.error("CSV file not found. Please make sure 'clinical_data.csv' is in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def plot_survival_analysis(df, biomarker, survival_type='os'):
    """
    Create Kaplan-Meier survival curves
    """
    try:
        kmf = KaplanMeierFitter()
        fig = go.Figure()
        
        # Get unique values for the biomarker
        unique_values = df[biomarker].dropna().unique()
        
        for value in unique_values:
            mask = df[biomarker] == value
            time_col = f'{survival_type}.time'
            event_col = f'{survival_type}.event'
            
            if time_col in df.columns and event_col in df.columns:
                # Filter data for this biomarker value
                subset = df[mask]
                
                # Check if we have any events for this group
                if len(subset) > 0 and subset[event_col].sum() > 0:
                    kmf.fit(subset[time_col], 
                           subset[event_col], 
                           label=f"{biomarker} = {value}")
                    
                    # Plot survival function
                    fig.add_trace(go.Scatter(
                        x=kmf.survival_function_.index,
                        y=kmf.survival_function_.iloc[:, 0],  # Use iloc instead of column name
                        mode='lines',
                        name=f"{biomarker} = {value}",
                        line=dict(width=3)
                    ))
                else:
                    st.warning(f"No events found for {biomarker} = {value}")
        
        survival_title = 'Overall Survival' if survival_type == 'os' else 'Recurrence-Free Survival'
        fig.update_layout(
            title=f'{survival_title} by {biomarker}',
            xaxis_title='Time (Months)',
            yaxis_title='Survival Probability',
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating survival plot: {e}")
        # Return empty figure instead of crashing
        return go.Figure()

def plot_biomarker_distribution(df, biomarker):
    """
    Simple distribution plot for biomarkers
    """
    try:
        fig = px.histogram(df, x=biomarker, 
                          title=f"Distribution of {biomarker}",
                          color=biomarker)
        return fig
    except Exception as e:
        st.error(f"Error creating distribution plot: {e}")
        return px.histogram()

def plot_treatment_response(df, biomarker):
    """
    Compare treatment response across biomarker groups
    """
    try:
        # Calculate average survival times by biomarker and treatment
        response_data = df.groupby([biomarker, 'trtgrp']).agg({
            'os.time': 'mean',
            'rfs.time': 'mean',
            'os.event': 'mean'
        }).reset_index()
        
        fig = px.bar(response_data, 
                    x=biomarker, 
                    y='os.time', 
                    color='trtgrp', 
                    barmode='group',
                    title=f"Average Overall Survival by {biomarker} and Treatment",
                    labels={'os.time': 'Mean Survival Time (Months)'})
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating treatment response plot: {e}")
        return px.bar()

def main():
    st.set_page_config(page_title="Cancer Biomarker Explorer", layout="wide")
    
    st.title("Colorectar Cancer Biomarker Explorer")
    st.markdown("Explore predictive and prognostic biomarkers for treatment response and survival")
    
    # Load data
    df = load_data()
    
    if df is None:
        st.stop()
    
    # Sidebar controls
    st.sidebar.header("🔬 Analysis Parameters")
    
    # Create a mapping from display names to column names
    biomarker_options = {
        'BRAF': 'BRAF',
        'KRAS': 'KRAS', 
        'MSI': 'MSI',
        'Tumor Grade': 'grade',  # Changed display name to "Tumor Grade"
        'Tumor Site': 'site',
        'Stage': 'stage',
        'Treatment Group': 'trtgrp'
    }
    
    # Get available biomarkers (only include those that exist in the data)
    available_biomarkers = {display_name: col_name 
                          for display_name, col_name in biomarker_options.items() 
                          if col_name in df.columns}
    
    if not available_biomarkers:
        st.error("No biomarker columns found in the data. Available columns: " + ", ".join(df.columns))
        st.stop()
    
    biomarker_display = st.sidebar.selectbox(
        "Select Biomarker",
        options=list(available_biomarkers.keys()),
        index=0
    )
    
    # Get the actual column name for the selected biomarker
    biomarker_column = available_biomarkers[biomarker_display]
    
    analysis_type = st.sidebar.radio(
        "Analysis Type",
        ["Survival Analysis", "Biomarker Distribution", "Treatment Response"]
    )
    
    # Use full dataset (no filtering)
    df_filtered = df
    
    # Survival type selection for survival analysis
    if analysis_type == "Survival Analysis":
        survival_type = st.sidebar.radio(
            "Survival Type",
            ["Overall Survival", "Recurrence-Free Survival"]
        )
        survival_code = 'os' if survival_type == "Overall Survival" else 'rfs'
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Analysis Results")
        
        # Display appropriate visualization
        if analysis_type == "Survival Analysis":
            fig = plot_survival_analysis(df_filtered, biomarker_column, survival_code)
        elif analysis_type == "Biomarker Distribution":
            fig = plot_biomarker_distribution(df_filtered, biomarker_column)
        else:
            fig = plot_treatment_response(df_filtered, biomarker_column)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Summary Statistics")
        
        # Basic stats
        total_patients = len(df_filtered)
        st.metric("Total Patients", total_patients)
        
        # Biomarker distribution
        if biomarker_column in df_filtered.columns:
            biomarker_counts = df_filtered[biomarker_column].value_counts()
            st.write(f"**{biomarker_display} Distribution:**")
            for value, count in biomarker_counts.items():
                percentage = (count / total_patients) * 100
                st.write(f"- {value}: {count} ({percentage:.1f}%)")
        
        # Survival statistics
        if 'os.time' in df_filtered.columns:
            avg_survival = df_filtered['os.time'].mean()
            st.metric("Average OS (months)", f"{avg_survival:.1f}")
        
        if 'rfs.time' in df_filtered.columns:
            avg_rfs = df_filtered['rfs.time'].mean()
            st.metric("Average RFS (months)", f"{avg_rfs:.1f}")
        
        # Event rates
        if 'os.event' in df_filtered.columns:
            event_rate = df_filtered['os.event'].mean()
            st.metric("OS Event Rate", f"{event_rate:.1%}")

if __name__ == "__main__":
    main()