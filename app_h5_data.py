import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import numpy as np
import h5py

@st.cache_data
def load_data():
    try:
        # csv - backup
        clinical_df = pd.read_csv('clinical_data.csv')
        expression_data = None
        gene_names = None
        
        try:
            with h5py.File('petacc3-student_project.h5', 'r') as f:
                if 'X' in f:
                    x_group = f['X']
                    
                    try:
                        axis0_data = x_group['axis0'][:]  # genes (13349)
                        axis1_data = x_group['axis1'][:]  # samples (688)
                        matrix_data = x_group['block0_values'][:]  # 688 x 13349 
                        
                        # convert bytes to strings 
                        axis0_data = [name.decode('utf-8') if isinstance(name, bytes) else name for name in axis0_data]
                        axis1_data = [name.decode('utf-8') if isinstance(name, bytes) else name for name in axis1_data]
                        
                        expression_data = pd.DataFrame(
                            data=matrix_data,
                            index=axis1_data,  # sample IDs as row index 
                            columns=axis0_data  # gene names as column names 
                        )
                        
                        gene_names = axis0_data
                        
                    except Exception as e:
                        pass

                clinical_from_h5 = None
                if 'clinical' in f:
                    clinical_group = f['clinical']
                    try:
                        if 'axis1' in clinical_group and 'block0_values' in clinical_group:
                            clinical_columns = [name.decode('utf-8') if isinstance(name, bytes) else name 
                                              for name in clinical_group['axis1'][:]]
                            if 'axis0' in clinical_group:
                                clinical_samples = [name.decode('utf-8') if isinstance(name, bytes) else name 
                                                  for name in clinical_group['axis0'][:]]
                            clinical_matrix = clinical_group['block0_values'][:]
                            clinical_from_h5 = pd.DataFrame(
                                clinical_matrix,
                                index=clinical_samples if 'axis0' in clinical_group else None,
                                columns=clinical_columns
                            )
                    except Exception as e:
                        pass
                
                if clinical_from_h5 is not None and len(clinical_from_h5) > 0:
                    clinical_df = clinical_from_h5
                    
        except FileNotFoundError:
            pass
        except Exception as e:
            pass
        
        clinical_df = clean_clinical_data(clinical_df)
        return clinical_df, expression_data, gene_names
        
    except FileNotFoundError:
        st.error("Clinical data file not found.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

def clean_clinical_data(df):
    """Clean and preprocess clinical data"""
    df_clean = df.copy()
  
    bool_columns = ['os.event', 'rfs.event', 'pfs.event']
    for col in bool_columns:
        if col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].map({'True': 1, 'False': 0, 'TRUE': 1, 'FALSE': 0}).fillna(df_clean[col])
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype(int)

    categorical_columns = ['trtgrp', 'grade', 'site', 'stage', 'BRAF', 'KRAS', 'MSI']
    for col in categorical_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype('category')
    
    return df_clean

def plot_survival_analysis(df, biomarker, survival_type='os'):
    """
    Create Kaplan-Meier survival curves with log-rank test
    """
    try:
        kmf = KaplanMeierFitter()
        fig = go.Figure()
        if pd.api.types.is_numeric_dtype(df[biomarker]) and df[biomarker].nunique() > 10:
            threshold = df[biomarker].median()
            df_analysis = df.copy()
            df_analysis[biomarker] = np.where(df_analysis[biomarker] > threshold, 'High', 'Low')
            unique_values = ['High', 'Low']
        else:
            df_analysis = df
            unique_values = df_analysis[biomarker].dropna().unique()

        valid_data = df_analysis[[biomarker, f'{survival_type}.time', f'{survival_type}.event']].dropna()
        
        if len(valid_data) == 0:
            st.warning(f"No valid data for survival analysis with {biomarker}")
            return go.Figure(), None
        unique_values = sorted(unique_values)
        groups_data = []
        for value in unique_values:
            mask = valid_data[biomarker] == value
            subset = valid_data[mask]
            
            if len(subset) > 0 and subset[f'{survival_type}.event'].sum() > 0:
                time_col = f'{survival_type}.time'
                event_col = f'{survival_type}.event'
                
                kmf.fit(subset[time_col], subset[event_col], label=f"{value}")

                fig.add_trace(go.Scatter(
                    x=kmf.survival_function_.index,
                    y=kmf.survival_function_.iloc[:, 0],
                    mode='lines',
                    name=f"{value} (n={len(subset)})",
                    line=dict(width=3),
                    hovertemplate='Time: %{x}<br>Survival: %{y:.3f}<extra></extra>'
                ))
                
                groups_data.append((subset[time_col], subset[event_col]))
        
        logrank_result = None
        if len(groups_data) == 2:
            try:
                result = logrank_test(groups_data[0][0], groups_data[1][0], 
                                    groups_data[0][1], groups_data[1][1])
                logrank_result = {
                    'p_value': result.p_value,
                    'test_statistic': result.test_statistic
                }
            except:
                pass
        
        survival_title = 'Overall Survival' if survival_type == 'os' else 'Recurrence-Free Survival'
        title = f'{survival_title} by {biomarker}'
        if logrank_result:
            title += f' (Log-rank p={logrank_result["p_value"]:.4f})'
        
        fig.update_layout(
            title=title,
            xaxis_title='Time (Months)',
            yaxis_title='Survival Probability',
            hovermode='x unified',
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        return fig, logrank_result
        
    except Exception as e:
        st.error(f"Error creating survival plot: {e}")
        return go.Figure(), None

def plot_biomarker_distribution(df, biomarker):
    """
    Enhanced distribution plot for biomarkers
    """
    try:
        if pd.api.types.is_numeric_dtype(df[biomarker]) and df[biomarker].nunique() > 10:
            fig = px.histogram(df, x=biomarker, 
                              title=f"Distribution of {biomarker}",
                              marginal="box",
                              hover_data=df.columns)
        else:
            value_counts = df[biomarker].value_counts().reset_index()
            value_counts.columns = [biomarker, 'count']
            fig = px.bar(value_counts, x=biomarker, y='count',
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
        # summary statistics
        response_data = df.groupby([biomarker, 'trtgrp']).agg({
            'os.time': ['mean', 'std', 'count'],
            'rfs.time': ['mean', 'std'],
            'os.event': 'mean'
        }).round(2).reset_index()
        response_data.columns = ['_'.join(col).strip('_') for col in response_data.columns.values]
        
        fig = px.bar(response_data, 
                    x=biomarker, 
                    y='os.time_mean', 
                    color='trtgrp', 
                    barmode='group',
                    error_y='os.time_std',
                    title=f"Average Overall Survival by {biomarker} and Treatment",
                    labels={'os.time_mean': 'Mean Survival Time (Months)'},
                    hover_data=['os.time_count'])
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating treatment response plot: {e}")
        return px.bar()

def plot_gene_expression(df, biomarker):
    """
    Plot gene expression distribution
    """
    try:
        fig = px.histogram(df, x=biomarker, 
                          title=f"Distribution of {biomarker} Expression",
                          marginal="box",
                          labels={biomarker: 'Expression Level'})
        return fig
    except Exception as e:
        st.error(f"Error creating gene expression plot: {e}")
        return px.histogram()

def create_biomarker_selection_table(clinical_df):
    """Create a table for selecting clinical biomarkers"""
    st.sidebar.subheader("📋 Clinical Biomarkers")
    
    # Define clinical biomarkers
    clinical_biomarkers = {
        'Stage': 'stage',
        'MSI Status': 'MSI',
        'Tumor Grade': 'grade',
        'Tumor Site': 'site',
        'Treatment Group': 'trtgrp',
        'Gender': 'gender',
        'Age Group': 'age_group'
    }
    
    # only include biomarkers that exist in the data
    available_biomarkers = {display_name: col_name 
                          for display_name, col_name in clinical_biomarkers.items() 
                          if col_name in clinical_df.columns}
    
    selected_biomarker = None
    
    # Create a selection table
    for display_name, col_name in available_biomarkers.items():
        unique_values = ['All'] + sorted([str(x) for x in clinical_df[col_name].dropna().unique()])
        selected_value = st.sidebar.selectbox(
            display_name,
            options=unique_values,
            index=0,
            key=f"biomarker_{col_name}"
        )

        if selected_value != 'All':
            selected_biomarker = (display_name, col_name, selected_value)
            # only one biomarker can be selected at a time
            break
    
    return selected_biomarker

def main():
    st.set_page_config(page_title="Colorectal Cancer Biomarker Explorer", 
                       layout="wide", 
                       page_icon="🔬")
    
    st.title("🔬 Colorectal Cancer Biomarker Explorer")
    st.markdown("Explore predictive and prognostic biomarkers for treatment response and survival")
    
    clinical_df, expression_data, gene_names = load_data()
    
    if clinical_df is None:
        st.stop()
    
    st.sidebar.header("🔧 Analysis Parameters")
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Dataset Info")
    st.sidebar.write(f"**Patients:** {len(clinical_df)}")
    if expression_data is not None:
        st.sidebar.write(f"**Genes:** {expression_data.shape[1]}")
    
    selected_gene_biomarker = None
    if expression_data is not None:
     
        st.sidebar.markdown(
            """
            <div style="background-color: #cde2ee; padding: 10px; border-radius: 10px; border: 1px solid #0b5394;">
                <h3 style="color: #0b5394; margin: 0;">🧬 Select gene</h3>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        selected_gene = st.sidebar.selectbox(
            "Select Gene",
            options=expression_data.columns.tolist(),
            index=0 if len(expression_data.columns) > 0 else 0,
            label_visibility="hidden"
        )
        if selected_gene:
            clinical_df = clinical_df.copy()
            min_samples = min(len(clinical_df), len(expression_data))
            clinical_df[selected_gene] = expression_data[selected_gene].values[:min_samples]
            selected_gene_biomarker = (f"Gene: {selected_gene}", selected_gene, "Expression")
    
    # biomarker selection table for clinical biomarkers
    selected_clinical_biomarker = create_biomarker_selection_table(clinical_df)

    if selected_clinical_biomarker:
        # Use clinical biomarker if selected
        biomarker_display, biomarker_column, selected_value = selected_clinical_biomarker
        # Filter data based on selection
        if selected_value != 'All':
            clinical_df = clinical_df[clinical_df[biomarker_column] == selected_value]
    elif selected_gene_biomarker:
        # Use gene biomarker if no clinical biomarker selected
        biomarker_display, biomarker_column, _ = selected_gene_biomarker
    else:
        # Default to first available clinical biomarker
        clinical_biomarkers = {
            'Stage': 'stage',
            'MSI Status': 'MSI',
            'Tumor Grade': 'grade',
            'Tumor Site': 'site',
            'Treatment Group': 'trtgrp',
            'Gender': 'gender',
            'Age Group': 'age_group'
        }
        available_biomarkers = {display_name: col_name 
                              for display_name, col_name in clinical_biomarkers.items() 
                              if col_name in clinical_df.columns}
        if available_biomarkers:
            biomarker_display = list(available_biomarkers.keys())[0]
            biomarker_column = available_biomarkers[biomarker_display]
        else:
            st.error("No biomarker columns found in the data.")
            st.write("Available columns:", list(clinical_df.columns))
            st.stop()
    
    analysis_type = st.sidebar.radio(
        "Analysis Type",
        ["Survival Analysis", "Biomarker Distribution", "Treatment Response"]
    )

    if analysis_type == "Survival Analysis":
        survival_type = st.sidebar.radio(
            "Survival Type",
            ["Overall Survival", "Recurrence-Free Survival"]
        )
        survival_code = {'Overall Survival': 'os', 
                        'Recurrence-Free Survival': 'rfs'}[survival_type]
 
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Analysis Results")
        
        if analysis_type == "Survival Analysis":
            fig, logrank_result = plot_survival_analysis(clinical_df, biomarker_column, survival_code)
            st.plotly_chart(fig, use_container_width=True)
            
            if logrank_result:
                st.info(f"**Log-rank Test:** p-value = {logrank_result['p_value']:.4f}")
                
        elif analysis_type == "Biomarker Distribution":
            # Use gene expression plot for genes, regular distribution for clinical biomarkers
            if selected_gene_biomarker:
                fig = plot_gene_expression(clinical_df, biomarker_column)
            else:
                fig = plot_biomarker_distribution(clinical_df, biomarker_column)
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = plot_treatment_response(clinical_df, biomarker_column)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Summary Statistics")
        
        if biomarker_column in clinical_df.columns:
            st.write(f"**{biomarker_display} Distribution:**")
            if pd.api.types.is_numeric_dtype(clinical_df[biomarker_column]) and clinical_df[biomarker_column].nunique() > 10:
                mean_val = clinical_df[biomarker_column].mean()
                std_val = clinical_df[biomarker_column].std()
                st.write(f"• **Mean:** {mean_val:.2f}")
                st.write(f"• **Std Dev:** {std_val:.2f}")
                st.write(f"• **Range:** {clinical_df[biomarker_column].min():.2f} - {clinical_df[biomarker_column].max():.2f}")
            else:
                biomarker_counts = clinical_df[biomarker_column].value_counts()
                for value, count in biomarker_counts.items():
                    percentage = (count / total_patients) * 100
                    st.write(f"• **{value}:** {count} ({percentage:.1f}%)")
        
        # Survival statistics
        survival_metrics = []
        for time_col, event_col, name in [('os.time', 'os.event', 'OS'),
                                         ('rfs.time', 'rfs.event', 'RFS')]:
            if time_col in clinical_df.columns:
                avg_time = clinical_df[time_col].mean()
                if event_col in clinical_df.columns:
                    event_rate = clinical_df[event_col].mean()
                    survival_metrics.append((f"{name} Time", f"{avg_time:.1f} mo"))
                    survival_metrics.append((f"{name} Events", f"{event_rate:.1%}"))
        
        for metric_name, value in survival_metrics:
            st.metric(metric_name, value)
        

if __name__ == "__main__":
    main()