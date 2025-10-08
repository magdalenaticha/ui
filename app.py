import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

st.set_page_config(page_title="Gene Analysis Dashboard", layout="wide")
st.title("Gene Analysis Dashboard")
st.markdown("Explore gene expression patterns, survival relationships, and treatment responses")


@st.cache_data
def generate_complete_sample_data():
    n_patients = 200
    n_genes = 50
    clinical_data = pd.DataFrame({
        'patient_id': [f'PAT_{i:03d}' for i in range(n_patients)],
        'age': np.random.normal(65, 10, n_patients).astype(int),
        'gender': np.random.choice(['Male', 'Female'], n_patients),
        'stage': np.random.choice(['I', 'II', 'III', 'IV'], n_patients, p=[0.3, 0.4, 0.2, 0.1]),
        'treatment': np.random.choice(['Drug_A', 'Drug_B', 'Placebo'], n_patients),
    })
    
    # survival data in days
    base_survival = np.random.exponential(365, n_patients)
    
    gene_names = [f'GENE_{i:03d}' for i in range(n_genes)]
    expression_data = pd.DataFrame()
    
    for i, gene in enumerate(gene_names):
        if i == 0:  # if gene 1 has high expression = higher survival rate
            expression_data[gene] = np.random.lognormal(mean=2, sigma=0.5, size=n_patients)
            clinical_data['survival_time'] = base_survival + (expression_data[gene] * 50)
        elif i == 1:  # gen 2 is predictive of treatment response
            expression_data[gene] = np.random.lognormal(mean=1.5, sigma=0.8, size=n_patients)
        else:
            expression_data[gene] = np.random.lognormal(mean=0, sigma=1, size=n_patients) # other genes with random effects
    
    # Add event data= death or cencoring
    clinical_data['event_occurred'] = np.random.choice([0, 1], n_patients, p=[0.3, 0.7]) # 70% of patienst die and 30% survive or are cencored
    clinical_data['survival_time'] = clinical_data['survival_time'].astype(int) # turns survival time into integers = whole days
    clinical_data['survival_time'] = np.maximum(clinical_data['survival_time'], 30)
    
    expression_data['patient_id'] = clinical_data['patient_id'].values
    
    return clinical_data, expression_data, gene_names

clinical_df, expression_df, gene_list = generate_complete_sample_data()

# sidebar
st.sidebar.header("Analysis Controls")

# gene selection
selected_gene = st.sidebar.selectbox("Select Gene to Analyze:", gene_list)

survival_threshold = st.sidebar.slider("Survival Analysis Threshold (%):", 
                                      min_value=10, max_value=90, value=50,)

st.sidebar.markdown("---")
st.sidebar.info(f"**Dataset Overview:**\n- {len(clinical_df)} patients\n- {len(gene_list)} genes")

st.header("Gene Expression Analysis")

col1, col2, col3, col4, col5 = st.columns(5)
gene_values = expression_df[selected_gene]

with col1:
    st.metric("Mean", f"{gene_values.mean():.2f}")
with col2:
    st.metric("Std Dev", f"{gene_values.std():.2f}")
with col3:
    st.metric("Min", f"{gene_values.min():.2f}")
with col4:
    st.metric("Max", f"{gene_values.max():.2f}")
with col5:
    st.metric("Median", f"{gene_values.median():.2f}")

dist_col1, dist_col2 = st.columns(2)

with dist_col1:
    fig_dist = px.histogram(expression_df, x=selected_gene,
                           title=f"Distribution of {selected_gene} Expression",
                           nbins=30, color_discrete_sequence=['#1f77b4'])
    st.plotly_chart(fig_dist, use_container_width=True)

with dist_col2:
    clinical_var = st.selectbox("Group by clinical variable:", 
                               ['gender', 'stage', 'treatment'])
    merged_data = expression_df.merge(clinical_df, on='patient_id')
    fig_box = px.box(merged_data, x=clinical_var, y=selected_gene,
                    title=f"{selected_gene} Expression by {clinical_var}",
                    color=clinical_var)
    st.plotly_chart(fig_box, use_container_width=True)

st.header("Survival Analysis")

survival_data = clinical_df.merge(expression_df[['patient_id', selected_gene]], on='patient_id')
cutoff = np.percentile(survival_data[selected_gene], survival_threshold)
survival_data['expression_group'] = ['High' if x > cutoff else 'Low' for x in survival_data[selected_gene]]

kmf = KaplanMeierFitter()
fig_survival = go.Figure()

colors = {'High': '#ff7f0e', 'Low': '#1f77b4'}

for group in ['High', 'Low']:
    group_data = survival_data[survival_data['expression_group'] == group]
    kmf.fit(group_data['survival_time'], group_data['event_occurred'], 
            label=f'{group} Expression (n={len(group_data)})')
    fig_survival.add_trace(go.Scatter(
        x=kmf.survival_function_.index,
        y=kmf.survival_function_[f'{group} Expression (n={len(group_data)})'],
        mode='lines',
        name=f'{group} Expression (n={len(group_data)})',
        line=dict(color=colors[group], width=3)
    ))

fig_survival.update_layout(
    title=f'Survival Analysis - {selected_gene} (Threshold: {survival_threshold}%)',
    xaxis_title='Time (Days)',
    yaxis_title='Survival Probability',
    hovermode='x unified'
)

st.plotly_chart(fig_survival, use_container_width=True)


surv_col1, surv_col2, surv_col3, surv_col4 = st.columns(4)

high_group = survival_data[survival_data['expression_group'] == 'High']
low_group = survival_data[survival_data['expression_group'] == 'Low']

results = logrank_test(
    high_group['survival_time'], 
    low_group['survival_time'],
    high_group['event_occurred'], 
    low_group['event_occurred']
)

with surv_col1:
    st.metric("Log-Rank p-value", f"{results.p_value:.4f}")
with surv_col2:
    st.metric("High Expression Patients", len(high_group))
with surv_col3:
    st.metric("Low Expression Patients", len(low_group))
with surv_col4:
    cutoff_val = f"{cutoff:.2f}"
    st.metric("Expression Cutoff", cutoff_val)

st.header("Treatment Response Analysis")

treatment_data = clinical_df.merge(expression_df[['patient_id', selected_gene]], on='patient_id')
treatment_data = treatment_data[treatment_data['treatment'].isin(['Drug_A', 'Drug_B'])]

treat_col1, treat_col2 = st.columns(2)

with treat_col1:
    fig_treat_box = px.box(treatment_data, x='treatment', y=selected_gene,
                          color='treatment', 
                          title=f"{selected_gene} Expression by Treatment",
                          color_discrete_sequence=['#1f77b4', '#ff7f0e'])
    st.plotly_chart(fig_treat_box, use_container_width=True)

with treat_col2:
    st.subheader("Treatment Group Summary")
    treat_summary = treatment_data.groupby('treatment').agg({
        selected_gene: ['mean', 'std', 'count'],
        'survival_time': 'median'
    }).round(3)
    st.dataframe(treat_summary)

st.subheader("Treatment Response Survival Analysis")

treatment_data['expression_group'] = treatment_data[selected_gene] > treatment_data[selected_gene].median()
treatment_data['treatment_group'] = treatment_data['treatment'] + ' - ' + \
                                   treatment_data['expression_group'].map({True: 'High', False: 'Low'})

kmf_treat = KaplanMeierFitter()
fig_treat_survival = go.Figure()

treatment_colors = {
    'Drug_A - High': '#d62728',
    'Drug_A - Low': '#ff9896',
    'Drug_B - High': '#9467bd', 
    'Drug_B - Low': '#c5b0d5'
}

for group in sorted(treatment_data['treatment_group'].unique()):
    group_data = treatment_data[treatment_data['treatment_group'] == group]
    kmf_treat.fit(group_data['survival_time'], group_data['event_occurred'], label=group)
    fig_treat_survival.add_trace(go.Scatter(
        x=kmf_treat.survival_function_.index,
        y=kmf_treat.survival_function_[group],
        mode='lines',
        name=group,
        line=dict(color=treatment_colors.get(group, '#000000'), width=3)
    ))

fig_treat_survival.update_layout(
    title=f'Treatment Response - {selected_gene}',
    xaxis_title='Time (Days)',
    yaxis_title='Survival Probability'
)

st.plotly_chart(fig_treat_survival, use_container_width=True)

st.markdown("---")
