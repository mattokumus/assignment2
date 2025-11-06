#!/usr/bin/env python3
"""
ECHR Case Explorer - Interactive Dashboard
Research Question: Does the European Court of Human Rights treat countries differently?

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# Page config
st.set_page_config(
    page_title="ECHR Case Explorer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Regional classification
EASTERN_EUROPE = [
    'Russian Federation', 'Ukraine', 'Poland', 'Romania', 'Hungary',
    'Bulgaria', 'Croatia', 'Slovenia', 'Slovakia', 'Czechia',
    'Lithuania', 'Latvia', 'Estonia', 'Moldova, Republic of',
    'Serbia', 'Bosnia and Herzegovina', 'North Macedonia', 'Albania',
    'Armenia', 'Azerbaijan', 'Georgia', 'Turkey', 'Montenegro'
]

WESTERN_EUROPE = [
    'United Kingdom', 'Germany', 'France', 'Italy', 'Spain',
    'Netherlands', 'Belgium', 'Austria', 'Switzerland', 'Sweden',
    'Norway', 'Denmark', 'Finland', 'Ireland', 'Portugal', 'Greece',
    'Cyprus', 'Malta', 'Luxembourg', 'Iceland', 'San Marino', 'Liechtenstein'
]

@st.cache_data
def load_data():
    """Load and preprocess data"""
    df = pd.read_csv('extracted_data.csv')

    # Add region
    df['region'] = df['country_name'].apply(
        lambda x: 'Eastern Europe' if x in EASTERN_EUROPE
        else 'Western Europe' if x in WESTERN_EUROPE
        else 'Other'
    )

    # Extract primary article
    def get_primary_article(articles_str):
        if pd.isna(articles_str) or articles_str == '':
            return 'Unknown'
        articles = [a.strip() for a in str(articles_str).split(',')]
        return articles[0] if articles else 'Unknown'

    df['primary_article'] = df['articles'].apply(get_primary_article)

    return df

# Load data
df = load_data()

# ============================================================================
# HEADER
# ============================================================================
st.markdown('<div class="main-header">⚖️ ECHR Case Explorer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Interactive Analysis of 1,904 European Court of Human Rights Cases (2000-2024)</div>', unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - FILTERS
# ============================================================================
st.sidebar.header("🔍 Filters")

# Country filter
countries = sorted(df['country_name'].unique())
selected_countries = st.sidebar.multiselect(
    "🌍 Select Countries",
    options=countries,
    default=countries[:10],  # Top 10 by default
    help="Select one or more countries to analyze"
)

# Region filter
region_filter = st.sidebar.radio(
    "🗺️ Region",
    options=['All', 'Eastern Europe', 'Western Europe'],
    index=0
)

# Year range filter
min_year, max_year = int(df['year'].min()), int(df['year'].max())
year_range = st.sidebar.slider(
    "📅 Year Range",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year),
    step=1
)

# Article filter
articles = sorted([a for a in df['primary_article'].unique() if a != 'Unknown'])
selected_articles = st.sidebar.multiselect(
    "📜 Articles",
    options=articles,
    default=articles[:5] if len(articles) >= 5 else articles,
    help="Select ECHR articles to analyze"
)

# Violation filter
violation_filter = st.sidebar.radio(
    "⚖️ Violation Status",
    options=['All', 'Violations Only', 'No Violations Only'],
    index=0
)

# Applicant type filter
applicant_types = sorted(df['applicant_type'].unique())
selected_applicant_types = st.sidebar.multiselect(
    "👥 Applicant Type",
    options=applicant_types,
    default=applicant_types,
    help="Select applicant types"
)

# ============================================================================
# APPLY FILTERS
# ============================================================================
filtered_df = df.copy()

# Apply country filter
if selected_countries:
    filtered_df = filtered_df[filtered_df['country_name'].isin(selected_countries)]

# Apply region filter
if region_filter != 'All':
    filtered_df = filtered_df[filtered_df['region'] == region_filter]

# Apply year filter
filtered_df = filtered_df[
    (filtered_df['year'] >= year_range[0]) &
    (filtered_df['year'] <= year_range[1])
]

# Apply article filter
if selected_articles:
    filtered_df = filtered_df[filtered_df['primary_article'].isin(selected_articles)]

# Apply violation filter
if violation_filter == 'Violations Only':
    filtered_df = filtered_df[filtered_df['has_violation'] == 1]
elif violation_filter == 'No Violations Only':
    filtered_df = filtered_df[filtered_df['has_violation'] == 0]

# Apply applicant type filter
if selected_applicant_types:
    filtered_df = filtered_df[filtered_df['applicant_type'].isin(selected_applicant_types)]

# ============================================================================
# KEY STATISTICS
# ============================================================================
st.markdown("---")
st.subheader("📊 Key Statistics")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="Total Cases",
        value=f"{len(filtered_df):,}",
        delta=f"{len(filtered_df)/len(df)*100:.1f}% of dataset"
    )

with col2:
    violation_rate = filtered_df['has_violation'].mean() * 100
    overall_rate = df['has_violation'].mean() * 100
    st.metric(
        label="Violation Rate",
        value=f"{violation_rate:.1f}%",
        delta=f"{violation_rate - overall_rate:+.1f}% vs overall"
    )

with col3:
    st.metric(
        label="Countries",
        value=filtered_df['country_name'].nunique()
    )

with col4:
    st.metric(
        label="Articles Cited",
        value=filtered_df['primary_article'].nunique()
    )

with col5:
    st.metric(
        label="Year Range",
        value=f"{filtered_df['year'].min()}-{filtered_df['year'].max()}"
    )

# ============================================================================
# VISUALIZATIONS
# ============================================================================
st.markdown("---")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["🌍 Country Analysis", "📜 Article Analysis", "🗺️ Regional Comparison", "📈 Temporal Trends"])

# ============================================================================
# TAB 1: COUNTRY ANALYSIS
# ============================================================================
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Violation Rate by Country")

        # Calculate violation rate by country
        country_stats = filtered_df.groupby('country_name').agg({
            'has_violation': ['mean', 'count']
        }).reset_index()
        country_stats.columns = ['Country', 'Violation_Rate', 'Case_Count']
        country_stats['Violation_Rate'] = country_stats['Violation_Rate'] * 100
        country_stats = country_stats.sort_values('Violation_Rate', ascending=False)

        # Create bar chart
        fig1 = px.bar(
            country_stats,
            x='Violation_Rate',
            y='Country',
            orientation='h',
            color='Violation_Rate',
            color_continuous_scale='RdYlGn_r',
            labels={'Violation_Rate': 'Violation Rate (%)', 'Country': ''},
            hover_data={'Case_Count': True}
        )
        fig1.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown("#### Case Count by Country")

        # Create treemap
        fig2 = px.treemap(
            country_stats,
            path=['Country'],
            values='Case_Count',
            color='Violation_Rate',
            color_continuous_scale='RdYlGn_r',
            labels={'Case_Count': 'Cases', 'Violation_Rate': 'Violation Rate (%)'}
        )
        fig2.update_layout(height=600)
        st.plotly_chart(fig2, use_container_width=True)

# ============================================================================
# TAB 2: ARTICLE ANALYSIS
# ============================================================================
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Article Distribution")

        # Article counts
        article_counts = filtered_df['primary_article'].value_counts().reset_index()
        article_counts.columns = ['Article', 'Count']

        # Create pie chart
        fig3 = px.pie(
            article_counts.head(10),
            values='Count',
            names='Article',
            title='Top 10 Most Cited Articles',
            hole=0.4
        )
        fig3.update_layout(height=500)
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        st.markdown("#### Violation Rate by Article")

        # Article violation rates
        article_stats = filtered_df.groupby('primary_article').agg({
            'has_violation': ['mean', 'count']
        }).reset_index()
        article_stats.columns = ['Article', 'Violation_Rate', 'Case_Count']
        article_stats['Violation_Rate'] = article_stats['Violation_Rate'] * 100
        article_stats = article_stats[article_stats['Case_Count'] >= 10]  # Min 10 cases
        article_stats = article_stats.sort_values('Violation_Rate', ascending=False).head(15)

        # Create bar chart
        fig4 = px.bar(
            article_stats,
            x='Article',
            y='Violation_Rate',
            color='Violation_Rate',
            color_continuous_scale='RdYlGn_r',
            labels={'Violation_Rate': 'Violation Rate (%)', 'Article': ''},
            hover_data={'Case_Count': True}
        )
        fig4.update_layout(height=500, showlegend=False)
        fig4.update_xaxes(tickangle=45)
        st.plotly_chart(fig4, use_container_width=True)

# ============================================================================
# TAB 3: REGIONAL COMPARISON
# ============================================================================
with tab3:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Regional Violation Rates")

        # Regional stats
        regional_stats = filtered_df.groupby('region').agg({
            'has_violation': ['mean', 'count']
        }).reset_index()
        regional_stats.columns = ['Region', 'Violation_Rate', 'Case_Count']
        regional_stats['Violation_Rate'] = regional_stats['Violation_Rate'] * 100

        # Create bar chart
        fig5 = px.bar(
            regional_stats,
            x='Region',
            y='Violation_Rate',
            color='Region',
            text='Violation_Rate',
            labels={'Violation_Rate': 'Violation Rate (%)'},
            color_discrete_map={
                'Eastern Europe': '#ff7f0e',
                'Western Europe': '#1f77b4',
                'Other': '#2ca02c'
            }
        )
        fig5.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig5.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig5, use_container_width=True)

        # Display comparison
        if 'Eastern Europe' in regional_stats['Region'].values and 'Western Europe' in regional_stats['Region'].values:
            east_rate = regional_stats[regional_stats['Region'] == 'Eastern Europe']['Violation_Rate'].values[0]
            west_rate = regional_stats[regional_stats['Region'] == 'Western Europe']['Violation_Rate'].values[0]
            gap = east_rate - west_rate

            st.info(f"""
            **Regional Gap:** Eastern Europe has a **{gap:.1f} percentage points** higher violation rate than Western Europe.

            - Eastern Europe: {east_rate:.1f}%
            - Western Europe: {west_rate:.1f}%
            """)

    with col2:
        st.markdown("#### Regional Case Distribution")

        # Create pie chart for regional distribution
        fig6 = px.pie(
            regional_stats,
            values='Case_Count',
            names='Region',
            title='Cases by Region',
            color='Region',
            color_discrete_map={
                'Eastern Europe': '#ff7f0e',
                'Western Europe': '#1f77b4',
                'Other': '#2ca02c'
            }
        )
        fig6.update_layout(height=400)
        st.plotly_chart(fig6, use_container_width=True)

        # Regional statistics table
        st.markdown("#### Regional Statistics Summary")
        regional_summary = regional_stats.copy()
        regional_summary['Violation_Rate'] = regional_summary['Violation_Rate'].apply(lambda x: f"{x:.1f}%")
        st.dataframe(
            regional_summary,
            column_config={
                "Region": "Region",
                "Violation_Rate": "Violation Rate",
                "Case_Count": "Case Count"
            },
            hide_index=True,
            use_container_width=True
        )

# ============================================================================
# TAB 4: TEMPORAL TRENDS
# ============================================================================
with tab4:
    st.markdown("#### Temporal Trends Over Time")

    col1, col2 = st.columns(2)

    with col1:
        # Cases over time
        temporal_counts = filtered_df.groupby('year').size().reset_index()
        temporal_counts.columns = ['Year', 'Case_Count']

        fig7 = px.line(
            temporal_counts,
            x='Year',
            y='Case_Count',
            markers=True,
            labels={'Case_Count': 'Number of Cases'},
            title='Cases per Year'
        )
        fig7.update_layout(height=400)
        st.plotly_chart(fig7, use_container_width=True)

    with col2:
        # Violation rate over time
        temporal_viol = filtered_df.groupby('year')['has_violation'].mean().reset_index()
        temporal_viol.columns = ['Year', 'Violation_Rate']
        temporal_viol['Violation_Rate'] = temporal_viol['Violation_Rate'] * 100

        fig8 = px.line(
            temporal_viol,
            x='Year',
            y='Violation_Rate',
            markers=True,
            labels={'Violation_Rate': 'Violation Rate (%)'},
            title='Violation Rate per Year'
        )
        fig8.update_layout(height=400)
        st.plotly_chart(fig8, use_container_width=True)

    # Regional trends over time
    st.markdown("#### Violation Rate Trends by Region")

    regional_temporal = filtered_df.groupby(['year', 'region'])['has_violation'].mean().reset_index()
    regional_temporal.columns = ['Year', 'Region', 'Violation_Rate']
    regional_temporal['Violation_Rate'] = regional_temporal['Violation_Rate'] * 100

    fig9 = px.line(
        regional_temporal,
        x='Year',
        y='Violation_Rate',
        color='Region',
        markers=True,
        labels={'Violation_Rate': 'Violation Rate (%)'},
        color_discrete_map={
            'Eastern Europe': '#ff7f0e',
            'Western Europe': '#1f77b4',
            'Other': '#2ca02c'
        }
    )
    fig9.update_layout(height=400)
    st.plotly_chart(fig9, use_container_width=True)

# ============================================================================
# DATA TABLE
# ============================================================================
st.markdown("---")
st.subheader("📋 Filtered Case Data")

# Display controls
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.markdown(f"**Showing {len(filtered_df)} cases** (filtered from {len(df)} total)")

with col2:
    # Sort options
    sort_column = st.selectbox(
        "Sort by",
        options=['year', 'country_name', 'has_violation', 'primary_article'],
        format_func=lambda x: {
            'year': 'Year',
            'country_name': 'Country',
            'has_violation': 'Violation',
            'primary_article': 'Article'
        }[x]
    )

with col3:
    sort_order = st.radio("Order", options=['Descending', 'Ascending'], horizontal=True)

# Sort dataframe
ascending = (sort_order == 'Ascending')
display_df = filtered_df.sort_values(sort_column, ascending=ascending)

# Select columns to display
display_columns = [
    'itemid', 'country_name', 'year', 'primary_article',
    'has_violation', 'applicant_type', 'docname'
]

# Display table
st.dataframe(
    display_df[display_columns].head(100),  # Limit to 100 rows for performance
    column_config={
        "itemid": "Case ID",
        "country_name": "Country",
        "year": "Year",
        "primary_article": "Article",
        "has_violation": st.column_config.CheckboxColumn("Violation"),
        "applicant_type": "Applicant Type",
        "docname": "Case Name"
    },
    hide_index=True,
    use_container_width=True
)

if len(filtered_df) > 100:
    st.info(f"Showing first 100 cases. Download CSV below for full dataset ({len(filtered_df)} cases).")

# ============================================================================
# DOWNLOAD BUTTON
# ============================================================================
st.markdown("---")
st.subheader("💾 Download Data")

col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("Download the filtered dataset as CSV for further analysis.")

with col2:
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"echr_filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p><strong>ECHR Case Explorer</strong> | Research Question: Does the European Court of Human Rights treat countries differently?</p>
    <p>Data: 1,904 cases (2000-2024) | Source: European Court of Human Rights</p>
    <p>📊 <a href="https://github.com/mattokumus/assignment2" target="_blank">GitHub Repository</a> |
       📖 <a href="METHODOLOGY.md" target="_blank">Methodology</a> |
       📄 <a href="ANALYSIS_REPORT_EN.md" target="_blank">Full Report</a></p>
</div>
""", unsafe_allow_html=True)

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("""
### ℹ️ About

This interactive dashboard allows you to explore ECHR case data across:

- **45 Countries**
- **25 Years** (2000-2024)
- **Multiple Articles**
- **Different Regions**

Use the filters above to customize your analysis.

**Key Finding:** Eastern Europe shows systematically higher violation rates (+21.6 pp, p<0.001)

---

**Created with:** Streamlit + Plotly
**Author:** [Your Name]
""")
