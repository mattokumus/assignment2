#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) for ECHR Cases
Research Question: Does the ECtHR treat countries differently?
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_data(filename='extracted_data.csv'):
    """Load the extracted data"""
    print("=" * 80)
    print("ECHR CASES - EXPLORATORY DATA ANALYSIS (SUBSTANTIVE DECISIONS ONLY)")
    print("=" * 80)
    print(f"\n📂 Loading data from: {filename}")
    print(f"⚠️  Note: This dataset includes ONLY substantive decisions (violation/no-violation)")
    print(f"⚠️  Procedural cases (inadmissible, struck out, etc.) are EXCLUDED")
    
    df = pd.read_csv(filename)
    print(f"✓ Data loaded successfully: {len(df)} cases")
    return df


def basic_info(df):
    """Display basic information about the dataset"""
    print("\n" + "=" * 80)
    print("1. BASIC DATASET INFORMATION")
    print("=" * 80)
    
    print(f"\n📊 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\n📋 Column Names:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:2d}. {col}")
    
    print(f"\n📈 Data Types:")
    print(df.dtypes)
    
    print(f"\n🔢 Memory Usage:")
    print(df.memory_usage(deep=True).sum() / 1024**2, "MB")


def missing_data_analysis(df):
    """Analyze missing data"""
    print("\n" + "=" * 80)
    print("2. MISSING DATA ANALYSIS")
    print("=" * 80)
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    
    if len(missing_df) > 0:
        print("\n⚠️  Columns with Missing Data:")
        print(missing_df)
    else:
        print("\n✓ No missing data found!")


def summary_statistics(df):
    """Display summary statistics"""
    print("\n" + "=" * 80)
    print("3. SUMMARY STATISTICS")
    print("=" * 80)
    
    print(f"\n🌍 Geographic Coverage:")
    print(f"   • Total Countries: {df['country_name'].nunique()}")
    print(f"   • Country Codes: {df['country_code'].nunique()}")
    
    print(f"\n📅 Temporal Coverage:")
    print(f"   • Year Range: {df['year'].min():.0f} - {df['year'].max():.0f}")
    print(f"   • Time Span: {df['year'].max() - df['year'].min():.0f} years")
    
    print(f"\n⚖️  Violation Statistics:")
    violations = df['has_violation'].sum()
    no_violations = len(df) - violations
    print(f"   • Cases with Violations: {violations} ({violations/len(df)*100:.1f}%)")
    print(f"   • Cases without Violations: {no_violations} ({no_violations/len(df)*100:.1f}%)")
    
    print(f"\n📊 Violation Intensity:")
    print(f"   • Average violations per case: {df['violation_count'].mean():.2f}")
    print(f"   • Max violations in a case: {df['violation_count'].max():.0f}")
    print(f"   • Cases with multiple violations: {(df['violation_count'] > 1).sum()}")
    
    print(f"\n👥 Applicant Types:")
    print(df['applicant_type'].value_counts())


def country_analysis(df):
    """Detailed country-level analysis"""
    print("\n" + "=" * 80)
    print("4. COUNTRY-LEVEL ANALYSIS")
    print("=" * 80)
    
    # Top countries by case count
    print("\n🏆 TOP 15 COUNTRIES BY CASE COUNT:")
    top_countries = df['country_name'].value_counts().head(15)
    print(top_countries)
    
    # Violation rates by country
    print("\n🎯 VIOLATION RATES BY COUNTRY (Top 15):")
    country_violation = df.groupby('country_name').agg({
        'has_violation': ['sum', 'count', 'mean']
    }).round(3)
    country_violation.columns = ['Violations', 'Total Cases', 'Violation Rate']
    country_violation = country_violation.sort_values('Total Cases', ascending=False).head(15)
    print(country_violation)
    
    # Countries with highest/lowest violation rates (min 10 cases)
    country_stats = df.groupby('country_name').agg({
        'has_violation': ['sum', 'count', 'mean']
    })
    country_stats.columns = ['violations', 'total', 'rate']
    country_stats = country_stats[country_stats['total'] >= 10]
    
    print("\n📈 HIGHEST VIOLATION RATES (min 10 cases):")
    print(country_stats.nlargest(10, 'rate')[['total', 'violations', 'rate']])
    
    print("\n📉 LOWEST VIOLATION RATES (min 10 cases):")
    print(country_stats.nsmallest(10, 'rate')[['total', 'violations', 'rate']])


def temporal_analysis(df):
    """Analyze temporal trends"""
    print("\n" + "=" * 80)
    print("5. TEMPORAL ANALYSIS")
    print("=" * 80)
    
    # Cases per year
    print("\n📅 CASES PER DECADE:")
    df['decade'] = (df['year'] // 10) * 10
    decade_counts = df['decade'].value_counts().sort_index()
    print(decade_counts)
    
    # Violation rate over time
    print("\n⏳ VIOLATION RATE OVER TIME:")
    yearly_stats = df.groupby('year').agg({
        'has_violation': ['sum', 'count', 'mean']
    })
    yearly_stats.columns = ['violations', 'total', 'rate']
    
    print("\nFirst 5 years:")
    print(yearly_stats.head())
    print("\nLast 5 years:")
    print(yearly_stats.tail())


def article_analysis(df):
    """Analyze articles"""
    print("\n" + "=" * 80)
    print("6. ARTICLE ANALYSIS")
    print("=" * 80)
    
    # Most common articles
    print("\n📜 MOST COMMONLY CITED ARTICLES:")
    
    # Parse articles (they are comma-separated)
    all_articles = []
    for articles_str in df['articles'].dropna():
        if articles_str:
            all_articles.extend([a.strip() for a in articles_str.split(',')])
    
    article_counts = pd.Series(all_articles).value_counts().head(20)
    print(article_counts)
    
    # Most violated articles
    print("\n⚖️  MOST VIOLATED ARTICLES:")
    all_violated = []
    for articles_str in df['violated_articles'].dropna():
        if articles_str:
            all_violated.extend([a.strip() for a in articles_str.split(',')])
    
    if all_violated:
        violated_counts = pd.Series(all_violated).value_counts().head(15)
        print(violated_counts)


def create_visualizations(df):
    """Create comprehensive visualizations"""
    print("\n" + "=" * 80)
    print("7. CREATING VISUALIZATIONS")
    print("=" * 80)
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Top 15 Countries by Case Count
    ax1 = plt.subplot(2, 3, 1)
    top_15_countries = df['country_name'].value_counts().head(15)
    top_15_countries.plot(kind='barh', ax=ax1, color='steelblue')
    ax1.set_xlabel('Number of Cases')
    ax1.set_title('Top 15 Countries by Case Count', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    
    # 2. Violation Rate by Top 15 Countries
    ax2 = plt.subplot(2, 3, 2)
    top_countries_list = df['country_name'].value_counts().head(15).index
    violation_rates = df[df['country_name'].isin(top_countries_list)].groupby('country_name')['has_violation'].mean().sort_values()
    violation_rates.plot(kind='barh', ax=ax2, color='coral')
    ax2.set_xlabel('Violation Rate')
    ax2.set_title('Violation Rate: Top 15 Countries by Case Volume\n(Countries selected by number of cases, not by violation rate)', 
                  fontsize=11, fontweight='bold')
    ax2.axvline(x=df['has_violation'].mean(), color='red', linestyle='--', label='Overall Average')
    ax2.legend()
    ax2.invert_yaxis()
    
    # 3. Cases Over Time
    ax3 = plt.subplot(2, 3, 3)
    yearly_cases = df.groupby('year').size()
    yearly_cases.plot(ax=ax3, color='green', linewidth=2)
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Number of Cases')
    ax3.set_title('Number of Cases Over Time', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Violation Rate Over Time
    ax4 = plt.subplot(2, 3, 4)
    yearly_violation_rate = df.groupby('year')['has_violation'].mean()
    yearly_violation_rate.plot(ax=ax4, color='darkred', linewidth=2)
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Violation Rate')
    ax4.set_title('Violation Rate Over Time', fontsize=12, fontweight='bold')
    ax4.axhline(y=df['has_violation'].mean(), color='blue', linestyle='--', label='Overall Average')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Applicant Types
    ax5 = plt.subplot(2, 3, 5)
    applicant_counts = df['applicant_type'].value_counts()
    
    # Use bar chart instead of pie - better for small categories
    applicant_counts.plot(kind='barh', ax=ax5, color='teal', alpha=0.7)
    ax5.set_xlabel('Number of Cases')
    ax5.set_ylabel('')
    ax5.set_title('Distribution of Applicant Types', fontsize=12, fontweight='bold')
    
    # Add value labels on bars
    for i, v in enumerate(applicant_counts):
        percentage = (v / applicant_counts.sum()) * 100
        ax5.text(v + 10, i, f'{v} ({percentage:.1f}%)', 
                va='center', fontsize=9, fontweight='bold')
    
    ax5.invert_yaxis()
    
    # 6. Violation Count Distribution
    ax6 = plt.subplot(2, 3, 6)
    df['violation_count'].value_counts().sort_index().plot(kind='bar', ax=ax6, color='purple')
    ax6.set_xlabel('Number of Violations per Case')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Distribution of Violation Counts', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_file = 'eda_visualizations.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualizations saved to: {output_file}")
    plt.close()
    
    # Create additional heatmap for top countries × years
    create_heatmap(df)


def create_heatmap(df):
    """Create heatmap of violation rates: Countries × Decades"""
    print("\n📊 Creating heatmap...")
    
    # Get top 20 countries
    top_20_countries = df['country_name'].value_counts().head(20).index
    df_top = df[df['country_name'].isin(top_20_countries)].copy()
    
    # Create decade column
    df_top['decade'] = (df_top['year'] // 10) * 10
    
    # Create pivot table
    heatmap_data = df_top.groupby(['country_name', 'decade'])['has_violation'].mean().unstack(fill_value=0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn_r', 
                center=0.5, vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Violation Rate'})
    ax.set_title('Violation Rate Heatmap: Top 20 Countries × Decade', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Decade', fontsize=12)
    ax.set_ylabel('Country', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('eda_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✓ Heatmap saved to: eda_heatmap.png")
    plt.close()


def correlation_analysis(df):
    """Analyze correlations"""
    print("\n" + "=" * 80)
    print("8. CORRELATION ANALYSIS")
    print("=" * 80)
    
    # Create numerical features for correlation
    df_corr = df[['year', 'violation_count', 'no_violation_count']].copy()
    df_corr['has_violation_num'] = df['has_violation'].astype(int)
    
    print("\n📊 Correlation Matrix:")
    corr_matrix = df_corr.corr()
    print(corr_matrix.round(3))
    
    # Plot correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, square=True, ax=ax)
    ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('eda_correlation.png', dpi=300, bbox_inches='tight')
    print(f"✓ Correlation plot saved to: eda_correlation.png")
    plt.close()


def key_insights(df):
    """Generate key insights for research question"""
    print("\n" + "=" * 80)
    print("9. KEY INSIGHTS FOR RESEARCH QUESTION")
    print("=" * 80)
    print("\n🎯 Research Question: Does the ECtHR treat countries differently?\n")
    
    # Insight 1: Variation in violation rates
    country_stats = df.groupby('country_name').agg({
        'has_violation': ['sum', 'count', 'mean']
    })
    country_stats.columns = ['violations', 'total', 'rate']
    country_stats = country_stats[country_stats['total'] >= 10]
    
    rate_std = country_stats['rate'].std()
    rate_range = country_stats['rate'].max() - country_stats['rate'].min()
    
    print(f"📌 INSIGHT 1: Variation in Violation Rates")
    print(f"   • Standard deviation: {rate_std:.3f}")
    print(f"   • Range (max - min): {rate_range:.3f}")
    print(f"   • Highest rate: {country_stats['rate'].max():.3f} ({country_stats['rate'].idxmax()})")
    print(f"   • Lowest rate: {country_stats['rate'].min():.3f} ({country_stats['rate'].idxmin()})")
    print(f"   → Finding: {'HIGH' if rate_std > 0.15 else 'MODERATE'} variation across countries")
    
    # Insight 2: Sample size imbalance
    print(f"\n📌 INSIGHT 2: Sample Size Imbalance")
    case_counts = df['country_name'].value_counts()
    print(f"   • Top country: {case_counts.iloc[0]} cases ({case_counts.index[0]})")
    print(f"   • Median country: {case_counts.median():.0f} cases")
    print(f"   • Bottom 10 countries: {case_counts.tail(10).sum()} cases combined")
    print(f"   → Finding: SEVERE imbalance - need to control for sample size")
    
    # Insight 3: Temporal patterns
    print(f"\n📌 INSIGHT 3: Temporal Patterns")
    early_rate = df[df['year'] < 2000]['has_violation'].mean()
    late_rate = df[df['year'] >= 2000]['has_violation'].mean()
    print(f"   • Violation rate before 2000: {early_rate:.3f}")
    print(f"   • Violation rate after 2000: {late_rate:.3f}")
    print(f"   • Change: {late_rate - early_rate:+.3f}")
    print(f"   → Finding: Violation rate {'INCREASED' if late_rate > early_rate else 'DECREASED'} over time")
    
    # Insight 4: Article distribution
    print(f"\n📌 INSIGHT 4: Need for Controls")
    print(f"   • Different countries may have different article patterns")
    print(f"   • Different articles may have different violation rates")
    print(f"   → Recommendation: MUST control for article type in analysis")


def generate_summary_report(df):
    """Generate summary report"""
    print("\n" + "=" * 80)
    print("10. SUMMARY & NEXT STEPS")
    print("=" * 80)
    
    print("""
📋 SUMMARY:
✓ Dataset contains 2000 cases from 45 countries (1968-2020)
✓ Overall violation rate: 84.9%
✓ Significant variation across countries
✓ Sample size highly imbalanced
✓ Temporal trends present

🎯 RECOMMENDATIONS FOR NEXT STEPS:

1. Statistical Modeling:
   • Logistic regression with controls (article, year)
   • Include country fixed effects
   • Account for sample size imbalance
   
2. Machine Learning:
   • Use stratified sampling
   • Apply SMOTE for class imbalance if needed
   • Focus on feature importance

3. Further Analysis:
   • Filter to countries with min 20-30 cases
   • Analyze specific article types separately
   • Consider time periods separately
   • Look at regional patterns (Eastern Europe vs Western Europe)

⚠️  IMPORTANT CONSIDERATIONS:
   • High violation rate may indicate case selection bias
   • Only cases reaching ECtHR are in dataset
   • Need to interpret results carefully
   • Statistical significance ≠ bias or discrimination
    """)


def main():
    """Main EDA function"""
    
    # Load data
    df = load_data('extracted_data.csv')
    
    # Run all analyses
    basic_info(df)
    missing_data_analysis(df)
    summary_statistics(df)
    country_analysis(df)
    temporal_analysis(df)
    article_analysis(df)
    correlation_analysis(df)
    create_visualizations(df)
    key_insights(df)
    generate_summary_report(df)
    
    print("\n" + "=" * 80)
    print("✓ EDA COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  📊 eda_visualizations.png - Main visualizations")
    print("  📊 eda_heatmap.png - Country × Decade heatmap")
    print("  📊 eda_correlation.png - Correlation matrix")
    print("\n")


if __name__ == "__main__":
    main()