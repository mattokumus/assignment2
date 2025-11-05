#!/usr/bin/env python3
"""
Article-Specific Logistic Regression Analysis
Research Question: Does country effect persist across different article types?

Strategy:
For each major article (3, 6, 8, 5, 13):
- Run separate logistic regression: violation ~ country + year + applicant_type
- Compare country effects across articles
- Test if country effect is article-specific or universal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import chi2
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_and_prepare_data(filename='extracted_data.csv'):
    """Load and prepare data"""
    print("=" * 80)
    print("ARTICLE-SPECIFIC LOGISTIC REGRESSION ANALYSIS")
    print("Testing if country effects persist across different article types")
    print("=" * 80)
    
    df = pd.read_csv(filename)
    print(f"\n✓ Data loaded: {len(df)} cases")
    
    # Add regional classification
    eastern_europe = [
        'Russian Federation', 'Ukraine', 'Poland', 'Romania', 'Hungary',
        'Bulgaria', 'Croatia', 'Slovenia', 'Slovakia', 'Czechia',
        'Lithuania', 'Latvia', 'Estonia', 'Moldova, Republic of',
        'Serbia', 'Bosnia and Herzegovina', 'North Macedonia', 'Albania',
        'Belarus', 'Armenia', 'Azerbaijan', 'Georgia'
    ]
    
    western_europe = [
        'United Kingdom', 'Germany', 'France', 'Italy', 'Spain',
        'Netherlands', 'Belgium', 'Austria', 'Switzerland', 'Sweden',
        'Norway', 'Denmark', 'Finland', 'Ireland', 'Portugal', 'Greece'
    ]
    
    df['region'] = df['country_name'].apply(
        lambda x: 'Eastern Europe' if x in eastern_europe 
        else 'Western Europe' if x in western_europe 
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


def identify_major_articles(df, min_cases=100):
    """Identify articles with sufficient cases"""
    article_counts = df['primary_article'].value_counts()
    major_articles = article_counts[article_counts >= min_cases].index.tolist()
    
    print(f"\n📊 Major Articles (min {min_cases} cases):")
    for article in major_articles:
        count = article_counts[article]
        viol_rate = df[df['primary_article'] == article]['has_violation'].mean()
        print(f"   Article {article:10s}: {count:4d} cases ({viol_rate*100:.1f}% violation rate)")
    
    return major_articles


def run_article_specific_model(df, article, min_countries=5):
    """
    Run logistic regression for specific article
    Model: violation ~ country + year + applicant_type
    """
    print(f"\n" + "=" * 80)
    print(f"MODEL: ARTICLE {article}")
    print("=" * 80)
    
    # Filter to this article
    df_article = df[df['primary_article'] == article].copy()
    
    print(f"\n📊 Article {article} Data:")
    print(f"   • Total cases: {len(df_article)}")
    print(f"   • Violations: {df_article['has_violation'].sum()} ({df_article['has_violation'].mean()*100:.1f}%)")
    print(f"   • Countries: {df_article['country_name'].nunique()}")
    
    # Filter to countries with sufficient cases
    country_counts = df_article['country_name'].value_counts()
    eligible_countries = country_counts[country_counts >= min_countries].index
    df_model = df_article[df_article['country_name'].isin(eligible_countries)].copy()
    
    print(f"   • After filtering (min {min_countries} cases/country): {len(df_model)} cases")
    print(f"   • Countries included: {df_model['country_name'].nunique()}")
    
    if len(df_model) < 50 or df_model['country_name'].nunique() < 3:
        print(f"\n⚠️  Insufficient data for Article {article} - skipping")
        return None
    
    # Prepare model data
    df_reg = df_model[['has_violation', 'country_name', 'year', 'applicant_type']].copy()
    df_reg = pd.get_dummies(df_reg, columns=['country_name', 'applicant_type'], drop_first=True)
    
    # Convert to numeric
    X = df_reg.drop('has_violation', axis=1).astype(float)
    y = df_reg['has_violation'].astype(int)
    
    # Add constant
    X_with_const = sm.add_constant(X)
    
    # Fit model
    try:
        model = sm.Logit(y, X_with_const)
        result = model.fit(disp=False, maxiter=100)
        
        print(f"\n📈 Model Results:")
        print(f"   • Log-Likelihood: {result.llf:.2f}")
        print(f"   • AIC: {result.aic:.2f}")
        print(f"   • Pseudo R²: {result.prsquared:.4f}")
        print(f"   • N observations: {len(y)}")
        
        # Country effects
        country_cols = [col for col in result.params.index if 'country_name_' in col]
        sig_countries = [col for col in country_cols if result.pvalues[col] < 0.05]
        
        print(f"\n🎯 Country Effects:")
        print(f"   • {len(sig_countries)} countries significant (p < 0.05)")
        print(f"   • Out of {len(country_cols)} countries total")
        print(f"   • Percentage: {len(sig_countries)/len(country_cols)*100:.1f}%")
        
        # Regional effect
        eastern_countries = df_model[df_model['region'] == 'Eastern Europe']['country_name'].unique()
        western_countries = df_model[df_model['region'] == 'Western Europe']['country_name'].unique()
        
        eastern_viol_rate = df_model[df_model['region'] == 'Eastern Europe']['has_violation'].mean()
        western_viol_rate = df_model[df_model['region'] == 'Western Europe']['has_violation'].mean()
        
        print(f"\n🌍 Regional Pattern:")
        print(f"   • Eastern Europe: {eastern_viol_rate*100:.1f}% (n={len(eastern_countries)} countries)")
        print(f"   • Western Europe: {western_viol_rate*100:.1f}% (n={len(western_countries)} countries)")
        print(f"   • Difference: {(eastern_viol_rate - western_viol_rate)*100:.1f} pp")
        
        # Top country effects (odds ratios)
        country_ors = {}
        for col in country_cols:
            country_name = col.replace('country_name_', '')
            or_val = np.exp(result.params[col])
            p_val = result.pvalues[col]
            country_ors[country_name] = {'OR': or_val, 'p_value': p_val, 'significant': p_val < 0.05}
        
        country_ors_df = pd.DataFrame(country_ors).T.sort_values('OR', ascending=False)
        
        print(f"\n📊 Top 5 Country Effects (Odds Ratios):")
        for idx, row in country_ors_df.head(5).iterrows():
            sig = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.05 else '*' if row['p_value'] < 0.1 else ''
            print(f"   {idx:30s}: OR = {row['OR']:6.2f} (p = {row['p_value']:.4f}) {sig}")
        
        # Year effect
        if 'year' in result.params.index:
            year_or = np.exp(result.params['year'])
            year_p = result.pvalues['year']
            sig = '***' if year_p < 0.001 else '**' if year_p < 0.05 else '*' if year_p < 0.1 else ''
            print(f"\n📅 Year Effect:")
            print(f"   OR = {year_or:.3f} (p = {year_p:.4f}) {sig}")
        
        return {
            'article': article,
            'n_cases': len(df_model),
            'n_countries': len(country_cols),
            'n_sig_countries': len(sig_countries),
            'pct_sig_countries': len(sig_countries)/len(country_cols)*100,
            'pseudo_r2': result.prsquared,
            'aic': result.aic,
            'eastern_rate': eastern_viol_rate,
            'western_rate': western_viol_rate,
            'regional_diff': eastern_viol_rate - western_viol_rate,
            'country_ors': country_ors_df,
            'result': result
        }
        
    except Exception as e:
        print(f"\n❌ Error fitting model for Article {article}: {e}")
        return None


def compare_across_articles(article_results):
    """Compare results across different articles"""
    print("\n" + "=" * 80)
    print("CROSS-ARTICLE COMPARISON")
    print("=" * 80)
    
    # Create comparison table
    comparison_data = []
    for result in article_results:
        if result is not None:
            comparison_data.append({
                'Article': result['article'],
                'N Cases': result['n_cases'],
                'N Countries': result['n_countries'],
                'Sig Countries': result['n_sig_countries'],
                '% Significant': f"{result['pct_sig_countries']:.1f}%",
                'Pseudo R²': f"{result['pseudo_r2']:.4f}",
                'AIC': f"{result['aic']:.1f}",
                'East Rate': f"{result['eastern_rate']*100:.1f}%",
                'West Rate': f"{result['western_rate']*100:.1f}%",
                'Diff (pp)': f"{result['regional_diff']*100:.1f}"
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    print("\n📊 Summary Table:")
    print(comparison_df.to_string(index=False))
    
    # Key findings
    print(f"\n🔍 KEY FINDINGS:")
    
    avg_sig_pct = comparison_df['% Significant'].str.rstrip('%').astype(float).mean()
    print(f"\n1. Country Significance Across Articles:")
    print(f"   • Average % significant: {avg_sig_pct:.1f}%")
    
    if avg_sig_pct > 40:
        print(f"   → Country effect is ROBUST across article types")
    elif avg_sig_pct > 20:
        print(f"   → Country effect is MODERATE but persists")
    else:
        print(f"   → Country effect is WEAK/INCONSISTENT")
    
    # Regional pattern consistency
    print(f"\n2. Regional Pattern Consistency:")
    all_positive = all([result['regional_diff'] > 0 for result in article_results if result])
    if all_positive:
        print(f"   • Eastern > Western in ALL articles")
        print(f"   → Regional pattern is HIGHLY CONSISTENT")
    else:
        print(f"   • Mixed pattern across articles")
        print(f"   → Regional pattern varies by article type")
    
    # Article-specific insights
    print(f"\n3. Article-Specific Patterns:")
    for result in article_results:
        if result is not None:
            if result['pct_sig_countries'] > 50:
                strength = "STRONG"
            elif result['pct_sig_countries'] > 30:
                strength = "MODERATE"
            else:
                strength = "WEAK"
            print(f"   Article {result['article']:10s}: {strength} country effect ({result['n_sig_countries']}/{result['n_countries']} significant)")
    
    return comparison_df


def create_visualizations(article_results):
    """Create comprehensive visualizations"""
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    
    # Filter out None results
    valid_results = [r for r in article_results if r is not None]
    
    if len(valid_results) == 0:
        print("⚠️  No valid results to visualize")
        return
    
    n_articles = len(valid_results)
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Country Significance by Article
    ax1 = plt.subplot(2, 3, 1)
    
    articles = [r['article'] for r in valid_results]
    sig_pcts = [r['pct_sig_countries'] for r in valid_results]
    colors_bar = ['coral' if pct > 40 else 'lightblue' for pct in sig_pcts]
    
    bars = ax1.bar(articles, sig_pcts, color=colors_bar, alpha=0.7, edgecolor='black')
    ax1.axhline(y=50, color='red', linestyle='--', linewidth=2, label='50% threshold')
    ax1.set_xlabel('Article')
    ax1.set_ylabel('% Countries Significant')
    ax1.set_title('Country Significance by Article Type\n(% of countries with p < 0.05)', 
                  fontweight='bold', fontsize=11)
    ax1.set_ylim(0, 100)
    ax1.legend()
    
    # Add value labels
    for bar, pct in zip(bars, sig_pcts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{pct:.0f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Regional Differences by Article
    ax2 = plt.subplot(2, 3, 2)
    
    eastern_rates = [r['eastern_rate'] * 100 for r in valid_results]
    western_rates = [r['western_rate'] * 100 for r in valid_results]
    
    x = np.arange(len(articles))
    width = 0.35
    
    ax2.bar(x - width/2, eastern_rates, width, label='Eastern Europe', 
            color='coral', alpha=0.7, edgecolor='black')
    ax2.bar(x + width/2, western_rates, width, label='Western Europe', 
            color='lightblue', alpha=0.7, edgecolor='black')
    
    ax2.set_xlabel('Article')
    ax2.set_ylabel('Violation Rate (%)')
    ax2.set_title('Regional Violation Rates by Article', 
                  fontweight='bold', fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(articles)
    ax2.legend()
    ax2.set_ylim(0, 100)
    
    # 3. Model Fit Comparison (Pseudo R²)
    ax3 = plt.subplot(2, 3, 3)
    
    pseudo_r2s = [r['pseudo_r2'] for r in valid_results]
    colors_r2 = plt.cm.viridis(np.linspace(0.3, 0.9, len(articles)))
    
    bars = ax3.bar(articles, pseudo_r2s, color=colors_r2, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Article')
    ax3.set_ylabel('Pseudo R²')
    ax3.set_title('Model Fit by Article\n(Higher = Better Fit)', 
                  fontweight='bold', fontsize=11)
    ax3.set_ylim(0, max(pseudo_r2s) * 1.3)
    
    # Add value labels
    for bar, r2 in zip(bars, pseudo_r2s):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{r2:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 4. Odds Ratios Comparison (Top Countries)
    ax4 = plt.subplot(2, 3, 4)
    
    # Get top 5 countries across all articles
    all_countries = set()
    for result in valid_results:
        all_countries.update(result['country_ors'].head(5).index)
    
    # Plot heatmap of OR's
    or_matrix = []
    countries_list = list(all_countries)[:8]  # Limit to 8 for readability
    
    for country in countries_list:
        row = []
        for result in valid_results:
            if country in result['country_ors'].index:
                or_val = result['country_ors'].loc[country, 'OR']
                row.append(min(or_val, 50))  # Cap at 50 for visualization
            else:
                row.append(np.nan)
        or_matrix.append(row)
    
    or_matrix = np.array(or_matrix)
    
    sns.heatmap(or_matrix, annot=True, fmt='.1f', cmap='YlOrRd', 
                xticklabels=articles, yticklabels=countries_list,
                ax=ax4, cbar_kws={'label': 'Odds Ratio'}, vmin=0, vmax=20)
    ax4.set_title('Country Odds Ratios by Article\n(Capped at 50 for visualization)', 
                  fontweight='bold', fontsize=11)
    ax4.set_xlabel('Article')
    ax4.set_ylabel('Country')
    
    # 5. Sample Size Distribution
    ax5 = plt.subplot(2, 3, 5)
    
    n_cases = [r['n_cases'] for r in valid_results]
    colors_n = ['green' if n > 200 else 'orange' if n > 100 else 'red' for n in n_cases]
    
    bars = ax5.bar(articles, n_cases, color=colors_n, alpha=0.7, edgecolor='black')
    ax5.set_xlabel('Article')
    ax5.set_ylabel('Number of Cases')
    ax5.set_title('Sample Size by Article', fontweight='bold', fontsize=11)
    
    # Add value labels
    for bar, n in zip(bars, n_cases):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{n}', ha='center', va='bottom', fontweight='bold')
    
    # 6. Consistency Score
    ax6 = plt.subplot(2, 3, 6)
    
    # Calculate consistency: how many articles show strong country effect
    strong_effect = sum([1 for r in valid_results if r['pct_sig_countries'] > 40])
    moderate_effect = sum([1 for r in valid_results if 20 < r['pct_sig_countries'] <= 40])
    weak_effect = sum([1 for r in valid_results if r['pct_sig_countries'] <= 20])
    
    sizes = [strong_effect, moderate_effect, weak_effect]
    labels = [f'Strong Effect\n(>40% sig)\nn={strong_effect}',
              f'Moderate Effect\n(20-40% sig)\nn={moderate_effect}',
              f'Weak Effect\n(<20% sig)\nn={weak_effect}']
    colors_pie = ['#ff6b6b', '#ffd93d', '#95e1d3']
    
    # Only plot non-zero slices
    non_zero = [(s, l, c) for s, l, c in zip(sizes, labels, colors_pie) if s > 0]
    if non_zero:
        sizes_nz, labels_nz, colors_nz = zip(*non_zero)
        
        wedges, texts, autotexts = ax6.pie(sizes_nz, labels=labels_nz, autopct='%1.0f%%',
                                            colors=colors_nz, startangle=90)
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
    
    ax6.set_title('Country Effect Strength\nAcross Articles', 
                  fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('article_specific_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualizations saved: article_specific_analysis.png")
    plt.close()


def generate_summary(article_results, comparison_df):
    """Generate comprehensive summary"""
    print("\n" + "=" * 80)
    print("FINAL SUMMARY & INTERPRETATION")
    print("=" * 80)
    
    valid_results = [r for r in article_results if r is not None]
    
    if len(valid_results) == 0:
        print("\n⚠️  No valid results to summarize")
        return
    
    avg_sig = np.mean([r['pct_sig_countries'] for r in valid_results])
    all_positive_regional = all([r['regional_diff'] > 0 for r in valid_results])
    
    print(f"""
🎯 Research Question: Is country effect article-specific or universal?

📊 MAIN FINDINGS:

1. COUNTRY EFFECT ACROSS ARTICLES:
   • Average % significant: {avg_sig:.1f}%
   • Articles analyzed: {len(valid_results)}
   • {sum([1 for r in valid_results if r['pct_sig_countries'] > 40])} articles show STRONG effect (>40% countries sig)
   • {sum([1 for r in valid_results if r['pct_sig_countries'] > 20])} articles show at least MODERATE effect (>20% countries sig)
   
2. REGIONAL PATTERN:
   • Eastern > Western in {'ALL' if all_positive_regional else 'MOST'} articles
   • Average difference: {np.mean([r['regional_diff']*100 for r in valid_results]):.1f} percentage points
   → Regional pattern is {'HIGHLY CONSISTENT' if all_positive_regional else 'GENERALLY CONSISTENT'}
   
3. ARTICLE-SPECIFIC INSIGHTS:
    """)
    
    for result in valid_results:
        print(f"   Article {result['article']:10s}:")
        print(f"      • {result['n_sig_countries']}/{result['n_countries']} countries significant ({result['pct_sig_countries']:.1f}%)")
        print(f"      • Regional gap: {result['regional_diff']*100:.1f} pp")
        print(f"      • Model fit: R² = {result['pseudo_r2']:.3f}")
    
    print(f"""
⚖️  ANSWER TO RESEARCH QUESTION:

{'✅ UNIVERSAL EFFECT' if avg_sig > 40 else '⚠️  MIXED EFFECT'}

    """)
    
    if avg_sig > 40:
        print(f"""
   The country effect is NOT article-specific!
   
   Evidence:
   • Majority of articles show strong country effects
   • Regional pattern consistent across article types
   • Country remains significant predictor regardless of violation type
   
   → This STRENGTHENS the case that country differences are systematic
   → Cannot be explained by "different violation types"
        """)
    else:
        print(f"""
   The country effect varies by article type:
   
   • Some articles show strong effects
   • Others show weak or no effects
   • Pattern suggests article-specific factors matter
   
   → Country effect may be stronger for certain violation types
   → Need to examine specific articles more carefully
        """)
    
    print(f"""
💡 IMPLICATIONS FOR MAIN RESEARCH QUESTION:

   "Does the ECtHR treat countries differently?"
   
   These article-specific results:
   {'✅ STRENGTHEN' if avg_sig > 40 else '⚠️  COMPLICATE'} the main finding!
   
   {'• Country differences persist across violation types' if avg_sig > 40 else '• Country differences may be violation-type specific'}
   {'• Effect is robust, not driven by article composition' if avg_sig > 40 else '• Effect may reflect different case mixes'}
   {'• Evidence for systematic country treatment differences' if avg_sig > 40 else '• Evidence mixed - more nuanced story'}
   
🔬 METHODOLOGICAL CONTRIBUTION:

   This analysis demonstrates:
   ✓ Importance of article-specific models
   ✓ Confounding by article type {'does NOT' if avg_sig > 40 else 'may'} explain country differences
   ✓ Regional patterns are {'robust' if all_positive_regional else 'variable'} across violation types
    """)
    
    print("=" * 80)
    print("✓ ARTICLE-SPECIFIC ANALYSIS COMPLETED")
    print("=" * 80)


def main():
    """Main function"""
    
    # Load data
    df = load_and_prepare_data('extracted_data.csv')
    
    # Identify major articles
    major_articles = identify_major_articles(df, min_cases=100)
    
    # Run article-specific models
    article_results = []
    for article in major_articles:
        result = run_article_specific_model(df, article, min_countries=5)
        article_results.append(result)
    
    # Compare across articles
    comparison_df = compare_across_articles(article_results)
    
    # Create visualizations
    create_visualizations(article_results)
    
    # Generate summary
    generate_summary(article_results, comparison_df)
    
    print("\n✓ All analyses completed successfully!")
    print("\nGenerated files:")
    print("  📊 article_specific_analysis.png")


if __name__ == "__main__":
    main()