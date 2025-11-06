#!/usr/bin/env python3
"""
Machine Learning Models Comparison for ECHR Cases
Research Question: Which ML model best predicts ECHR violation outcomes?

Models Compared:
1. Logistic Regression (baseline)
2. Random Forest
3. XGBoost
4. Gradient Boosting

Evaluation: 5-fold cross-validation with multiple metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, f1_score, precision_score, recall_score, accuracy_score
)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Regional classification constants
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


def load_and_prepare_data(filename='extracted_data.csv'):
    """Load and prepare data for ML modeling"""
    print("=" * 80)
    print("MACHINE LEARNING MODELS COMPARISON: ECHR CASES")
    print("=" * 80)

    df = pd.read_csv(filename)
    print(f"\n✓ Data loaded: {len(df)} cases")

    # Add regional classification
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

    # Calculate case age
    df['case_age'] = 2024 - df['year']

    print(f"✓ Features engineered")
    print(f"  - Countries: {df['country_name'].nunique()}")
    print(f"  - Primary articles: {df['primary_article'].nunique()}")
    print(f"  - Applicant types: {df['applicant_type'].nunique()}")
    print(f"  - Year range: {df['year'].min()} - {df['year'].max()}")

    return df


def prepare_features(df):
    """
    Prepare features for ML models
    Returns: X (features), y (target), feature_names, df_model (for temporal split)
    """
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING")
    print("=" * 80)

    # Select features for modeling
    # NOTE: Excluding violation_count and no_violation_count to avoid data leakage
    # (these are derived from the target variable has_violation)
    features_to_use = ['country_name', 'region', 'primary_article', 'year',
                       'applicant_type', 'judge_count']

    # Create a copy for modeling
    df_model = df[features_to_use + ['has_violation']].copy()

    # Filter to include only countries with sufficient cases (min 20)
    country_counts = df_model['country_name'].value_counts()
    valid_countries = country_counts[country_counts >= 20].index
    df_model = df_model[df_model['country_name'].isin(valid_countries)]

    print(f"✓ Filtered to {len(valid_countries)} countries with ≥20 cases")
    print(f"✓ Dataset size: {len(df_model)} cases")

    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(
        df_model,
        columns=['country_name', 'region', 'primary_article', 'applicant_type'],
        drop_first=True  # Avoid multicollinearity
    )

    # Separate features and target
    X = df_encoded.drop('has_violation', axis=1)
    y = df_encoded['has_violation'].astype(int)

    print(f"✓ One-hot encoding completed")
    print(f"  - Total features: {X.shape[1]}")
    print(f"  - Target distribution: {y.value_counts().to_dict()}")
    print(f"  - Violation rate: {y.mean():.1%}")

    return X, y, X.columns.tolist(), df_model


def train_models(X_train, X_test, y_train, y_test):
    """
    Train multiple ML models with given train/test split
    """
    print(f"\n✓ Training/Test split:")
    print(f"  - Training set: {len(X_train)} cases ({len(X_train)/(len(X_train)+len(X_test))*100:.1f}%)")
    print(f"  - Test set: {len(X_test)} cases ({len(X_test)/(len(X_train)+len(X_test))*100:.1f}%)")

    # Define models with pipelines (scaling for logistic regression)
    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                max_iter=5000,
                random_state=42,
                class_weight='balanced',
                solver='saga'
            ))
        ]),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
    }

    # Train each model
    trained_models = {}
    for name, model in models.items():
        print(f"\n⏳ Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"   ✓ {name} trained")

    return trained_models


def evaluate_models(models, X_train, X_test, y_train, y_test, X, y):
    """
    Evaluate all models using cross-validation and test set
    """
    print("\n" + "=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)

    results = []

    # Cross-validation on full dataset
    print("\n📊 5-Fold Cross-Validation Results:")
    print("-" * 80)
    print(f"{'Model':<25} {'ROC-AUC':<12} {'F1-Score':<12} {'Precision':<12} {'Recall':<12}")
    print("-" * 80)

    for name, model in models.items():
        # Cross-validation
        scoring = ['roc_auc', 'f1', 'precision', 'recall', 'accuracy']
        cv_results = cross_validate(model, X, y, cv=5, scoring=scoring, n_jobs=-1)

        # Store results
        result = {
            'model': name,
            'roc_auc_mean': cv_results['test_roc_auc'].mean(),
            'roc_auc_std': cv_results['test_roc_auc'].std(),
            'f1_mean': cv_results['test_f1'].mean(),
            'f1_std': cv_results['test_f1'].std(),
            'precision_mean': cv_results['test_precision'].mean(),
            'precision_std': cv_results['test_precision'].std(),
            'recall_mean': cv_results['test_recall'].mean(),
            'recall_std': cv_results['test_recall'].std(),
            'accuracy_mean': cv_results['test_accuracy'].mean(),
            'accuracy_std': cv_results['test_accuracy'].std(),
        }
        results.append(result)

        # Print cross-validation results
        print(f"{name:<25} {result['roc_auc_mean']:.3f}±{result['roc_auc_std']:.3f}  "
              f"{result['f1_mean']:.3f}±{result['f1_std']:.3f}  "
              f"{result['precision_mean']:.3f}±{result['precision_std']:.3f}  "
              f"{result['recall_mean']:.3f}±{result['recall_std']:.3f}")

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Find best model
    best_model_name = results_df.loc[results_df['roc_auc_mean'].idxmax(), 'model']
    best_roc_auc = results_df['roc_auc_mean'].max()

    print("-" * 80)
    print(f"\n🏆 Best Model: {best_model_name} (ROC-AUC: {best_roc_auc:.3f})")

    # Test set evaluation
    print("\n" + "=" * 80)
    print("TEST SET PERFORMANCE")
    print("=" * 80)

    test_results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        test_results[name] = {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'f1': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'accuracy': accuracy_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

        print(f"\n{name}:")
        print(f"  ROC-AUC:   {test_results[name]['roc_auc']:.3f}")
        print(f"  F1-Score:  {test_results[name]['f1']:.3f}")
        print(f"  Precision: {test_results[name]['precision']:.3f}")
        print(f"  Recall:    {test_results[name]['recall']:.3f}")
        print(f"  Accuracy:  {test_results[name]['accuracy']:.3f}")

    return results_df, test_results


def get_feature_importance(models, feature_names):
    """
    Extract feature importance from tree-based models
    """
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)

    importance_data = {}

    for name, model in models.items():
        # Handle pipeline models
        if hasattr(model, 'named_steps'):
            # Extract the actual classifier from pipeline
            actual_model = model.named_steps['classifier']
        else:
            actual_model = model

        if hasattr(actual_model, 'feature_importances_'):
            # Tree-based models
            importance_data[name] = pd.DataFrame({
                'feature': feature_names,
                'importance': actual_model.feature_importances_
            }).sort_values('importance', ascending=False).head(15)

            print(f"\n{name} - Top 10 Features:")
            for idx, row in importance_data[name].head(10).iterrows():
                print(f"  {row['feature']:<50} {row['importance']:.4f}")

        elif hasattr(actual_model, 'coef_'):
            # Logistic Regression
            importance_data[name] = pd.DataFrame({
                'feature': feature_names,
                'importance': np.abs(actual_model.coef_[0])
            }).sort_values('importance', ascending=False).head(15)

            print(f"\n{name} - Top 10 Features (|coefficient|):")
            for idx, row in importance_data[name].head(10).iterrows():
                print(f"  {row['feature']:<50} {row['importance']:.4f}")

    return importance_data


def create_visualizations(models, results_df, test_results, importance_data,
                         X_test, y_test, X, y):
    """
    Create comprehensive visualizations
    """
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)

    fig = plt.figure(figsize=(20, 16))

    # 1. ROC Curves Comparison
    ax1 = plt.subplot(3, 3, 1)
    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = test_results[name]['roc_auc']
        ax1.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})', linewidth=2)

    ax1.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.500)', linewidth=1)
    ax1.set_xlabel('False Positive Rate', fontsize=11)
    ax1.set_ylabel('True Positive Rate', fontsize=11)
    ax1.set_title('ROC Curves Comparison', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. Cross-Validation Metrics Comparison
    ax2 = plt.subplot(3, 3, 2)
    metrics = ['roc_auc_mean', 'f1_mean', 'precision_mean', 'recall_mean']
    metric_labels = ['ROC-AUC', 'F1-Score', 'Precision', 'Recall']
    x = np.arange(len(metrics))
    width = 0.2

    for i, (_, row) in enumerate(results_df.iterrows()):
        values = [row[m] for m in metrics]
        ax2.bar(x + i*width, values, width, label=row['model'], alpha=0.8)

    ax2.set_xlabel('Metrics', fontsize=11)
    ax2.set_ylabel('Score', fontsize=11)
    ax2.set_title('Cross-Validation Performance Comparison', fontsize=13, fontweight='bold')
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(metric_labels, fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 1])

    # 3. Model Ranking
    ax3 = plt.subplot(3, 3, 3)
    results_sorted = results_df.sort_values('roc_auc_mean', ascending=True)
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(results_sorted)))
    bars = ax3.barh(results_sorted['model'], results_sorted['roc_auc_mean'], color=colors)
    ax3.set_xlabel('ROC-AUC Score', fontsize=11)
    ax3.set_title('Model Ranking (Cross-Validation)', fontsize=13, fontweight='bold')
    ax3.set_xlim([0, 1])
    for i, (idx, row) in enumerate(results_sorted.iterrows()):
        ax3.text(row['roc_auc_mean'] + 0.01, i, f"{row['roc_auc_mean']:.3f}",
                va='center', fontsize=10)
    ax3.grid(True, alpha=0.3, axis='x')

    # 4-7. Confusion Matrices
    for idx, (name, model) in enumerate(models.items()):
        ax = plt.subplot(3, 3, 4 + idx)
        cm = test_results[name]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                   xticklabels=['No Violation', 'Violation'],
                   yticklabels=['No Violation', 'Violation'])
        ax.set_title(f'{name}\nAccuracy: {test_results[name]["accuracy"]:.3f}',
                    fontsize=11, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=10)
        ax.set_xlabel('Predicted Label', fontsize=10)

    # 8. Feature Importance (Best Tree-based Model)
    ax8 = plt.subplot(3, 3, 8)
    best_tree_model = None
    best_tree_auc = 0
    for name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            if test_results[name]['roc_auc'] > best_tree_auc:
                best_tree_model = name
                best_tree_auc = test_results[name]['roc_auc']

    if best_tree_model:
        imp_data = importance_data[best_tree_model].head(15)
        colors_imp = plt.cm.viridis(np.linspace(0.3, 0.9, len(imp_data)))
        ax8.barh(range(len(imp_data)), imp_data['importance'], color=colors_imp)
        ax8.set_yticks(range(len(imp_data)))
        ax8.set_yticklabels(imp_data['feature'], fontsize=8)
        ax8.set_xlabel('Importance', fontsize=11)
        ax8.set_title(f'Top 15 Features - {best_tree_model}', fontsize=13, fontweight='bold')
        ax8.invert_yaxis()
        ax8.grid(True, alpha=0.3, axis='x')

    # 9. Test Set Performance Comparison
    ax9 = plt.subplot(3, 3, 9)
    test_metrics = ['roc_auc', 'f1', 'precision', 'recall']
    test_metric_labels = ['ROC-AUC', 'F1', 'Precision', 'Recall']
    x = np.arange(len(test_metrics))
    width = 0.2

    for i, (name, result) in enumerate(test_results.items()):
        values = [result[m] for m in test_metrics]
        ax9.bar(x + i*width, values, width, label=name, alpha=0.8)

    ax9.set_xlabel('Metrics', fontsize=11)
    ax9.set_ylabel('Score', fontsize=11)
    ax9.set_title('Test Set Performance Comparison', fontsize=13, fontweight='bold')
    ax9.set_xticks(x + width * 1.5)
    ax9.set_xticklabels(test_metric_labels, fontsize=9)
    ax9.legend(fontsize=8)
    ax9.grid(True, alpha=0.3, axis='y')
    ax9.set_ylim([0, 1])

    plt.suptitle('Machine Learning Models Comparison - ECHR Case Violation Prediction',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_file = 'ml_models_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved: {output_file}")
    plt.close()


def main():
    """Main execution function with both random and temporal splits"""

    # Load and prepare data
    df = load_and_prepare_data()

    # Prepare features
    X, y, feature_names, df_model = prepare_features(df)

    # ========================================================================
    # STRATEGY 1: RANDOM SPLIT (Stratified)
    # ========================================================================
    print("\n" + "=" * 80)
    print("STRATEGY 1: RANDOM SPLIT (STRATIFIED)")
    print("All years mixed - standard ML approach")
    print("=" * 80)

    # Random split
    X_train_random, X_test_random, y_train_random, y_test_random = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train models
    print("\n" + "=" * 80)
    print("MODEL TRAINING - RANDOM SPLIT")
    print("=" * 80)
    models_random = train_models(X_train_random, X_test_random, y_train_random, y_test_random)

    # Evaluate models
    results_random, test_results_random = evaluate_models(
        models_random, X_train_random, X_test_random, y_train_random, y_test_random, X, y
    )

    # ========================================================================
    # STRATEGY 2: TEMPORAL SPLIT (2015 cutoff)
    # ========================================================================
    print("\n" + "=" * 80)
    print("STRATEGY 2: TEMPORAL SPLIT (2015 CUTOFF)")
    print("Train: 2000-2014, Test: 2015-2020 - realistic generalization")
    print("=" * 80)

    # Get year information from original df_model (before encoding)
    # We need to align indices
    year_series = df_model['year']

    # Create temporal split
    train_mask = year_series < 2015
    test_mask = year_series >= 2015

    X_train_temporal = X[train_mask]
    X_test_temporal = X[test_mask]
    y_train_temporal = y[train_mask]
    y_test_temporal = y[test_mask]

    print(f"\n✓ Temporal split:")
    print(f"  - Training: years < 2015 → {len(X_train_temporal)} cases")
    print(f"  - Test: years ≥ 2015 → {len(X_test_temporal)} cases")
    print(f"  - Train violation rate: {y_train_temporal.mean()*100:.1f}%")
    print(f"  - Test violation rate: {y_test_temporal.mean()*100:.1f}%")

    # Train models
    print("\n" + "=" * 80)
    print("MODEL TRAINING - TEMPORAL SPLIT")
    print("=" * 80)
    models_temporal = train_models(X_train_temporal, X_test_temporal, y_train_temporal, y_test_temporal)

    # Evaluate models
    results_temporal, test_results_temporal = evaluate_models(
        models_temporal, X_train_temporal, X_test_temporal, y_train_temporal, y_test_temporal, X, y
    )

    # ========================================================================
    # COMPARISON
    # ========================================================================
    print("\n" + "=" * 80)
    print("SPLIT STRATEGY COMPARISON")
    print("=" * 80)

    print("\n📊 Random Split vs Temporal Split (Test Set ROC-AUC):")
    print("-" * 80)
    print(f"{'Model':<25} {'Random Split':<15} {'Temporal Split':<15} {'Difference':<15}")
    print("-" * 80)

    for model_name in test_results_random.keys():
        random_auc = test_results_random[model_name]['roc_auc']
        temporal_auc = test_results_temporal[model_name]['roc_auc']
        diff = temporal_auc - random_auc
        print(f"{model_name:<25} {random_auc:<15.3f} {temporal_auc:<15.3f} {diff:+.3f}")

    print("-" * 80)

    # Get feature importance (using random split models)
    importance_data = get_feature_importance(models_random, feature_names)

    # Create visualizations (using random split for main viz)
    create_visualizations(models_random, results_random, test_results_random, importance_data,
                         X_test_random, y_test_random, X, y)

    # Final summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\n📊 Key Findings:")
    print(f"  - Best model (Random Split): {results_random.loc[results_random['roc_auc_mean'].idxmax(), 'model']}")
    print(f"  - Best CV ROC-AUC: {results_random['roc_auc_mean'].max():.3f}")
    print(f"  - Best model (Temporal Split): {results_temporal.loc[results_temporal['roc_auc_mean'].idxmax(), 'model']}")
    print(f"  - Best CV ROC-AUC: {results_temporal['roc_auc_mean'].max():.3f}")
    print(f"  - Total features: {len(feature_names)}")
    print(f"  - Dataset size: {len(X)} cases")
    print(f"\n✓ Output saved: ml_models_comparison.png")
    print("\n💡 Interpretation:")
    print("  - Random split: Optimistic performance (all years mixed)")
    print("  - Temporal split: Realistic generalization (past→future prediction)")
    print("  - Small differences suggest model generalizes well across time")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
