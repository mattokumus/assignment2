# 🚀 Usage Guide - Step by Step

This guide walks you through running the complete ECHR analysis pipeline from start to finish.

**Repository:** [https://github.com/mattokumus/assignment2](https://github.com/mattokumus/assignment2)

---

## 📋 Prerequisites

### Required Software
- **Python 3.8 or higher** ([Download](https://www.python.org/downloads/))
- **Git** ([Download](https://git-scm.com/downloads))
- **Git LFS** (for large data files) ([Download](https://git-lfs.github.com/))

### Check Your Python Version
```bash
python3 --version
# Should show: Python 3.8.x or higher
```

---

## 🔧 Step 1: Clone Repository & Install Dependencies

```bash
# Clone the repository
git clone https://github.com/mattokumus/assignment2.git
cd assignment2

# Pull large files (cases-2000.json)
git lfs pull

# Install Python dependencies
pip install -r requirements.txt

# Verify installation
python3 -c "import pandas, numpy, scipy, statsmodels, sklearn, matplotlib, seaborn, plotly, xgboost; print('✅ All packages installed successfully!')"
```

**Expected output:** `✅ All packages installed successfully!`

---

## 📊 Step 2: Run the Complete Analysis Pipeline

### Stage 0 (Optional): JSON Documentation

**Purpose:** Understand the structure of raw ECHR case data.

```bash
python3 jsondocumenting.py
```

**Outputs:**
- `cases-2000_documentation.md` - Field descriptions, examples, types
- `cases-2000_schema.json` - JSON schema specification

**Do you need this?** Only if you want to understand the raw data structure. Skip if you just want to run the analysis.

---

### Stage 1: Data Extraction & Preprocessing

**Purpose:** Convert JSON to structured CSV, filter to substantive cases only.

```bash
python3 data_extraction.py
```

**Outputs:**
- `extracted_data.csv` - 2,000 substantive cases (violation/no-violation only)

**What it does:**
- Extracts relevant variables: country, articles, year, judges, applicant type
- **Critical filtering:** Excludes procedural decisions (inadmissible, struck out)
- Only keeps merit-based judgments (violation vs. no-violation)

**Runtime:** ~5-10 seconds

---

### Stage 2: Exploratory Data Analysis (EDA)

**Purpose:** Understand patterns in the data before statistical modeling.

```bash
python3 eda_analysis.py
```

**Outputs:**
- `eda_visualizations.png` - 6 static charts (country distributions, temporal trends)
- `eda_heatmap.png` - Countries × Decades violation rate heatmap
- `eda_correlation.png` - Variable correlation matrix
- `eda_interactive.html` - **Interactive dashboard** 🎯 (open in browser!)

**What to look at:**
- Top 15 countries by case count
- Violation rates: Eastern vs. Western Europe
- Temporal trends (1968-2020)
- Most violated articles

**Runtime:** ~20-30 seconds

**🌐 View Interactive Dashboard:**
```bash
# macOS
open eda_interactive.html

# Linux
xdg-open eda_interactive.html

# Windows
start eda_interactive.html
```

---

### Stage 3: Hypothesis Testing

**Purpose:** Test if country-outcome associations are statistically significant.

```bash
python3 hypotesis_testing.py
```

**Outputs:**
- `hypothesis_test_visualizations.png` - 6 static charts
- `hypothesis_test_interactive.html` - **Interactive dashboard** 🎯

**Tests performed:**
1. **Chi-square test:** Country × Violation independence test
2. **Proportion tests:** Compare violation rates between country pairs
3. **Regional comparison:** Eastern vs. Western Europe (p-values)
4. **Temporal analysis:** Before vs. After 2000

**What to look at:**
- Chi-square p-value (< 0.001 = highly significant)
- Cramér's V effect size
- Regional gap: Eastern Europe +21.6 pp higher violation rate

**Runtime:** ~30-40 seconds

---

### Stage 4: Logistic Regression

**Purpose:** Do country effects persist after controlling for confounders?

```bash
python3 logistic_regression.py
```

**Outputs:**
- `logistic_regression_analysis.png` - 6 static charts
- `logistic_regression_interactive.html` - **Interactive dashboard** 🎯

**Models:**
1. **Baseline:** `violation ~ country` (no controls)
2. **Full model:** `violation ~ country + article + year + applicant_type`
3. **Regional model:** `violation ~ region + controls`

**What to look at:**
- Odds ratios (OR > 1 = higher violation rate vs. reference country)
- P-values (< 0.05 = significant)
- Pseudo R² (model fit)
- How many countries remain significant after controls?

**Key finding:** 56.2% of countries remain significant after controls!

**Runtime:** ~1-2 minutes

**⚠️ Note:** Only countries with ≥30 cases included (for statistical reliability).

---

### Stage 5: Judge-Level Analysis

**Purpose:** Are country effects due to judge assignment or systematic patterns?

```bash
python3 judge_analysis.py
```

**Outputs:**
- `judge_analysis_visualizations.png` - 6 static charts
- `judge_analysis_interactive.html` - **Interactive dashboard** 🎯

**Key questions:**
1. Do individual judges vary in violation rates?
2. Do judges treat Eastern vs. Western Europe differently?
3. Do country effects persist after controlling for judge identity?

**What to look at:**
- Judge violation rate distribution (171 judges analyzed)
- Regional bias: Average +29.1 pp for Eastern Europe across judges
- Model comparison: With vs. without judge controls

**Key finding:** Country effects persist even after controlling for judges!

**Runtime:** ~1-2 minutes

---

### Stage 6: Machine Learning Validation

**Purpose:** Can we predict violations based on country (and other features)?

```bash
python3 ml_models_comparison.py
```

**Outputs:**
- `ml_models_comparison.png` - 9 static charts
- `ml_models_interactive.html` - **Interactive dashboard** 🎯

**Models compared:**
1. Logistic Regression (baseline)
2. Random Forest
3. XGBoost
4. Gradient Boosting

**Validation strategies:**
- **Random split:** 80/20 train/test (stratified)
- **Temporal split:** Train on 1968-2014, test on 2015-2020

**What to look at:**
- ROC-AUC scores (higher = better prediction)
- Feature importance (which variables matter most?)
- Random vs. temporal performance (is the pattern stable over time?)

**Key finding:** 89% accuracy, AUC = 0.801 in predicting violations!

**Runtime:** ~2-3 minutes

---

## 🎯 Quick Summary: What Each Script Does

| Script | Purpose | Key Output | Runtime |
|--------|---------|------------|---------|
| `jsondocumenting.py` | Document JSON structure | Schema documentation | 5 sec |
| `data_extraction.py` | JSON → CSV, filter substantive cases | `extracted_data.csv` | 10 sec |
| `eda_analysis.py` | Explore patterns | Interactive dashboard | 30 sec |
| `hypotesis_testing.py` | Statistical significance tests | Chi-square, proportion tests | 40 sec |
| `logistic_regression.py` | Country effects with controls | Odds ratios, model fit | 2 min |
| `judge_analysis.py` | Test judge vs. country effects | Judge-country disentangling | 2 min |
| `ml_models_comparison.py` | Predictive modeling | Model comparison, AUC scores | 3 min |

**Total runtime:** ~8-10 minutes for complete pipeline

---

## 📁 Output Files Summary

### Data Files
- `extracted_data.csv` - Processed dataset (2,000 cases)

### Static Visualizations (PNG)
- `eda_visualizations.png`
- `eda_heatmap.png`
- `eda_correlation.png`
- `hypothesis_test_visualizations.png`
- `logistic_regression_analysis.png`
- `judge_analysis_visualizations.png`
- `ml_models_comparison.png`

### Interactive Dashboards (HTML) 🎯
- `eda_interactive.html`
- `hypothesis_test_interactive.html`
- `logistic_regression_interactive.html`
- `judge_analysis_interactive.html`
- `ml_models_interactive.html`

**💡 Tip:** Interactive HTML files work offline! Just double-click to open in browser.

---

## 🔍 How to Interpret Results

### Violation Rates
- **Eastern Europe:** 93.9% violation rate
- **Western Europe:** 72.2% violation rate
- **Gap:** +21.6 percentage points (p < 0.001)

### Logistic Regression Odds Ratios (OR)
- **OR = 1:** No effect (same as reference country)
- **OR > 1:** Higher violation rate than reference
- **OR < 1:** Lower violation rate than reference
- **Example:** Ukraine OR = 32.45 → 32× higher odds of violation vs. reference

### Statistical Significance
- **p < 0.001:** Highly significant (★★★)
- **p < 0.01:** Very significant (★★)
- **p < 0.05:** Significant (★)
- **p ≥ 0.05:** Not significant

### Model Performance (ML)
- **Accuracy:** % of correct predictions
- **Precision:** When model says "violation", how often is it correct?
- **Recall:** Of all actual violations, how many did model catch?
- **AUC-ROC:** Overall discrimination ability (0.5 = random, 1.0 = perfect)

---

## ❓ Troubleshooting

### "ModuleNotFoundError: No module named 'X'"
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### "FileNotFoundError: cases-2000.json"
```bash
# Pull large files with Git LFS
git lfs pull
```

### Python version error
```bash
# Check version
python3 --version

# If < 3.8, update Python:
# macOS: brew install python@3.11
# Ubuntu: sudo apt install python3.11
# Windows: Download from python.org
```

### Out of memory error
```bash
# Reduce sample size in scripts (not recommended for publication)
# Or run on a machine with more RAM (8GB+ recommended)
```

---

## 📧 Questions or Issues?

- **GitHub Issues:** [https://github.com/mattokumus/assignment2/issues](https://github.com/mattokumus/assignment2/issues)
- **Documentation:** See `README.md`, `METHODOLOGY.md`, `DATA_PROVENANCE.md`
- **Research Design:** See `RESEARCH_DESIGN.md` (Task 2.1)

---

## 📜 Citation

If you use this code or methodology, please cite:

```
[Your Name] (2024). Does the European Court of Human Rights Treat Countries
Differently? A Statistical Analysis of 2,000 Cases (1968-2020).
GitHub repository: https://github.com/mattokumus/assignment2
```

---

**Last Updated:** November 2025

**Status:** ✅ Complete - All scripts tested and working
