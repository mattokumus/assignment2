# Research Design: Does the ECtHR Discriminate Against Certain Countries?

## Task 2.1 - Research Design (250 words)

**Research Question:** To what extent does respondent country predict European Court of Human Rights case outcomes, independent of legal, temporal, and judicial confounders?

**Methodology:** This study employs a multi-method triangulation strategy combining exploratory analysis, statistical hypothesis testing, regression modeling, judge-level analysis, and machine learning.

**Data & Sample:** Analysis utilizes 1,904 substantive ECHR judgments (1968-2020), excluding procedural decisions (inadmissible/struck out) to focus on merit-based outcomes. The binary outcome distinguishes violations found versus no violations. Key predictors include respondent country (45 countries), regional classification (Eastern/Western Europe), violated ECHR articles, judgment year, applicant type, and judicial panel composition (president identity, judge names, panel size).

**Analytical Pipeline:**

1. **Data Documentation** (jsondocumenting.py): Systematic documentation of JSON structure, generating schema specifications and field mappings to understand all available case properties (country, articles, judges, conclusions, dates).

2. **Data Extraction & Preprocessing** (data_extraction.py): Converts JSON to structured CSV format, extracting relevant variables (country, articles, year, applicant type, judicial panel). Critically filters to substantive decisions only (violation/no-violation), excluding procedural outcomes (inadmissible, struck out) to avoid confounding merit-based judgments with jurisdictional decisions.

3. **Exploratory Data Analysis** (eda_analysis.py): Identifies cross-country variation in violation rates, temporal patterns, and article-specific trends through descriptive statistics and visualizations.

4. **Hypothesis Testing** (hypotesis_testing.py): Chi-square and proportion tests assess statistical significance of country-outcome associations. Regional comparisons (Eastern vs. Western Europe) and temporal analysis (before/after 2000).

5. **Logistic Regression** (logistic_regression.py): Hierarchical models isolate country effects after controlling for article type, temporal trends, and applicant characteristics.

6. **Judge-Level Analysis** (judge_analysis.py): Tests whether country effects persist after controlling for panel composition (president identity, judge names), distinguishing systematic patterns from judge-specific variation.

7. **Machine Learning Validation** (ml_models_comparison.py): Random Forest, XGBoost, and Gradient Boosting validate findings through 5-fold cross-validation and temporal validation (1968-2014 train, 2015-2020 test).

**Robustness Checks:** Minimum sample thresholds (≥30 cases/country), regional aggregation, temporal split validation, and judge fixed-effects ensure finding robustness. Statistical association cannot definitively prove discrimination, as unmeasured confounders (case complexity, representation quality) may explain observed patterns.

