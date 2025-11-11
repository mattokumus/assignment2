# Research Design: Does the ECtHR Discriminate Against Certain Countries?

## Task 2.1 - Research Design (250 words)

**Research Question:** To what extent does respondent country predict European Court of Human Rights case outcomes, independent of legal, temporal, and judicial confounders?

**Methodology:** This study employs a multi-method triangulation strategy combining exploratory analysis, statistical hypothesis testing, regression modeling, judge-level analysis, and machine learning.

**Data & Sample:** Analysis utilizes 2,000 substantive ECHR judgments (1968-2020), excluding procedural decisions (inadmissible/struck out) to focus on merit-based outcomes. The binary outcome distinguishes violations found versus no violations. Key predictors include respondent country (45 countries), regional classification (Eastern/Western Europe), violated ECHR articles, judgment year, applicant type, and judicial panel composition (president identity, judge names, panel size).

**Analytical Pipeline:**

1. **Exploratory Data Analysis** (eda_analysis.py): Identifies cross-country variation in violation rates, temporal patterns, and article-specific trends through descriptive statistics and visualizations.

2. **Hypothesis Testing** (hypotesis_testing.py): Chi-square tests and proportion tests assess statistical significance of country-outcome associations. Regional comparisons (Eastern vs. Western Europe) and temporal analysis (before/after 2000).

3. **Logistic Regression** (logistic_regression.py): Hierarchical models isolate country effects after controlling for article type, temporal trends, and applicant characteristics. Country fixed effects tested with and without controls.

4. **Judge-Level Analysis** (judge_analysis.py): Critical robustness check testing whether country effects persist after controlling for panel composition, distinguishing systematic institutional patterns from judge-specific variation.

5. **Machine Learning Validation** (ml_models_comparison.py): Random Forest and XGBoost models validate findings through cross-validation and temporal validation (pre-2015 training, post-2015 testing), assessing pattern stability.

**Robustness Checks:** Minimum sample thresholds (≥30 cases/country), regional aggregation, temporal split validation, and judge fixed-effects ensure finding robustness. Statistical association cannot definitively prove discrimination, as unmeasured confounders (case complexity, representation quality) may explain observed patterns.

---

**Word count:** 250 words
