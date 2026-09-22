# Comment Category Prediction — Multi-Class Toxicity Classification

A multi-class text classification pipeline that predicts the toxicity level of user comments (non-toxic vs. three levels of toxicity), combining TF-IDF text features, engineered metadata, and a tuned soft-voting ensemble of linear and gradient-boosted models. Built for the **Comment Category Prediction Challenge** (Kaggle).

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Approach](#approach)
- [Feature Engineering](#feature-engineering)
- [Modeling](#modeling)
- [Results](#results)
- [Key Insights](#key-insights)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Future Work](#future-work)
- [License](#license)

---

## Problem Statement

Given a user-generated comment along with engagement metadata (upvotes/downvotes, reaction emoticons, timestamps, and self-reported demographic attributes), predict which of **4 toxicity categories** it belongs to:

| Label | Meaning |
|---|---|
| 0 | Non-toxic |
| 1 | Mildly toxic |
| 2 | Moderately toxic |
| 3 | Severely toxic |

## Dataset

- **Train:** 198,000 rows × 15 columns · **Test:** 102,000 rows × 14 columns
- Each row combines a raw `comment` string with structured metadata: `upvote`, `downvote`, three `emoticon_*` reaction counts, two anonymized indicator flags (`if_1`, `if_2`), self-reported `race` / `religion` / `gender` (each ~73% missing and imputed as `"unknown"`), a `disability` flag, and a `created_date` timestamp.
- **Class imbalance is severe:** non-toxic comments (label 0) make up ~57.7% of the data, while the most severe toxicity class (label 3) accounts for only ~2.8% — this drove the choice of **macro F1-score** as the primary evaluation metric over raw accuracy.

## Approach

1. **Clean and engineer features** from the raw comment text and surrounding metadata.
2. **Build a unified preprocessing pipeline** (`ColumnTransformer`) that vectorizes text, scales numeric features, and one-hot encodes categoricals in a single, leakage-safe step.
3. **Benchmark a broad model zoo** — from a naive baseline through linear models, Naive Bayes, KNN, SVM, and gradient boosters — on a stratified train/validation split.
4. **Tune and blend the strongest models** into a weighted soft-voting ensemble, optimizing blend weights directly against macro F1 on the validation set.
5. **Hyperparameter-search the best single model** (LightGBM) with `RandomizedSearchCV`.
6. **Retrain the final tuned ensemble on 100% of the training data** and generate test-set predictions.

## Feature Engineering

**Text cleaning** — lowercasing, URL normalization, and stripping non-alphanumeric characters (while preserving `!` and `?`, which carry sentiment signal).

**Engineered numeric features:**
- `comment_length`, `word_count`
- `exclamation_count`, `question_count`, `uppercase_ratio` (shouting/emphasis signals)
- `has_link` (binary)
- `emoji_count` (aggregated from the three emoticon reaction columns)
- `profanity_count` — frequency of terms from a curated profanity/slur lexicon; empirically the **single strongest predictor** of toxicity level
- Temporal features: `hour`, `dayofweek`, `is_weekend` (extracted from `created_date`)

**Categorical features:** `race`, `religion`, `gender`, `disability` — one-hot encoded, with missing demographic fields imputed as `"unknown"` rather than dropped.

**Text feature:** `clean_comment`, vectorized with **TF-IDF** (unigrams + bigrams, `max_features=18000`, `sublinear_tf=True`, English stop words removed).

## Modeling

All models share the same `ColumnTransformer` preprocessing pipeline (TF-IDF + `StandardScaler` + `OneHotEncoder`), evaluated on an 80/20 stratified train/validation split.

| Stage | Details |
|---|---|
| Baseline | `DummyClassifier` (most-frequent strategy) |
| Linear models | Logistic Regression (`class_weight="balanced"`), SGDClassifier (modified Huber loss) |
| Other classical | Complement Naive Bayes (count-vectorized), Linear SVM, KNN (trained on a 10k subsample for tractability) |
| Gradient boosting | XGBoost, LightGBM |
| Ensembling | Soft-voting `VotingClassifier` over LR + XGBoost + LightGBM, with blend weights grid-searched to directly maximize validation macro F1 |
| Hyperparameter tuning | `RandomizedSearchCV` (10 iterations, 3-fold CV, macro F1 scoring) on LightGBM's `n_estimators`, `learning_rate`, `num_leaves`, `min_child_samples` |
| Final model | Tuned ensemble, **retrained on the full training set** before generating test predictions |

## Results

Validation macro F1-score, ranked:

| Model | Macro F1 |
|---|---|
| **Tuned Ensemble (LR + XGB + LightGBM)** | **0.8230** |
| LightGBM | 0.7980 |
| XGBoost | 0.7838 |
| SGD Classifier | 0.7791 |
| Logistic Regression | 0.7716 |
| Linear SVM | 0.6896 |
| Naive Bayes | 0.5087 |

The tuned ensemble improved over the best single model (LightGBM) by **+0.025 macro F1 (~+3.1% relative)**.

## Key Insights

- **Profanity count is the strongest single predictor** of toxicity level, with clear separation across classes.
- **Temporal features carry almost no signal** — posting hour, day of week, and weekend flag showed near-zero correlation with toxicity, and were kept mainly for completeness/interpretability rather than predictive value.
- **Feature engineering outweighed model complexity**: the jump from raw TF-IDF to TF-IDF + engineered meta-features produced a larger accuracy gain than switching from Logistic Regression to XGBoost.
- **Ensembling clearly won**: linear models handle high-dimensional sparse TF-IDF features well, while gradient boosters capture non-linear interactions between engineered features — combining both outperformed any single model family.
- **Macro F1 over accuracy**: with ~58% of the data in one class, accuracy alone would reward a model that mostly ignores minority (more severely toxic) classes.

## Tech Stack

`Python` · `pandas` / `NumPy` · `scikit-learn` (pipelines, `ColumnTransformer`, `TfidfVectorizer`, `RandomizedSearchCV`, `VotingClassifier`) · `XGBoost` · `LightGBM` · `matplotlib` / `seaborn` (EDA)

## Repository Structure

```
.
├── notebooks/
│   └── comment_toxicity_classification.ipynb   # EDA, feature engineering, modeling, ensembling
├── submission.csv                               # final test-set predictions
└── README.md
```

## Getting Started

1. Open `notebooks/comment_toxicity_classification.ipynb` (Kaggle or local).
2. Ensure `train.csv` / `test.csv` from the *Comment Category Prediction Challenge* dataset are available.
3. Install dependencies:
   ```bash
   pip install scikit-learn xgboost lightgbm pandas numpy matplotlib seaborn scipy
   ```
4. Run all cells to reproduce: EDA → feature engineering → model benchmarking → ensemble weight tuning → LightGBM hyperparameter search → final retrained model → `submission.csv`.

## Future Work

- Swap TF-IDF for contextual embeddings (e.g., a fine-tuned transformer) to capture semantics TF-IDF misses (sarcasm, negation, reclaimed language).
- Add per-class precision/recall/confusion-matrix reporting to understand where the model confuses adjacent toxicity levels.
- Explore fairness auditing across the demographic fields (`race`, `religion`, `gender`, `disability`) to check for disparate error rates before any real-world deployment.
- Replace the hand-curated profanity lexicon with a learned or continuously updated toxic-term list to reduce maintenance overhead and coverage gaps.

## License

This project is released under the [MIT License](LICENSE).
