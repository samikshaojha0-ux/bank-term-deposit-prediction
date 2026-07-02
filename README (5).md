# Bank Term Deposit Prediction

Predicting whether a client will subscribe to a term deposit using data from a Portuguese bank's direct marketing (phone call) campaigns.

## 📌 Overview

Banks run outbound calling campaigns to convince clients to subscribe to term deposits, but most calls don't convert. This project analyzes ~41,000 client records to understand **what drives subscription** and builds a **logistic regression model** to predict it.

## 📊 Dataset

- **Source:** [UCI Machine Learning Repository – Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **Size:** 41,188 rows × 21 columns
- **Target:** `y` — whether the client subscribed to a term deposit (`yes` / `no`)
- **Features:** client demographics (age, job, marital status, education), contact details (contact type, month, day), campaign info (number of contacts, previous outcome), and macroeconomic indicators (Euribor rate, employment variation rate, etc.)
- **Class balance:** ~11% of clients subscribed — a fairly imbalanced target, handled during modeling.

## 🔍 What's in the notebook

1. **Data loading & quality check** — shape, dtypes, missing values, target distribution
2. **Customer demographics** — age distribution, job distribution, and subscription rate by job
3. **Call duration analysis** — since this dataset doesn't include account balance, call duration was used as the key behavioral signal instead
4. **Campaign effectiveness** — contact method (cellular vs. telephone), number of contacts, and effect of previous campaign outcome
5. **Correlation heatmap** of numerical features
6. **Predictive modeling** — Logistic Regression with one-hot encoding, feature scaling, and class balancing
7. **Model evaluation** — accuracy, classification report, confusion matrix, ROC curve
8. **Feature importance** — top coefficients driving predictions

## 📈 Results

| Metric | Score |
|---|---|
| Accuracy | 86.5% |
| ROC-AUC | 0.944 |
| Recall (subscribers) | 91% |

**Key insights:**
- Call duration is the single strongest predictor — but it's only known *after* a call happens, so it can't be used to decide who to target in advance.
- Clients with a **successful previous campaign outcome** are far more likely to subscribe again.
- **Cellular contact** outperforms telephone contact.
- Clients contacted **fewer times** tend to convert better — over-contacting hurts conversion.
- Macroeconomic indicators (Euribor rate, employment variation) meaningfully correlate with subscription likelihood.

## 🛠️ Tools & Libraries

`pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn`

## 🚀 How to run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook Advance_Bank_Term_Deposit.ipynb
```

## 📁 Files

- `Advance_Bank_Term_Deposit.ipynb` — full analysis and model notebook (with outputs)
- `data.csv` — dataset used (or link to source above if omitted)
