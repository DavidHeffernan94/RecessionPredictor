\# RecessionPredictor



Predicting US recession risk from macroeconomic and financial indicators using machine learning. Covers data collection, EDA, feature engineering, model comparison and interpretability. Built as a portfolio project.



\## Overview



This project develops and evaluates machine learning models to predict US recession risk using historical macroeconomic and financial data sourced from the FRED API and Yahoo Finance. The project covers the full data science pipeline from data collection through to a live recession risk assessment that updates automatically each time the notebook is run.



\*\*Key result:\*\* Tuned XGBoost on the 1960 dataset achieves CV AUC of 0.979 and perfect test AUC of 1.000, correctly identifying all recession months in the test period with no false positives.



\*\*Current recession risk (as of last run):\*\* Both models indicate LOW RISK as of July 2026.



\## Repository Contents



RecessionPredictor/

├── RecessionPredictor.ipynb # Full analysis notebook

├── gold\_historical.csv # Historical gold prices from 1915

├── sp500\_historical.csv # Historical S\&P 500 prices from 1927

├── .gitignore

├── README.md

├── LICENSE

└── requirements.txt



\## Datasets



Two datasets are constructed from 16 macroeconomic and financial indicators:



| Dataset | Period | Observations | Recession Episodes |

|---|---|---|---|

| 1960 Dataset | 1960 - present | 787 | 9 (all NBER recessions) |

| 1987 Dataset | 1987 - present | 451 | 4 (Gulf War, Dot-Com, GFC, COVID-19) |



\### Indicators Used



\*\*FRED (via API):\*\* Unemployment, CPI, Federal Funds Rate, Industrial Production, Housing Starts, Consumer Credit, M2 Money Supply, Corporate Profits, Federal Debt as % of GDP, Yield Curve (T10Y2Y), Credit Spread (BAA10Y), Jobless Claims, Mortgage Rates, House Prices



\*\*Yahoo Finance + Historical CSV:\*\* S\&P 500, Gold - historical CSVs included in this repository extend both series back to 1960, with Yahoo Finance automatically filling forward to the present day



\## Methodology



\### Exploratory Data Analysis

\- Descriptive statistics and recession vs non-recession means

\- Stationarity testing (Augmented Dickey-Fuller) for all indicators

\- Time series plots with recession shading across all nine episodes

\- Phase analysis (before/during/after) across all recession episodes

\- Recession lead/lag correlation analysis (lags -24 to +24 months)

\- Distribution analysis and correlation heatmaps



\### Feature Engineering

\- Lag features at 3, 6 and 12 months for all indicators

\- Rolling means at 3, 6 and 12 months

\- Percentage change at 1, 3 and 12 months

\- Rolling standard deviation at 6 and 12 months

\- Extended yield curve lags at 18 and 24 months

\- Interaction features for yield curve, credit spread, fed funds and S\&P 500

\- Feature selection using 0.95 correlation threshold



\### Modelling

\- Temporal 80/20 train/test split preserving chronological order

\- TimeSeriesSplit cross validation with 5 folds

\- Three models compared: Logistic Regression, Random Forest, XGBoost

\- LSTM neural network across 8 configurations (dataset x sequence length x features)

\- Class imbalance handled via class weighting and oversampling

\- Hyperparameter tuning with GridSearchCV and RandomizedSearchCV

\- Threshold analysis using precision-recall curves



\### Interpretability

\- SHAP bar, dot and waterfall plots for XGBoost on both datasets

\- Predicted recession probability time series across full historical period

\- Current recession risk assessment updated automatically on each run



\## Results



| Model | Dataset | CV AUC | Test AUC | Test F1 |

|---|---|---|---|---|

| Logistic Regression | 1960 | 0.962 | 0.994 | 0.082 |

| Random Forest | 1960 | 0.976 | - | - |

| XGBoost | 1960 | 0.930 | - | - |

| Logistic Regression (Tuned) | 1960 | 0.975 | 0.994 | 0.082 |

| \*\*XGBoost (Tuned)\*\* | \*\*1960\*\* | \*\*0.979\*\* | \*\*1.000\*\* | \*\*1.000\*\* |

| Logistic Regression (Tuned) | 1987 | 0.363\* | 0.994 | 0.182 |

| XGBoost (Tuned) | 1987 | 0.392\* | 1.000 | 0.500 |

| LSTM (best config) | 1960 | - | 0.883 | 0.114 |



\*Tuned using F1 scoring due to sparse recession months in some CV folds



\*\*Recommended model:\*\* Tuned XGBoost on the 1960 dataset



\## Setup



\### Requirements



```bash

pip install -r requirements.txt

```



\### FRED API Key



A free FRED API key is required. Register at https://fred.stlouisfed.org/docs/api/api\_key.html



Create a `.env` file in the project root:

FRED\_API\_KEY=your\_key\_here



\### Running the Notebook



```bash

jupyter notebook RecessionPredictor.ipynb

```



Run all cells in order. The notebook will:

1\. Fetch the latest data from FRED and Yahoo Finance automatically

2\. Combine with the included historical CSVs for gold and S\&P 500

3\. Run the full analysis pipeline

4\. Output a current recession risk assessment for the most recent available month



\## Key Findings



\- \*\*Percentage change features dominate:\*\* The top SHAP features across both datasets are momentum and rate-of-change variants rather than raw levels, validating the feature engineering approach and stationarity testing findings

\- \*\*Yield curve leads by up to 22 months:\*\* The lead/lag analysis confirms yield curve inversion is the longest leading recession indicator, justifying the extended 18 and 24 month lag features

\- \*\*Credit spread is the strongest near-term signal:\*\* Peaking at lag -1 with correlation of 0.490 in the 1987 dataset, credit spread widening is the most timely leading indicator

\- \*\*LSTM underperforms on limited data:\*\* With fewer than 700 training sequences, traditional ML models outperform LSTM across all configurations

\- \*\*Post-2020 economic stress detected:\*\* Both models generate elevated signals in the post-pandemic period that have not resulted in a formal recession declaration, reflecting the unusual economic dynamics of the 2020s

\- \*\*Current risk is low:\*\* As of July 2026 both models assign recession probability below 0.5%



\## Data Sources



\- \*\*FRED\*\* - Federal Reserve Economic Data, Federal Reserve Bank of St. Louis (https://fred.stlouisfed.org)

\- \*\*Yahoo Finance\*\* - via the yfinance Python library

\- \*\*MacroTrends\*\* - historical gold and S\&P 500 price data (https://www.macrotrends.net)

\- \*\*NBER\*\* - recession dating (https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions)



\## Technologies



\- \*\*Python\*\* - pandas, NumPy, scikit-learn, XGBoost, TensorFlow, SHAP, matplotlib, seaborn, statsmodels

\- \*\*Data\*\* - FRED API (fredapi), Yahoo Finance API (yfinance), historical CSV files

\- \*\*Environment\*\* - Jupyter Notebook, python-dotenv



\## Author



David Heffernan | \[GitHub](https://github.com/DavidHeffernan94) | \[LinkedIn](https://www.linkedin.com/in/david-heffernan-871a85192/)



\## License



MIT License - see LICENSE file for details

