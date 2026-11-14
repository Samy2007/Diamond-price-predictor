# 💎 Diamond Price Estimator

Jewelers price thousands of diamonds by hand, which is slow and inconsistent. This project predicts a fair price for a diamond from its certificate details (carat, cut, color, clarity and size), so a retailer can set listing prices faster and sanity-check a supplier's asking price.

## 🔗 Live app
**https://diamond-price-predictor-3cpqj68v7rma5o2aihmjpa.streamlit.app/**

## Results (held-out test set)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Baseline (always predict the mean price) | $2,993 | $3,941 | 0.000 |
| **Gradient boosting (final, 7 features)** | **$268** | **$530** | **0.982** |

On diamonds it had never seen, the model is off by about $268 on average, 91% lower than the baseline.

## Key findings
- Carat is the strongest driver of price (correlation 0.92), but the relationship is not linear, and prices jump at round weights (1.0, 1.5, 2.0 carat).
- Better cut, color and clarity look *cheaper* in raw charts because high-quality stones tend to be smaller, so size hides the effect of quality.
- Error is a similar share of the price at every price level (about 3.5% to 5.2%), but it is larger for stones above 2 carat (median 6.6%) where training data is scarce.

## Approach
1. Cleaned the data (removed rows with impossible dimensions and duplicates) and split train/test before fitting anything.
2. Built a scikit-learn Pipeline (ordinal encoding for cut/color/clarity, scaling, log-transformed target).
3. Compared a baseline with Linear Regression, Random Forest and Gradient Boosting using 5-fold cross-validation, then tuned the best candidates on the training data only.
4. Chose gradient boosting because it matched Random Forest's accuracy at a far smaller file size (0.6 MB vs 185 MB).
5. Evaluated once on the untouched test set, followed by error analysis.

## Limitations
- The data is a fixed historical price snapshot, not live market prices.
- Prices in the data stop at about $18.8k, so the model should not be used above that.
- Only certificate attributes are used. Fluorescence, the grading lab and market conditions are not.
- The raw data contains a few rows with inconsistent dimensions that were not caught by the cleaning rules.
- The test set was evaluated twice (9 features, then 7); the choice of 7 was made by cross-validation before the second evaluation.

## Run it locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Data
Diamonds dataset (53,940 diamonds), originally from the R package [ggplot2](https://ggplot2.tidyverse.org/reference/diamonds.html). CSV copy on [Kaggle](https://www.kaggle.com/shivam2503/diamonds). The dataset is not included in this repo.
![App screenshot](screenshot.png)
