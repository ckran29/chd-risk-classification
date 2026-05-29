
# CHD Classification Analysis — CDC BRFSS 2024

Predicting coronary heart disease (CHD) status in U.S. adults using 
demographic, lifestyle, and chronic condition variables from the 2024 
CDC Behavioral Risk Factor Surveillance System (BRFSS) in R.

## Overview

This project applies supervised classification methods to identify 
individuals at risk for CHD using publicly available survey data. Three 
models were trained and evaluated: decision tree, logistic regression, 
and XGBoost. LASSO regularization was used for variable selection prior 
to model training. Logistic regression was selected as the final model 
based on AUC performance and interpretability.

## Results

| Model               | Accuracy | Sensitivity | Specificity | AUC   |
|---------------------|----------|-------------|-------------|-------|
| Decision Tree       | 75.66%   | 81.34%      | 69.98%      | 0.810 |
| Logistic Regression | 76.11%   | 78.11%      | 74.10%      | 0.839 |
| XGBoost             | 75.88%   | 78.93%      | 72.82%      | 0.838 |

## Files

| File | Description |
|------|-------------|
| `CDC_Analysis.R` | Clean annotated R script |
| `BUS 4440_CDC.Rmd` | R Markdown source for the report |
| `BUS 4440_CDC.html` | Knitted HTML report |

## Data

Raw data is not included in this repository. The 2024 BRFSS dataset 
(`LLCP2024.XPT`) can be downloaded directly from the CDC:

https://www.cdc.gov/brfss/annual_data/annual_2024.html

Place the XPT file in the project root directory before running 
`CDC_analysis.R`.

## Requirements

```r
install.packages(c("haven", "tidyverse", "dplyr", "caret", "rpart",
                   "rpart.plot", "glmnet", "ggplot2", "pROC", 
                   "xgboost", "Ckmeans.1d.dp", "doParallel"))
```

## Course

BUS 4440 Data Mining — Florida Southern College  
Instructor: Dr. Shankar Ghimire
