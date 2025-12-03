# IntroML Capstone – House Price Prediction

This repository contains my capstone project for **Intro to Machine Learning**.  
The goal is to predict house sale prices on the **Kaggle “House Prices – Advanced Regression Techniques”** dataset using several classic and literature-based ML models.

---

## Repository Structure

### Data

- `data/train.csv` – Kaggle training data (includes `SalePrice`).
- `data/test.csv` – Kaggle test data (no `SalePrice`, used for submission).
- (These files are **not** uploaded to GitHub; download them from Kaggle and put them in the `data/` folder.)

### Notebooks

- `Ridge regression – as Method 1.ipynb`  
  - Builds the preprocessing pipeline (imputation, one-hot encoding, scaling).  
  - Trains a **Ridge Regression** model.  
  - Evaluates performance (R², RMSE, MAE, within-10% accuracy).  
  - Serves as the **baseline linear model**.

- `Random Forest model – as Method 2.ipynb`  
  - Reuses the same preprocessing pipeline.  
  - Trains a **Random Forest Regressor** with hyperparameter tuning (GridSearchCV).  
  - Compares its performance to Ridge.

- `Gradient Boosting – as Method 3.ipynb`  
  - Implements **GradientBoostingRegressor** with tuning.  
  - This is the strongest of the three classic models in my experiments.  
  - Includes evaluation metrics and residual/error plots.

**Literature-Based Models: XGBoost and LightGBM**

- `XGBoost – Literature Method 1.ipynb`  
  - Implements **XGBRegressor** inspired by the XGBoost paper / house-price paper.  
  - Tunes key parameters (n_estimators, learning_rate, max_depth, subsample, colsample_bytree).  
  - Achieves one of the best performances among all models.

- `LightGBM – Literature Method 2.ipynb`  
  - Implements **LGBMRegressor** based on the LightGBM paper.  
  - Uses a small hyperparameter grid (n_estimators, learning_rate, num_leaves).  
  - Compares LightGBM to the classic models.



- `submission_csv`  
  - Uses the best model to train on the full training data.  
  - Predicts on `test.csv` and creates `submission.csv` for Kaggle.

