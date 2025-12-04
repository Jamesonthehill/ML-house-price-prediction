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

**Classic Models: Ridge, Random Forest, and Gradient Boosting**

- `Linear Regression.ipynb`  
  - Uses the shared preprocessing pipeline (imputation, one-hot encoding, scaling).  
  - Trains a **Linear Regression** model without regularization.  
  - Evaluates performance (R², RMSE, MAE, within-10% accuracy).  
  - Serves as the **simplest baseline** to show how much variance can be explained by a purely linear model.

- `Ridge regression.ipynb`  
  - Builds the preprocessing pipeline (imputation, one-hot encoding, scaling).  
  - Trains a **Ridge Regression** model (Linear Regression with L2 regularization).  
  - Evaluates performance (R², RMSE, MAE, within-10% accuracy).  
  - Serves as the **regularized linear baseline** to reduce overfitting and stabilize coefficients.

- `Polynomial Ridge Regression.ipynb`  
  - Reuses the same preprocessing pipeline.  
  - Adds **PolynomialFeatures** (degree = 3) to create nonlinear and interaction terms.  
  - Trains a **Ridge Regression** model on the expanded polynomial feature space.  
  - Evaluates performance (R², RMSE, MAE, within-10% accuracy).  
  - Explores whether **nonlinear relationships** between features and `SalePrice` improve performance compared to plain Ridge.


**Literature-Based Models: XGBoost and LightGBM**

- `Random Forest model.ipynb`  
  - Reuses the same preprocessing pipeline.  
  - Trains a **Random Forest Regressor** with hyperparameter tuning (GridSearchCV).  
  - Compares its performance to Ridge.

- `Gradient Boosting.ipynb`  
  - Implements **GradientBoostingRegressor** with tuning.  
  - This is the strongest of the three classic models in my experiments.  
  - Includes evaluation metrics and residual/error plots.

- `XGBoost.ipynb`  
  - Implements **XGBRegressor** inspired by the XGBoost paper / house-price paper.  
  - Tunes key parameters (n_estimators, learning_rate, max_depth, subsample, colsample_bytree).  
  - Achieves one of the best performances among all models.

- `LightGBM.ipynb`  
  - Implements **LGBMRegressor** based on the LightGBM paper.  
  - Uses a small hyperparameter grid (n_estimators, learning_rate, num_leaves).  
  - Compares LightGBM to the classic models.

---

- `submission_csv`  
  - Uses the best model to train on the full training data.  
  - Predicts on `test.csv` and creates `submission.csv` for Kaggle.

