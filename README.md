# Appliances-Energy-Prediction

This repository contains my solutions to Homework 1 for Statistical Learning.  
The project applies linear regression, ridge regression, lasso regression, and stochastic gradient descent (SGD) to the Appliances Energy Prediction dataset.

---

## Repository Contents

- **Stat Learning hmwk1.ipynb**  
  Jupyter notebook with code for preprocessing, modeling, hyperparameter tuning, and evaluation.

- **Statistical Learning Homework Questions.pdf**  
  The assignment prompt with detailed instructions.

- **Statistical Learning Questions Report.pdf**  
  A written report summarizing methodology, metrics, and plots.

---

## Problem Description

We use the Appliances Energy Prediction dataset (UCI ML Repository) to predict appliance energy usage using sensor features and weather variables sampled at 10-minute intervals.  
The assignment specifies distinct train/validation/test splits by timestamp and requires implementations of standard linear regression, ridge, lasso, and ridge via SGD, with reporting of RMSE and R² across splits, coefficient path plots, and hyperparameter analysis.

---

## Methods

1. **Preprocessing**
   - Drop non-numeric or irrelevant columns (for example, date/time text fields) to reduce noise.
   - Remove rows with missing values for a clean training signal.
   - Separate features from target (`Appliances`).
   - Scale features with a standard scaler to normalize magnitude (important for ridge/lasso).

2. **Linear Regression (Baseline)**
   - Train on the training set only; evaluate on train, validation, and test.
   - Report RMSE and R² for each split.

3. **Ridge and Lasso Regression**
   - Sweep multiple regularization strengths (lambda/alpha).
   - Report metrics per split in a comparison table.
   - Select the optimal regularization by lowest validation RMSE.
   - Plot coefficient paths to show shrinkage (ridge) and sparsity (lasso).

4. **Ridge via SGD**
   - Optimize the objective: (1/2)||y − Xβ||²₂ + λ||β||²₂.
   - Return a history of loss per epoch.
   - Tune learning rate and batch size; visualize convergence.

---

## Key Results

- Ridge: optimal alpha around 0.001 based on validation RMSE.  
- Lasso: optimal alpha around 10.0 based on validation RMSE.  
- Coefficient paths:
  - Ridge shrinks coefficients smoothly toward zero (rarely exactly zero).
  - Lasso drives many coefficients exactly to zero at moderate/high alpha (embedded feature selection).
- SGD tuning:
  - Learning rate of 0.001 showed stable, smooth convergence.
  - Batch sizes 16 or 32 provided a good trade-off between stability and speed.

See the notebook for metric tables and plots reproducing these findings.

---

## Conclusion

This homework demonstrates how regularization and optimization choices impact generalization and training dynamics:

-   Ridge improves stability by shrinking coefficients.
-   Lasso performs feature selection by setting coefficients exactly to zero.
-   SGD hyperparameters (learning rate, batch size) strongly affect convergence speed and stability; `lr≈0.001` and batch size `16–32` worked best in our runs.

---

## Author

Ifesinachi Aroh
Data Science
Auburn University

