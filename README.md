# Weaving Fabric Rejection Prediction

A data mining mini project that predicts the **quantity of rejected and cut fabric pieces** (`Rej_and_Cut_Piece`) in textile weaving production using machine learning regression models.

Accurately estimating rejections before production helps textile manufacturers plan material requirements, reduce waste, and control costs.

---

## Dataset

- **Title:** full_weaving_dataset
- **Author:** Toufique Ahmed (National Institute of Textile Engineering and Research)
- **Published:** 3 January 2023, Mendeley Data, Version 1
- **DOI:** [10.17632/nxb4shgs9h.1](https://doi.org/10.17632/nxb4shgs9h.1)
- **Link:** https://data.mendeley.com/datasets/nxb4shgs9h/1
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Size:** 121,148 rows x 18 columns (9 months of weaving production data)
- **File used:** `weaving_dataset_full.csv`
- **Target variable:** `Rej_and_Cut_Piece` (rejected and cut fabric quantity)
- **Features:** production order details such as required finished fabric, fabric allowance, beam length, assumed/actual crimp, required grey fabric, machine production, warp/weft count, EPI, PPI, and categorical order attributes.

> The dataset is **not included** in this repository. Please download it from the link above and check its license on the Mendeley page.

---

## Project Workflow

1. **Data loading and exploration**: shape, data types, summary statistics, missing values, duplicates, target distribution
2. **Data cleaning**: column name normalisation, duplicate removal, string-to-numeric conversion, removal of missing/negative target values, dropping the order ID
3. **Train/test split**: 80% training, 20% testing (`random_state=42`)
4. **Preprocessing pipeline**:
   - Numerical: median imputation
   - Categorical: most-frequent imputation + one-hot encoding (`min_frequency=10`)
5. **Model training**: six models trained on the same processed data
6. **Evaluation**: MAE, RMSE and R² on the test set
7. **Analysis**: actual vs predicted plot, residual distribution, feature importance (Random Forest and XGBoost)

---

## Models Compared

| # | Model |
|---|-------|
| 1 | Mean Baseline (DummyRegressor) |
| 2 | Linear Regression |
| 3 | Decision Tree Regressor |
| 4 | Random Forest Regressor |
| 5 | Gradient Boosting Regressor |
| 6 | XGBoost Regressor |

---

## Results

| Rank | Algorithm | MAE | RMSE | R² Score |
|------|-----------|-----|------|----------|
| 1 | **Decision Tree** | **0.9979** | **9.8738** | **0.9975** |
| 2 | Random Forest | 1.1333 | 15.7654 | 0.9935 |
| 3 | XGBoost | 3.0570 | 19.9606 | 0.9896 |
| 4 | Gradient Boosting | 4.7201 | 22.6516 | 0.9866 |
| 5 | Linear Regression | 10.3253 | 68.7545 | 0.8769 |
| 6 | Mean Baseline | 27.9456 | 195.9679 | -0.0000 |

### Key Findings

- The **Decision Tree** achieved the best scores (R² = 0.9975, MAE = 0.9979), closely followed by **Random Forest** (R² = 0.9935).
- Tree-based models clearly outperform **Linear Regression** (R² = 0.8769), which suggests the relationship between production parameters and rejections is non-linear.
- All models beat the **Mean Baseline** (R² ≈ 0) by a large margin.
- MAE is much smaller than RMSE for the top models, meaning most predictions are very accurate while a few large-quantity orders produce bigger errors.

### Limitations

- Very high R² scores can indicate that some features (for example, production quantity or required fabric columns) are strongly tied to the target. Feature importance and possible data leakage should be examined before using the model in a real production setting.
- Results come from a single 80/20 split. Cross-validation and hyperparameter tuning would give a more reliable estimate.

---

## Repository Structure

```
weaving-rejection-prediction/
├── DM_Mini_Project.ipynb      # Main notebook (code + outputs)
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
└── results/                   # (optional) saved output CSVs
    ├── model_performance_results.csv
    ├── xgboost_feature_importance.csv
    └── actual_vs_predicted.csv
```

---

## How to Run

### Option 1: Google Colab (recommended)

1. Download the dataset from the Mendeley link above.
2. Upload `weaving_dataset_full.csv` to your Google Drive.
3. Open `DM_Mini_Project.ipynb` in Google Colab.
4. Update `REJECTION_DATA_PATH` in the data loading cell to match your file location.
5. Run all cells (**Runtime → Run all**).

### Option 2: Locally

```bash
git clone https://github.com/<your-username>/weaving-rejection-prediction.git
cd weaving-rejection-prediction
pip install -r requirements.txt
```

Then open the notebook in Jupyter, remove the Colab-specific lines (`!pip install`, `google.colab` imports), set the dataset path to your local CSV, and run all cells.

---

## Technologies Used

- Python 3
- pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn
- XGBoost

---

## Output Files

Running the notebook generates:

- `model_performance_results.csv`: comparison of all models
- `xgboost_feature_importance.csv`: XGBoost feature importance
- `actual_vs_predicted.csv`: test-set actual vs XGBoost predicted values

---

## License and Attribution

**Code:** released under the [MIT License](LICENSE).

**Dataset:** the data used in this project is by Toufique Ahmed and is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). No changes were made to the original files. The data was cleaned and transformed inside the notebook (duplicate removal, numeric conversion, missing-value imputation, encoding) for model training only.

**Citation:**

> Ahmed, Toufique (2023), "full_weaving_dataset", Mendeley Data, V1, doi: 10.17632/nxb4shgs9h.1

---

## Author

**Devanand Ashok Prajapati**
Data Mining Mini Project
<Vivekanand Education Society's Institute of Technology (VESIT) / Artificial Intelligence & Data Science>
