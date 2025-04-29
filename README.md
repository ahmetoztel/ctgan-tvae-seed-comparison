# ctgan-tvae-seed-comparison
# CTGAN vs TVAE Model Comparison for Synthetic Data Generation

This repository contains Python code for comparing the performance of CTGAN and TVAE models in generating synthetic tabular data. The evaluation is based on the statistical similarity between the real dataset and the synthetic datasets generated with different random seeds.

## 📋 Project Overview

- **Goal:**  
  To identify the most suitable model (CTGAN or TVAE) for synthetic data generation by comparing their performance across 100 different random seed values.
  
- **Dataset Input:**  
  A real dataset in Excel format, with variable types (nominal or scale) specified in the first row and variable names in the second row.

- **Models:**  
  - [CTGAN](https://arxiv.org/abs/1907.00503)
  - [TVAE](https://arxiv.org/abs/1907.00503)

- **Evaluation Metrics:**
  - **Kolmogorov-Smirnov Test (KS)** for continuous variables
  - **Chi-Square Goodness-of-Fit Test (Chi²)** for categorical variables
  - **Success Rate:**  
    Percentage of variables with p-value > 0.05, indicating no significant difference between real and synthetic distributions.

## 🔧 Requirements

```bash
Python 3.8+
pip install pandas numpy scipy ctgan torch seaborn matplotlib openpyxl
```

> Recommended: Use a virtual environment.

## 🚀 How It Works

- **Select an Excel File:**  
  The code prompts you to select a dataset file via a file dialog.

- **Data Preprocessing:**  
  - Converts comma decimals to dots.
  - Fills missing values:
    - Categorical variables: random sampling according to frequency.
    - Continuous variables: random normal distribution based on column mean and standard deviation.

- **Model Training and Evaluation:**
  - For each model (CTGAN and TVAE), 100 synthetic datasets are generated with different random seeds.
  - Each synthetic dataset is compared against the real dataset using KS and Chi² tests.
  - Success rates are calculated for each seed.

- **Result Saving:**
  - Progress and final results are saved under a `progress/` directory.
  - Best-performing seed for each model is recorded.

## 📂 Output Files

- `progress/CTGAN_results.xlsx`
- `progress/TVAE_results.xlsx`
- `progress/model_comparison_summary.xlsx`
- Intermediate `.csv` and `.txt` progress logs.

## 📈 Example Progress Output

| Model  | BestSeed | Matching Variables | Success Rate |
|--------|----------|---------------------|--------------|
| CTGAN  | 2931     | 10/16               | 62.5%        |
| TVAE   | 3113     | 13/16               | 81.3%        |

## 📌 Important Notes

- **Random Seed Control:**  
  Random states of `random`, `numpy`, and `torch` libraries are fixed for each run to ensure reproducibility.
  
- **Epochs:**  
  Each model is trained for **3000 epochs** by default.

- **Sample Size:**  
  Each synthetic dataset contains **5000 rows**.

- **Error Handling:**  
  Any issues during KS or Chi-Square tests are logged without stopping the entire process.

## 📚 References

- Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K. (2019). *Modeling Tabular Data using Conditional GAN*. [arXiv:1907.00503](https://arxiv.org/abs/1907.00503)

## ✨ License

This project is licensed under the MIT License - feel free to use and adapt it!

