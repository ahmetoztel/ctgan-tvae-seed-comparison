import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import random
import torch
from ctgan import CTGAN, TVAE
from scipy.stats import ks_2samp, chi2_contingency
import tkinter as tk
from tkinter import filedialog
import os

# ✅ Kalite değerlendirme fonksiyonu
def statistical_similarity_tests(real_df, synthetic_df, categorical_cols):
    ks_results = []
    chi2_results = []
    for col in real_df.columns:
        if col in synthetic_df.columns:
            if col in categorical_cols:
                real_counts = real_df[col].value_counts().sort_index()
                synth_counts = synthetic_df[col].value_counts().sort_index()
                aligned = pd.concat([real_counts, synth_counts], axis=1).fillna(0)
                aligned.columns = ['Real', 'Synthetic']
                try:
                    aligned = aligned.astype(float)
                    chi2, p, _, _ = chi2_contingency(aligned)
                    chi2_results.append((col, chi2, p))
                except Exception as e:
                    print(f"❌ Chi-Square failed for {col}: {e}")
            else:
                try:
                    stat, p = ks_2samp(real_df[col].dropna(), synthetic_df[col].dropna())
                    ks_results.append((col, stat, p))
                except Exception as e:
                    print(f"❌ KS test failed for {col}: {e}")
    return ks_results, chi2_results

# 📁 Excel dosyası seçme
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="Select Excel file", filetypes=[("Excel files", "*.xlsx *.xls")])
raw_df = pd.read_excel(file_path, header=[0, 1])

# 🧼 Sütun ayarları
var_types = raw_df.columns.get_level_values(0).tolist()
var_names = raw_df.columns.get_level_values(1).tolist()
raw_df.columns = var_names
raw_df = raw_df.apply(lambda col: col.map(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x))

# 📊 Değişken türleri
categorical_columns = [name for typ, name in zip(var_types, var_names) if typ.lower() == "nominal"]
continuous_columns = [name for typ, name in zip(var_types, var_names) if typ.lower() == "scale"]
if "CaseNumber" in raw_df.columns:
    raw_df.drop(columns=["CaseNumber"], inplace=True)
    continuous_columns = [col for col in continuous_columns if col != "CaseNumber"]

# 🔧 Tür düzeltme ve eksik verileri doldurma
df = raw_df.copy()
for col in df.columns:
    if col in categorical_columns:
        df[col] = df[col].astype("category")
        if df[col].isnull().any():
            probs = df[col].value_counts(normalize=True)
            fill_values = np.random.choice(probs.index, size=df[col].isnull().sum(), p=probs.values)
            df.loc[df[col].isnull(), col] = fill_values
    elif col in continuous_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isnull().any():
            mean, std = df[col].mean(), df[col].std()
            values = np.random.normal(mean, std, size=df[col].isnull().sum())
            df.loc[df[col].isnull(), col] = values

# 🎯 Seed listesi (güvenli ve rastgele 100 değer)
random.seed(42)
seed_list = random.sample(range(1, 5001), 100)
print(f"🧷 Kullanılacak seed listesi: {seed_list}")

# 🧠 Modeller
model_funcs = {
    "CTGAN": lambda: CTGAN(epochs=3000),
    "TVAE": lambda: TVAE(epochs=3000)
}

total_vars = len(categorical_columns) + len(continuous_columns)
os.makedirs("progress", exist_ok=True)
summary_rows = []

# 🔁 Model ve seed döngüsü
for model_name, model_func in model_funcs.items():
    print(f"\n🚀 MODEL: {model_name}")
    result_list = []

    for seed in seed_list:
        print(f"🎯 Seed = {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        try:
            model = model_func()
            model.fit(df, discrete_columns=categorical_columns)
            synthetic_data = model.sample(5000)

            ks_results, chi2_results = statistical_similarity_tests(df, synthetic_data, categorical_columns)
            success_count = sum(p > 0.05 for _, _, p in ks_results) + sum(p > 0.05 for _, _, p in chi2_results)
            success_rate = success_count / total_vars

            print(f"✅ {seed} → {success_count}/{total_vars} ({success_rate:.2%})")
            result_list.append({"Seed": seed, "Matching_Variables": success_count, "Success_Rate": success_rate})

            with open(f"progress/results_{model_name}.txt", "a", encoding="utf-8") as txt_file:
                txt_file.write(f"{seed}\t{success_count}\t{success_rate:.4f}\n")

            pd.DataFrame([{
                "Seed": seed,
                "Matching_Variables": success_count,
                "Success_Rate": success_rate
            }]).to_csv(f"progress/progress_{model_name}.csv", mode='a', header=not os.path.exists(f"progress/progress_{model_name}.csv"), index=False)

        except Exception as e:
            print(f"⚠️ Hata oluştu (Model: {model_name}, Seed: {seed}): {e}")
            continue

    result_df = pd.DataFrame(result_list)
    result_df.to_excel(f"progress/{model_name}_results.xlsx", index=False)
    best_row = max(result_list, key=lambda x: x["Matching_Variables"])
    summary_rows.append({
        "Model": model_name,
        "BestSeed": best_row["Seed"],
        "BestScore": best_row["Matching_Variables"],
        "SuccessRate": best_row["Success_Rate"]
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_excel("progress/model_comparison_summary.xlsx", index=False)
print("\n📁 Tüm işlemler tamamlandı. Sonuçlar progress klasörüne kaydedildi.")
