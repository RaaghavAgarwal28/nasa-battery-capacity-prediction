# NASA Battery Capacity Predictor

A complete end-to-end machine learning research repository for predicting lithium-ion battery capacity degradation using the **NASA Battery Aging Dataset**.

---

## 📌 Problem Statement

Predict battery **Capacity (Ah)** as a **regression** task using electrochemical sensor data collected across charge/discharge cycles.

---

## 🗂️ Project Structure

```
nasa battery predictor/
│
├── data/
│   ├── metadata.csv         # Raw experiment metadata
│   └── data.zip             # Raw CSV files per experiment
│
├── outputs/
│   ├── modelling_dataset.csv
│   ├── linear_regression/
│   ├── random_forest/
│   ├── lstm/
│   ├── comparison/
│   ├── eda/
│   ├── engineering_plots/
│   └── report_graphs/
│
├── reports/
│   └── NASA_Battery_Report.docx  # Final Word report with all graphs
│
├── models/
│
├── src/
│   ├── 01_extract_data.py
│   ├── 02_feature_engineering.py
│   ├── 03_engineering_plots.py
│   ├── 04_eda.py
│   ├── 05_linear_regression.py
│   ├── 06_random_forest.py
│   ├── 07_lstm.py
│   ├── 08_compare_models.py
│   ├── 09_final_report_graphs.py
│   ├── 10_generate_word_report.py
│   └── utils.py
│
└── main.py                   # Full pipeline orchestrator
```

---

## ⚙️ Pipeline

Run the entire end-to-end pipeline with a single command:

```bash
python main.py
```

This executes all 8 steps in sequence:

| Step | Script | Description |
|------|--------|-------------|
| 1 | `01_extract_data.py` | Extract & clean raw CSV data |
| 2 | `02_feature_engineering.py` | Build modelling_dataset.csv with 17 features |
| 3 | `03_engineering_plots.py` | Engineering diagnostic plots |
| 4 | `04_eda.py` | Exploratory Data Analysis |
| 5 | `05_linear_regression.py` | Train & evaluate Linear Regression |
| 6 | `06_random_forest.py` | Train & evaluate Random Forest |
| 7 | `07_lstm.py` | Train & evaluate LSTM |
| 8 | `08_compare_models.py` | Compare all models & pick best |

---

## 📊 Model Results

| Model | MAE ↓ | RMSE ↓ | R² ↑ |
|-------|--------|--------|------|
| Linear Regression | 0.1029 | 0.1523 | 0.8874 |
| LSTM | 0.0784 | 0.1169 | 0.9337 |
| **Random Forest** | **0.0142** | **0.0585** | **0.9834** |

🏆 **Best model: Random Forest** (R² = 0.983)

---

## 🔑 Key Features Used

- `Cycle_Number`, `Ambient_Temperature`
- Voltage stats: Mean, Std, Min, Max
- Current stats: Mean, Std, Min, Max
- Temperature stats: Mean, Std, Min, Max
- `Discharge_Duration`
- `Re` (Electrolyte Resistance), `Rct` (Charge Transfer Resistance)

---

## 📦 Dependencies

```
numpy
pandas
matplotlib
scikit-learn
tensorflow
scipy
openpyxl
python-docx
```

Install with:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow scipy openpyxl python-docx
```

---

## 📁 Dataset

**NASA Battery Aging Dataset** — available from the [NASA Prognostics Center of Excellence](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

Place `metadata.csv` and `data.zip` inside the `data/` folder before running.

---

## 👤 Author

**Raaghav Agarwal**
