# Teen Phone Addiction Prediction using Machine Learning

## Overview

This project focuses on predicting **teen smartphone addiction levels** using **Machine Learning and Ensemble Learning techniques**. The study analyzes behavioral, social, and lifestyle-related factors to estimate smartphone addiction severity among teenagers.

The project follows a complete **data science pipeline**, including:

- Data preprocessing
- Feature selection using correlation analysis
- Feature engineering
- Exploratory Data Analysis (EDA)
- Multiple regression model comparisons
- Cross-validation
- Ensemble learning using **LightGBM** and **XGBoost**
- Performance evaluation using multiple regression metrics

The objective is to identify key behavioral indicators contributing to smartphone addiction and build an accurate predictive system.

---

## Problem Statement

Excessive smartphone usage among teenagers has become a growing concern affecting:

- Sleep quality
- Mental health
- Academic performance
- Social interaction
- Daily productivity

This project aims to build a predictive model capable of estimating **phone addiction levels** based on behavioral patterns such as:

- Daily phone usage
- Gaming time
- Social media usage
- Phone checking frequency
- Sleep duration
- App usage intensity

---

## Dataset

The project uses a **Teen Phone Addiction Dataset** containing behavioral and lifestyle attributes of teenagers.

### Features Used

After correlation-based feature selection, the most relevant features were identified:

| Feature | Description |
|----------|-------------|
| Daily_Usage_Hours | Daily smartphone usage duration |
| Apps_Used_Daily | Number of apps used per day |
| Time_on_Social_Media | Time spent on social platforms |
| Time_on_Gaming | Daily gaming duration |
| Phone_Checks_Per_Day | Frequency of phone checks |
| Sleep_Hours | Daily sleep duration |
| Age | Age of teenager |
| Gender | Gender category |

### Feature Engineering

Additional engineered features were created to improve predictive performance:

#### 1. Usage-Sleep Ratio
Measures smartphone usage relative to sleep duration.

```python
Usage_Sleep_Ratio =
Daily_Usage_Hours / Sleep_Hours
```

#### 2. Phone Intensity
Measures overall phone interaction intensity.

```python
Phone_Intensity =
Phone_Checks_Per_Day × Daily_Usage_Hours
```

---

## Methodology

### 1. Data Preprocessing

The following preprocessing steps were applied:

- Removed missing values
- Dropped unnecessary identifier columns (`ID`, `Name`)
- Label encoding for categorical features
- Feature scaling using `StandardScaler`
- Train-test split
- K-Fold Cross Validation

### 2. Feature Selection

Feature importance was determined using **correlation analysis**.

Only features with:

```text
|correlation| ≥ 0.03
```

were selected for model training.

### 3. Exploratory Data Analysis (EDA)

Various visualizations were created to understand relationships between features and addiction levels:

- Correlation heatmaps
- Feature importance charts
- Social media usage vs age
- Daily usage vs addiction level
- Exercise hours vs addiction level
- Model performance comparison graphs
- Learning curves

---

## Models Implemented

### Baseline Models

The following regression models were evaluated:

1. Linear Regression
2. Decision Tree Regressor
3. Support Vector Regression (SVR)
4. Perceptron (MLPRegressor)

### Advanced Models

To improve predictive accuracy, ensemble boosting methods were implemented:

- **LightGBM Regressor**
- **XGBoost Regressor**
- Optimized weighted ensemble model

---

## Model Evaluation Metrics

Performance was evaluated using multiple metrics:

| Metric | Purpose |
|--------|---------|
| RMSE | Measures prediction error |
| MAE | Average absolute error |
| R² Score | Variance explained |
| MAPE | Percentage prediction error |
| Accuracy@ε | Prediction tolerance accuracy |

### Epsilon Accuracy

The model was additionally evaluated using tolerance-based accuracy:

- **ε = 0.0** → Exact prediction
- **ε = 0.3** → Prediction within ±0.3
- **ε = 0.5** → Prediction within ±0.5

---

## Results

### Traditional Model Performance

| Model | RMSE | R² Score |
|--------|------|-----------|
| Linear Regression | 0.8138 | 0.7371 |
| Decision Tree | 0.8868 | 0.6879 |
| SVR (C=1) | **0.2234** | **0.9802** |
| Perceptron | 0.8135 | 0.7373 |

### Ensemble Model Performance

| Model | RMSE | MAE | R² Score | Accuracy (±0.5) |
|--------|------|-----|----------|------------------|
| LightGBM | **0.2934** | **0.1864** | **0.9662** | **91.29%** |
| XGBoost | 0.3522 | 0.2398 | 0.9513 | 87.14% |
| Ensemble Model | 0.2986 | 0.1916 | 0.9650 | 90.86% |

### Best Performing Model

**LightGBM achieved the best balance between accuracy, robustness, and generalization performance**, achieving:

- **RMSE:** 0.2934
- **R² Score:** 0.9662
- **Accuracy (±0.5):** 91.29%

---

## Key Insights

The most influential predictors of phone addiction were:

1. Usage-Sleep Ratio
2. Phone Intensity
3. Apps Used Daily
4. Gaming Time
5. Social Media Usage

This indicates that **behavioral intensity and sleep disruption strongly correlate with smartphone addiction levels**.

---

## Tech Stack

### Languages
- Python

### Libraries Used

#### Data Processing
- Pandas
- NumPy

#### Visualization
- Matplotlib
- Seaborn

#### Machine Learning
- Scikit-Learn
- LightGBM
- XGBoost

#### Deep Learning
- PyTorch

#### Optimization
- SciPy

---

## Project Structure

```text
Teen-Phone-Addiction-Prediction/
│── dataset/
│   └── teen_phone_addiction_dataset.csv
│
│── notebooks/
│   └── analysis.ipynb
│
│── outputs/
│   ├── plots/
│   ├── evaluation_results/
│
│── README.md
│── requirements.txt
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Teen-Phone-Addiction-Prediction.git
```

Move into the project directory:

```bash
cd Teen-Phone-Addiction-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
analysis.ipynb
```

---

## Future Improvements

Possible future enhancements include:

- Deep learning architectures for tabular regression
- Explainable AI (SHAP values)
- Hyperparameter optimization
- Real-world mobile usage dataset integration
- Deployment using Flask/Streamlit

---

## Conclusion

This project demonstrates the effectiveness of **Machine Learning and Ensemble Models** in predicting smartphone addiction behavior among teenagers.

Through feature engineering, correlation-driven selection, and robust evaluation strategies, the project achieved **high predictive performance**, making it useful for behavioral analytics and digital wellness research.

---

## Author

**Darpan T**

If you found this project useful, consider starring the repository.
