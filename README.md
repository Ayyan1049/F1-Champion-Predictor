# 🏎️ F1 World Champion Predictor

> A machine learning project that predicts whether a Formula 1 driver will become a **World Champion** based on their career statistics — built from scratch in Python with no ML libraries for the model.

<br>

## 👥 Team

Ayyan Khan
Sohaib Irshad

<br>

## 📌 Project Overview

This project applies **Multiple Linear Regression** trained via **Batch Gradient Descent** to classify F1 drivers as potential World Champions or not. The model is built entirely from scratch using only Python's built-in modules — no scikit-learn or any ML library was used for training.

The dataset contains real career statistics for **868 Formula 1 drivers** from **1950 to 2024**, sourced from Kaggle. After preprocessing, **781 driver records** are used for training and evaluation.

<br>

## 🗂️ Project Structure

```
F1-Champion-Predictor/
│
├── app.py                          # Streamlit web application
├── F1_Champion_Predictor.ipynb     # Complete Jupyter Notebook
├── F1DriversDataset.csv            # Raw dataset
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

<br>

## 🧠 How It Works

### The Model — Multiple Linear Regression
The model learns a weighted formula from the training data:

```
score = (w1 × Race_Wins) + (w2 × Points) + (w3 × Win_Rate) + ... + bias
```

- If `score >= 0.5` → **World Champion**
- If `score < 0.5` → **Not a Champion**

### Training — Batch Gradient Descent
- **Epochs:** 10,000
- **Learning Rate:** 0.01
- **Train/Test Split:** 80% / 20%
- All weights start at 0 and are adjusted each epoch to minimise error

<br>

## 📊 Dataset

| Property | Value |
|---|---|
| Source | Kaggle — F1 Drivers Dataset |
| Total Records | 868 drivers |
| After Preprocessing | 781 drivers |
| World Champions | 34 |
| Non-Champions | 747 |
| Features Used | 10 |
| Target Variable | Champion (True / False) |

### Independent Variables (Features)

| Feature | Type | Description |
|---|---|---|
| Race_Entries | Integer | Total race entries in career |
| Race_Wins | Integer | Total race wins |
| Pole_Positions | Integer | Total pole positions |
| Podiums | Integer | Total podium finishes (top 3) |
| Fastest_Laps | Integer | Total fastest laps recorded |
| Points | Float | Total career championship points |
| Win_Rate | Float | Race wins / Race entries |
| Podium_Rate | Float | Podiums / Race entries |
| FastLap_Rate | Float | Fastest laps / Race entries |
| Points_Per_Entry | Float | Average points per race |

### Dependent Variable (Target)
| Feature | Type | Description |
|---|---|---|
| Champion | Boolean | True if driver won World Championship |

<br>

## ⚙️ Data Preprocessing

1. **Drop irrelevant columns** — removed Seasons, Championship Years, Decade, Active, Start_Rate, Nationality
2. **Remove non-racing records** — dropped 87 drivers with zero Race_Entries or Race_Starts
3. **Check for duplicates** — no duplicate Driver entries found
4. **Check missing values** — all 10 feature columns had zero missing values
5. **Min-Max Normalisation** — all features scaled to [0, 1] using:

```
x_norm = (x - x_min) / (x_max - x_min)
```

<br>

## 📈 Model Evaluation Results

### Regression Metrics
| Metric | Formula | Result |
|---|---|---|
| MAE | Mean Absolute Error | ~0.04 |
| MSE | Mean Squared Error | ~0.02 |
| RMSE | √MSE | ~0.13 |
| R² | 1 - (SS_res / SS_tot) | ~0.85 |

### Classification Metrics
| Metric | Result |
|---|---|
| Accuracy | ~97% |
| Precision | High |
| Recall | High |
| F1 Score | High |

<br>

## 🌐 Web Application

The Streamlit app provides a clean **Ferrari-themed** user interface with 4 tabs:

| Tab | Description |
|---|---|
| 📋 Dataset | View raw data and statistical summary |
| 📊 Visualizations | Charts, heatmaps, and training loss curve |
| 🧪 Evaluation | All regression and classification metrics + confusion matrix |
| 🏆 Predict | Enter any driver's stats and get a live prediction |

<br>

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/F1-Champion-Predictor.git
cd F1-Champion-Predictor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Jupyter Notebook
```bash
jupyter notebook F1_Champion_Predictor.ipynb
```

### 4. Run the Streamlit App
```bash
streamlit run app.py
```
The app will open automatically in your browser.

> ⚠️ Make sure `F1DriversDataset.csv` is in the same folder as `app.py` and the notebook.

<br>

## 📦 Requirements

```
streamlit
pandas
matplotlib
seaborn
scikit-learn
jupyter
```

Or install all at once:
```bash
pip install streamlit pandas matplotlib seaborn scikit-learn jupyter
```

<br>

## 🔑 Key Concepts Used

| Concept | Description |
|---|---|
| Multiple Linear Regression | Core prediction model built from scratch |
| Batch Gradient Descent | Training algorithm — uses all data each epoch |
| Min-Max Normalisation | Scales features to [0, 1] range |
| Train/Test Split | 80% training, 20% testing |
| Confusion Matrix | Visual breakdown of correct vs wrong predictions |
| F1 Score | Balance between Precision and Recall |

<br>

## 📚 Domain

**Predictive Analytics & Sports Data Science**

Real-world applications of this model include:
- F1 team recruitment — spotting champion-calibre drivers early
- Sports media — statistical storytelling and driver comparisons
- Fantasy F1 platforms — long-term driver value assessment
- Academic research — quantifying what separates champions from competitors

<br>

---

*Built from scratch in Python · No ML libraries used for model training · Dataset sourced from Kaggle*
