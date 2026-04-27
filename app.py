import streamlit as st
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="F1 Champion Predictor",
    page_icon="🏎️",
    layout="wide"
)

# ─────────────────────────────────────────
# HELPER FUNCTIONS (model from scratch)
# ─────────────────────────────────────────

FEATURE_COLS = [
    'Race_Entries', 'Race_Wins', 'Pole_Positions', 'Podiums',
    'Fastest_Laps', 'Points', 'Win_Rate', 'Podium_Rate',
    'FastLap_Rate', 'Points_Per_Entry'
]

def predict(features, weights, bias):
    return bias + sum(w * x for w, x in zip(weights, features))

def minmax(data, cols):
    mins, maxs = {}, {}
    for col in cols:
        mins[col] = data[col].min()
        maxs[col] = data[col].max()
    return mins, maxs

def normalize_row(row, mins, maxs):
    result = []
    for col in FEATURE_COLS:
        denom = maxs[col] - mins[col]
        result.append((row[col] - mins[col]) / denom if denom != 0 else 0.0)
    return result

def train_model(X_train, y_train, lr=0.01, epochs=10000):
    n = len(X_train)
    num_features = len(X_train[0])
    weights = [0.0] * num_features
    bias = 0.0
    loss_history = []

    for epoch in range(1, epochs + 1):
        w_grad = [0.0] * num_features
        b_grad = 0.0
        for features, label in zip(X_train, y_train):
            error = predict(features, weights, bias) - label
            for i, x in enumerate(features):
                w_grad[i] += error * x
            b_grad += error
        for i in range(num_features):
            weights[i] -= lr * (w_grad[i] / n)
        bias -= lr * (b_grad / n)
        if epoch % 500 == 0:
            mse = sum((predict(f, weights, bias) - l)**2
                      for f, l in zip(X_train, y_train)) / n
            loss_history.append((epoch, mse))

    return weights, bias, loss_history

# ─────────────────────────────────────────
# LOAD & TRAIN (cached so it runs once)
# ─────────────────────────────────────────

@st.cache_resource
def load_and_train():
    df = pd.read_csv("F1DriversDataset.csv")

    # Preprocess
    df = df[(df['Race_Entries'] > 0) & (df['Race_Starts'] > 0)].copy()
    df['Champion_int'] = df['Champion'].astype(int)

    mins, maxs = minmax(df, FEATURE_COLS)

    # Normalize
    df_norm = df.copy()
    for col in FEATURE_COLS:
        denom = maxs[col] - mins[col]
        df_norm[col] = (df[col] - mins[col]) / denom if denom != 0 else 0

    combined = list(zip(
        df_norm[FEATURE_COLS].values.tolist(),
        df_norm['Champion_int'].values.tolist()
    ))
    random.seed(42)
    random.shuffle(combined)

    split = int(len(combined) * 0.8)
    train_data = combined[:split]
    test_data  = combined[split:]

    X_train = [r[0] for r in train_data]
    y_train = [r[1] for r in train_data]
    X_test  = [r[0] for r in test_data]
    y_test  = [r[1] for r in test_data]

    weights, bias, loss_history = train_model(X_train, y_train)

    return df, weights, bias, mins, maxs, X_test, y_test, loss_history

# ─────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────

st.title("🏎️ F1 World Champion Predictor")
st.markdown("**Roll Numbers:** 24L-0679 | 24L-0912 | 24L-0518")
st.markdown("---")

with st.spinner("Loading data and training model... please wait ⏳"):
    df, weights, bias, mins, maxs, X_test, y_test, loss_history = load_and_train()

st.success("Model trained successfully!")

# ── TABS ──
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dataset Overview",
    "📈 Visualizations",
    "🧪 Model Evaluation",
    "🏆 Predict a Driver"
])

# ────────────────────────────────────────
# TAB 1 — Dataset Overview
# ────────────────────────────────────────
with tab1:
    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Drivers", len(df))
    col2.metric("Champions", int(df['Champion'].sum()))
    col3.metric("Non-Champions", int((~df['Champion']).sum()))
    col4.metric("Features Used", len(FEATURE_COLS))

    st.subheader("Sample Data (first 10 rows)")
    st.dataframe(df[['Driver'] + FEATURE_COLS + ['Champion']].head(10), use_container_width=True)

    st.subheader("Statistical Summary")
    st.dataframe(df[FEATURE_COLS].describe().round(3), use_container_width=True)

# ────────────────────────────────────────
# TAB 2 — Visualizations
# ────────────────────────────────────────
with tab2:
    st.header("Data Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Champion Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        counts = df['Champion'].value_counts()
        ax.bar(['Non-Champion', 'Champion'], counts.values,
               color=['#4C72B0', '#DD8452'])
        ax.set_ylabel("Number of Drivers")
        ax.set_title("Champion vs Non-Champion")
        for i, v in enumerate(counts.values):
            ax.text(i, v + 2, str(v), ha='center', fontweight='bold')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Race Wins Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(df[df['Champion']==False]['Race_Wins'], bins=30,
                color='#4C72B0', alpha=0.7, label='Non-Champion')
        ax.hist(df[df['Champion']==True]['Race_Wins'], bins=15,
                color='#DD8452', alpha=0.9, label='Champion')
        ax.set_xlabel("Race Wins")
        ax.set_ylabel("Drivers")
        ax.set_title("Race Wins by Champion Status")
        ax.legend()
        st.pyplot(fig)
        plt.close()

    st.subheader("Correlation Heatmap")
    corr_df = df[FEATURE_COLS].copy()
    corr_df['Champion'] = df['Champion'].astype(int)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_df.corr(), annot=True, fmt='.2f',
                cmap='coolwarm', center=0, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    st.pyplot(fig)
    plt.close()

    st.subheader("Top 10 Drivers by Race Wins")
    top10 = df.nlargest(10, 'Race_Wins')[['Driver', 'Race_Wins', 'Champion']]
    bar_colors = ['#DD8452' if c else '#4C72B0' for c in top10['Champion']]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(top10['Driver'], top10['Race_Wins'], color=bar_colors)
    ax.set_xlabel("Race Wins")
    ax.set_title("Top 10 Drivers  (Orange = Champion)")
    ax.invert_yaxis()
    st.pyplot(fig)
    plt.close()

    st.subheader("Training Loss Curve")
    epochs_plot = [x[0] for x in loss_history]
    mse_plot    = [x[1] for x in loss_history]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(epochs_plot, mse_plot, color='#DD8452', linewidth=2)
    ax.set_title("MSE Loss over Training Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()

# ────────────────────────────────────────
# TAB 3 — Model Evaluation
# ────────────────────────────────────────
with tab3:
    st.header("Model Evaluation")

    raw_preds = [predict(f, weights, bias) for f in X_test]
    y_pred = [1 if p >= 0.5 else 0 for p in raw_preds]
    y_true = [int(v) for v in y_test]

    n = len(y_true)
    mse  = sum((p-a)**2 for p,a in zip(raw_preds, y_true)) / n
    rmse = math.sqrt(mse)
    mae  = sum(abs(p-a) for p,a in zip(raw_preds, y_true)) / n
    mean_a = sum(y_true) / n
    ss_tot = sum((a-mean_a)**2 for a in y_true)
    ss_res = sum((a-p)**2 for a,p in zip(y_true, raw_preds))
    r2 = 1 - (ss_res/ss_tot) if ss_tot != 0 else 0

    acc       = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    cm        = confusion_matrix(y_true, y_pred)

    st.subheader("Regression Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE",  f"{mae:.4f}")
    c2.metric("MSE",  f"{mse:.4f}")
    c3.metric("RMSE", f"{rmse:.4f}")
    c4.metric("R²",   f"{r2:.4f}")

    st.subheader("Classification Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  f"{acc*100:.2f}%")
    c2.metric("Precision", f"{precision:.4f}")
    c3.metric("Recall",    f"{recall:.4f}")
    c4.metric("F1 Score",  f"{f1:.4f}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Non-Champion','Champion'],
                    yticklabels=['Non-Champion','Champion'])
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Metrics Bar Chart")
        fig, ax = plt.subplots(figsize=(5, 4))
        metrics = ['Accuracy','Precision','Recall','F1 Score']
        values  = [acc, precision, recall, f1]
        bars = ax.bar(metrics, values,
                      color=['#4C72B0','#DD8452','#55A868','#C44E52'])
        ax.set_ylim(0, 1.15)
        ax.set_title("Classification Metrics")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', fontweight='bold')
        st.pyplot(fig)
        plt.close()

# ────────────────────────────────────────
# TAB 4 — Predict a Driver
# ────────────────────────────────────────
with tab4:
    st.header("🏆 Predict a Driver's Championship Potential")
    st.markdown("Enter a driver's career statistics below and the model will predict if they could be a World Champion.")

    col1, col2 = st.columns(2)

    with col1:
        race_entries   = st.number_input("Race Entries",    min_value=0, max_value=400, value=50)
        race_wins      = st.number_input("Race Wins",       min_value=0, max_value=200, value=5)
        pole_positions = st.number_input("Pole Positions",  min_value=0, max_value=250, value=3)
        points         = st.number_input("Career Points",   min_value=0.0, max_value=5000.0, value=200.0)
        podiums        = st.number_input("Podiums",         min_value=0, max_value=300, value=10)

    with col2:
        fastest_laps    = st.number_input("Fastest Laps",      min_value=0, max_value=100, value=3)
        win_rate        = st.number_input("Win Rate (wins/entries)", min_value=0.0, max_value=1.0, value=0.10, step=0.01)
        podium_rate     = st.number_input("Podium Rate",       min_value=0.0, max_value=1.0, value=0.20, step=0.01)
        fastlap_rate    = st.number_input("Fastest Lap Rate",  min_value=0.0, max_value=1.0, value=0.06, step=0.01)
        points_per_entry= st.number_input("Points Per Entry",  min_value=0.0, max_value=50.0, value=4.0, step=0.1)

    if st.button("🔍 Predict", use_container_width=True):
        raw = {
            'Race_Entries': race_entries, 'Race_Wins': race_wins,
            'Pole_Positions': pole_positions, 'Podiums': podiums,
            'Fastest_Laps': fastest_laps, 'Points': points,
            'Win_Rate': win_rate, 'Podium_Rate': podium_rate,
            'FastLap_Rate': fastlap_rate, 'Points_Per_Entry': points_per_entry
        }
        norm_features = normalize_row(raw, mins, maxs)
        score = predict(norm_features, weights, bias)
        score_clamped = max(0.0, min(1.0, score))

        st.markdown("---")
        if score >= 0.5:
            st.success(f"🏆 **LIKELY A WORLD CHAMPION!**  (Confidence score: {score_clamped:.2%})")
        else:
            st.error(f"❌ **Likely NOT a World Champion**  (Confidence score: {score_clamped:.2%})")

        st.progress(score_clamped)
        st.caption(f"Raw model output: {score:.4f}  |  Threshold: 0.5")
