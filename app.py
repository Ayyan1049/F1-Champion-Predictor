import streamlit as st
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

st.set_page_config(
    page_title="F1 Champion Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Racing+Sans+One&family=Barlow:wght@300;400;500;600;700&family=Barlow+Condensed:wght@400;600;700&display=swap');

:root {
    --ferrari-red:   #DC0000;
    --ferrari-light: #FF3333;
    --ferrari-gold:  #C8A951;
    --black:         #0A0A0A;
    --dark-gray:     #111111;
    --card-bg:       #141414;
    --border:        rgba(220,0,0,0.25);
    --text-primary:  #F5F5F5;
    --text-muted:    #888888;
}

html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
    background-color: var(--black) !important;
    color: var(--text-primary) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

.hero {
    background: linear-gradient(135deg, #0A0A0A 0%, #1a0000 50%, #0A0A0A 100%);
    border-bottom: 3px solid var(--ferrari-red);
    padding: 48px 60px 36px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(220,0,0,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-badge {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 11px; font-weight: 700;
    letter-spacing: 4px; color: var(--ferrari-red);
    text-transform: uppercase; margin-bottom: 12px;
}
.hero-title {
    font-family: 'Racing Sans One', sans-serif;
    font-size: clamp(32px, 5vw, 60px);
    color: #FFFFFF; line-height: 1; margin-bottom: 8px;
    text-shadow: 0 0 40px rgba(220,0,0,0.4);
}
.hero-title span { color: var(--ferrari-red); }
.hero-sub {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 16px; color: var(--text-muted);
    letter-spacing: 2px; margin-bottom: 20px;
}
.roll-pills { display: flex; gap: 10px; flex-wrap: wrap; }
.roll-pill {
    background: rgba(220,0,0,0.12);
    border: 1px solid rgba(220,0,0,0.35);
    color: var(--ferrari-red);
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 13px; font-weight: 700; letter-spacing: 2px;
    padding: 5px 14px; border-radius: 2px;
}

.stTabs [data-baseweb="tab-list"] {
    background: var(--dark-gray) !important;
    border-bottom: 2px solid var(--border) !important;
    gap: 0 !important; padding: 0 60px !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 14px !important; font-weight: 600 !important;
    letter-spacing: 2px !important; text-transform: uppercase !important;
    color: var(--text-muted) !important;
    padding: 16px 28px !important;
    border-bottom: 3px solid transparent !important;
    border-radius: 0 !important; background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--ferrari-red) !important;
    border-bottom: 3px solid var(--ferrari-red) !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: var(--black) !important;
    padding: 40px 60px !important;
}

[data-testid="metric-container"] {
    background: var(--card-bg) !important;
    border: 1px solid var(--border) !important;
    border-top: 3px solid var(--ferrari-red) !important;
    border-radius: 4px !important; padding: 20px !important;
}
[data-testid="metric-container"] label {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 11px !important; letter-spacing: 2px !important;
    text-transform: uppercase !important; color: var(--text-muted) !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Racing Sans One', sans-serif !important;
    font-size: 28px !important; color: var(--ferrari-red) !important;
}

.section-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 11px; font-weight: 700; letter-spacing: 4px;
    color: var(--ferrari-red); text-transform: uppercase; margin-bottom: 6px;
}
.section-heading {
    font-family: 'Racing Sans One', sans-serif;
    font-size: 28px; color: var(--text-primary);
    margin-bottom: 28px; padding-bottom: 14px;
    border-bottom: 1px solid var(--border);
}

.stNumberInput input {
    background: var(--card-bg) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important; border-radius: 3px !important;
}
.stNumberInput input:focus {
    border-color: var(--ferrari-red) !important;
    box-shadow: 0 0 0 2px rgba(220,0,0,0.2) !important;
}
.stNumberInput label {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 12px !important; letter-spacing: 1.5px !important;
    text-transform: uppercase !important; color: var(--text-muted) !important;
}

.stButton > button {
    background: var(--ferrari-red) !important; color: white !important;
    font-family: 'Racing Sans One', sans-serif !important;
    font-size: 18px !important; letter-spacing: 3px !important;
    border: none !important; border-radius: 3px !important;
    padding: 16px 40px !important; width: 100% !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: var(--ferrari-light) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(220,0,0,0.4) !important;
}

.stProgress > div > div { background: var(--ferrari-red) !important; }
.stProgress > div { background: var(--card-bg) !important; border: 1px solid var(--border) !important; }
.stSpinner > div { border-top-color: var(--ferrari-red) !important; }
hr { border-color: var(--border) !important; margin: 30px 0 !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--black); }
::-webkit-scrollbar-thumb { background: var(--ferrari-red); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# Hero
st.markdown("""
<div class="hero">
    <div class="hero-badge">&#9679; AI Project — Deliverable 2</div>
    <div class="hero-title">F1 <span>Champion</span> Predictor</div>
    <div class="hero-sub">Multiple Linear Regression &nbsp;·&nbsp; Built from Scratch in Python</div>
    <div class="roll-pills">
        <span class="roll-pill">24L-0679</span>
        <span class="roll-pill">24L-0912</span>
        <span class="roll-pill">24L-0518</span>
    </div>
</div>
""", unsafe_allow_html=True)

FEATURE_COLS = [
    'Race_Entries','Race_Wins','Pole_Positions','Podiums',
    'Fastest_Laps','Points','Win_Rate','Podium_Rate',
    'FastLap_Rate','Points_Per_Entry'
]
FERRARI_RED  = '#DC0000'
FERRARI_GOLD = '#C8A951'
DARK_BG      = '#0A0A0A'
CARD_BG      = '#141414'
TEXT_COLOR   = '#F5F5F5'
MUTED_COLOR  = '#444444'

def set_ferrari_style():
    plt.rcParams.update({
        'figure.facecolor': DARK_BG, 'axes.facecolor': CARD_BG,
        'axes.edgecolor': '#2a2a2a', 'axes.labelcolor': TEXT_COLOR,
        'axes.titlecolor': TEXT_COLOR, 'axes.titlesize': 13,
        'axes.titleweight': 'bold', 'axes.labelsize': 10,
        'xtick.color': '#666666', 'ytick.color': '#666666',
        'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'text.color': TEXT_COLOR, 'grid.color': '#222222',
        'grid.linewidth': 0.6, 'legend.facecolor': CARD_BG,
        'legend.edgecolor': '#2a2a2a', 'legend.fontsize': 9,
    })

def predict(features, weights, bias):
    return bias + sum(w * x for w, x in zip(weights, features))

def normalize_row(row, mins, maxs):
    result = []
    for col in FEATURE_COLS:
        denom = maxs[col] - mins[col]
        result.append((row[col] - mins[col]) / denom if denom != 0 else 0.0)
    return result

def train_model(X_train, y_train, lr=0.01, epochs=10000):
    n = len(X_train)
    nf = len(X_train[0])
    weights = [0.0] * nf
    bias = 0.0
    loss_history = []
    for epoch in range(1, epochs + 1):
        wg = [0.0] * nf
        bg = 0.0
        for f, l in zip(X_train, y_train):
            err = predict(f, weights, bias) - l
            for i, x in enumerate(f): wg[i] += err * x
            bg += err
        for i in range(nf): weights[i] -= lr * (wg[i] / n)
        bias -= lr * (bg / n)
        if epoch % 500 == 0:
            mse = sum((predict(f, weights, bias)-l)**2 for f,l in zip(X_train,y_train))/n
            loss_history.append((epoch, mse))
    return weights, bias, loss_history

@st.cache_resource
def load_and_train():
    df = pd.read_csv("F1DriversDataset.csv")
    df = df[(df['Race_Entries'] > 0) & (df['Race_Starts'] > 0)].copy()
    df['Champion_int'] = df['Champion'].astype(int)
    mins = {col: df[col].min() for col in FEATURE_COLS}
    maxs = {col: df[col].max() for col in FEATURE_COLS}
    df_norm = df.copy()
    for col in FEATURE_COLS:
        d = maxs[col] - mins[col]
        df_norm[col] = (df[col] - mins[col]) / d if d != 0 else 0
    combined = list(zip(df_norm[FEATURE_COLS].values.tolist(), df_norm['Champion_int'].values.tolist()))
    random.seed(42); random.shuffle(combined)
    split = int(len(combined) * 0.8)
    tr, te = combined[:split], combined[split:]
    Xtr=[r[0] for r in tr]; ytr=[r[1] for r in tr]
    Xte=[r[0] for r in te]; yte=[r[1] for r in te]
    w, b, lh = train_model(Xtr, ytr)
    return df, w, b, mins, maxs, Xte, yte, lh

with st.spinner("Warming up the engine... 🏎️"):
    df, weights, bias, mins, maxs, X_test, y_test, loss_history = load_and_train()

tab1, tab2, tab3, tab4 = st.tabs(["📋  DATASET","📊  VISUALIZATIONS","🧪  EVALUATION","🏆  PREDICT"])

with tab1:
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Dataset Overview</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Drivers", len(df))
    c2.metric("World Champions", int(df['Champion'].sum()))
    c3.metric("Non-Champions", int((~df['Champion']).sum()))
    c4.metric("Features Used", len(FEATURE_COLS))
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Sample Records</div>', unsafe_allow_html=True)
    st.dataframe(df[['Driver']+FEATURE_COLS+['Champion']].head(10).reset_index(drop=True), use_container_width=True, height=380)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Statistical Summary</div>', unsafe_allow_html=True)
    st.dataframe(df[FEATURE_COLS].describe().round(3), use_container_width=True)

with tab2:
    set_ferrari_style()
    st.markdown('<div class="section-title">Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Data Visualizations</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6,4))
        counts = df['Champion'].value_counts()
        bars = ax.bar(['Non-Champion','Champion'], counts.values, color=[MUTED_COLOR, FERRARI_RED], width=0.5, edgecolor='none')
        ax.set_title('Champion Distribution', pad=14); ax.set_ylabel('Drivers')
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        for bar, v in zip(bars, counts.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5, str(v), ha='center', color=TEXT_COLOR, fontsize=11, fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6,4))
        ax.hist(df[df['Champion']==False]['Race_Wins'], bins=30, color=MUTED_COLOR, alpha=0.8, label='Non-Champion', edgecolor='none')
        ax.hist(df[df['Champion']==True]['Race_Wins'],  bins=15, color=FERRARI_RED, alpha=1.0, label='Champion',     edgecolor='none')
        ax.set_title('Race Wins Distribution', pad=14); ax.set_xlabel('Race Wins'); ax.set_ylabel('Drivers')
        ax.legend(); ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    corr_df = df[FEATURE_COLS].copy(); corr_df['Champion'] = df['Champion'].astype(int)
    fig, ax = plt.subplots(figsize=(11,6))
    sns.heatmap(corr_df.corr(), annot=True, fmt='.2f', cmap=sns.diverging_palette(10,220,as_cmap=True),
                center=0, ax=ax, linewidths=0.5, linecolor='#0A0A0A',
                annot_kws={'size':8}, cbar_kws={'shrink':0.8})
    ax.set_title('Feature Correlation Heatmap', pad=16, fontsize=14)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    col1, col2 = st.columns(2)
    with col1:
        top10 = df.nlargest(10,'Race_Wins')[['Driver','Race_Wins','Champion']]
        bar_colors = [FERRARI_RED if c else MUTED_COLOR for c in top10['Champion']]
        fig, ax = plt.subplots(figsize=(6,5))
        ax.barh(top10['Driver'], top10['Race_Wins'], color=bar_colors, edgecolor='none')
        ax.set_xlabel('Race Wins'); ax.set_title('Top 10 Drivers by Race Wins', pad=14); ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.legend(handles=[mpatches.Patch(color=FERRARI_RED,label='Champion'), mpatches.Patch(color=MUTED_COLOR,label='Non-Champion')])
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        ep = [x[0] for x in loss_history]; ms = [x[1] for x in loss_history]
        fig, ax = plt.subplots(figsize=(6,5))
        ax.plot(ep, ms, color=FERRARI_RED, linewidth=2.5)
        ax.fill_between(ep, ms, alpha=0.15, color=FERRARI_RED)
        ax.set_title('Training Loss Curve (MSE)', pad=14); ax.set_xlabel('Epoch'); ax.set_ylabel('MSE')
        ax.grid(alpha=0.3)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

with tab3:
    set_ferrari_style()
    st.markdown('<div class="section-title">Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Model Evaluation</div>', unsafe_allow_html=True)

    raw_preds = [predict(f, weights, bias) for f in X_test]
    y_pred = [1 if p >= 0.5 else 0 for p in raw_preds]
    y_true = [int(v) for v in y_test]
    n = len(y_true)
    mse  = sum((p-a)**2 for p,a in zip(raw_preds,y_true))/n
    rmse = math.sqrt(mse)
    mae  = sum(abs(p-a) for p,a in zip(raw_preds,y_true))/n
    ma   = sum(y_true)/n
    ss_tot = sum((a-ma)**2 for a in y_true)
    ss_res = sum((a-p)**2 for a,p in zip(y_true,raw_preds))
    r2 = 1-(ss_res/ss_tot) if ss_tot!=0 else 0
    acc       = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    cm        = confusion_matrix(y_true, y_pred)

    st.markdown('<div class="section-title">Regression Metrics</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("MAE",  f"{mae:.4f}"); c2.metric("MSE",  f"{mse:.4f}")
    c3.metric("RMSE", f"{rmse:.4f}"); c4.metric("R²",  f"{r2:.4f}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Classification Metrics</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Accuracy",  f"{acc*100:.2f}%"); c2.metric("Precision", f"{precision:.4f}")
    c3.metric("Recall",    f"{recall:.4f}");   c4.metric("F1 Score",  f"{f1:.4f}")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5,4.5))
        sns.heatmap(cm, annot=True, fmt='d', cmap=sns.light_palette(FERRARI_RED,as_cmap=True), ax=ax,
                    xticklabels=['Non-Champion','Champion'], yticklabels=['Non-Champion','Champion'],
                    linewidths=2, linecolor=DARK_BG, annot_kws={'size':14,'weight':'bold','color':TEXT_COLOR})
        ax.set_ylabel('Actual',fontsize=11); ax.set_xlabel('Predicted',fontsize=11)
        ax.set_title('Confusion Matrix',pad=14,fontsize=13)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(5,4.5))
        metrics_list = ['Accuracy','Precision','Recall','F1 Score']
        values = [acc, precision, recall, f1]
        bar_cols = [FERRARI_RED, FERRARI_GOLD, '#CC4444','#AA2222']
        bars = ax.bar(metrics_list, values, color=bar_cols, edgecolor='none', width=0.55)
        ax.set_ylim(0,1.18); ax.set_title('Classification Metrics',pad=14,fontsize=13)
        ax.set_ylabel('Score'); ax.grid(axis='y',alpha=0.3)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        for bar,val in zip(bars,values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.03, f'{val:.3f}',
                    ha='center', color=TEXT_COLOR, fontsize=11, fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()

with tab4:
    st.markdown('<div class="section-title">Prediction Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Predict Championship Potential</div>', unsafe_allow_html=True)
    st.markdown("Enter a driver's career statistics and the model will determine their championship potential.")
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Career Stats**")
        race_entries   = st.number_input("Race Entries",   min_value=0, max_value=400, value=50)
        race_wins      = st.number_input("Race Wins",      min_value=0, max_value=200, value=5)
        pole_positions = st.number_input("Pole Positions", min_value=0, max_value=250, value=3)
        points         = st.number_input("Career Points",  min_value=0.0, max_value=5000.0, value=200.0)
    with col2:
        st.markdown("**Performance**")
        podiums      = st.number_input("Podiums",      min_value=0, max_value=300, value=10)
        fastest_laps = st.number_input("Fastest Laps", min_value=0, max_value=100, value=3)
        win_rate     = st.number_input("Win Rate",     min_value=0.0, max_value=1.0, value=0.10, step=0.01)
        podium_rate  = st.number_input("Podium Rate",  min_value=0.0, max_value=1.0, value=0.20, step=0.01)
    with col3:
        st.markdown("**Rate Stats**")
        fastlap_rate     = st.number_input("Fastest Lap Rate",  min_value=0.0, max_value=1.0, value=0.06, step=0.01)
        points_per_entry = st.number_input("Points Per Entry",  min_value=0.0, max_value=50.0, value=4.0,  step=0.1)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🏁  ANALYSE DRIVER"):
        raw = {
            'Race_Entries': race_entries, 'Race_Wins': race_wins,
            'Pole_Positions': pole_positions, 'Podiums': podiums,
            'Fastest_Laps': fastest_laps, 'Points': points,
            'Win_Rate': win_rate, 'Podium_Rate': podium_rate,
            'FastLap_Rate': fastlap_rate, 'Points_Per_Entry': points_per_entry
        }
        norm_features = normalize_row(raw, mins, maxs)
        score = predict(norm_features, weights, bias)
        score_pct = max(0.0, min(1.0, score)) * 100

        if score >= 0.5:
            st.markdown(f"""
            <div style="
                margin-top:32px;
                background:linear-gradient(135deg,#1a0000 0%,#0f0000 100%);
                border:1px solid rgba(220,0,0,0.5);
                border-left:5px solid #DC0000;
                border-radius:8px;
                padding:40px;
                text-align:center;
            ">
                <div style="font-size:56px;margin-bottom:14px;">🏆</div>
                <div style="font-family:'Racing Sans One',sans-serif;font-size:30px;color:#DC0000;letter-spacing:2px;margin-bottom:8px;">WORLD CHAMPION</div>
                <div style="font-family:'Barlow Condensed',sans-serif;font-size:13px;letter-spacing:3px;color:#666;text-transform:uppercase;margin-bottom:28px;">This driver has championship potential</div>
                <div style="display:inline-block;background:rgba(220,0,0,0.15);border:1px solid rgba(220,0,0,0.4);border-radius:4px;padding:12px 32px;font-family:'Barlow Condensed',sans-serif;font-size:24px;font-weight:700;color:#F5F5F5;letter-spacing:2px;">
                    Confidence: {score_pct:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="
                margin-top:32px;
                background:linear-gradient(135deg,#0d0d0d 0%,#111 100%);
                border:1px solid rgba(255,255,255,0.07);
                border-left:5px solid #333;
                border-radius:8px;
                padding:40px;
                text-align:center;
            ">
                <div style="font-size:56px;margin-bottom:14px;">❌</div>
                <div style="font-family:'Racing Sans One',sans-serif;font-size:30px;color:#666;letter-spacing:2px;margin-bottom:8px;">NOT A CHAMPION</div>
                <div style="font-family:'Barlow Condensed',sans-serif;font-size:13px;letter-spacing:3px;color:#444;text-transform:uppercase;margin-bottom:28px;">This driver is unlikely to win the championship</div>
                <div style="display:inline-block;background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);border-radius:4px;padding:12px 32px;font-family:'Barlow Condensed',sans-serif;font-size:24px;font-weight:700;color:#666;letter-spacing:2px;">
                    Confidence: {score_pct:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
