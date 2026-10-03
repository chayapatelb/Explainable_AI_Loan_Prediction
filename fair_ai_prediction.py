# -*- coding: utf-8 -*-
"""
Fair AI Prediction  (fixed, complete version)
=============================================

Pipeline
--------
1. Install any missing libraries (pinned versions)
2. Load the UCI Adult Income dataset (with local cache + OpenML fallback)
3. Clean / encode the data and measure bias in the raw data
4. Train three models:
      - Baseline LightGBM
      - Fair LightGBM (fairlearn ExponentiatedGradient + DemographicParity)
      - Small PyTorch neural network
5. Compare accuracy and fairness gap, explain the baseline with SHAP
6. Measure inference energy with CodeCarbon
7. Save models/results and build a deployment package for the dashboard
   (fair_ai_dashboard.py must sit next to this file)

Run:
    python fair_ai_prediction.py
Then:
    streamlit run fair_ai_dashboard.py
"""

# ----------------------------------------------------------------------------
# 0. Install missing libraries
# ----------------------------------------------------------------------------
import importlib.util
import os
import subprocess
import sys

# import-name -> pip requirement. Pinned to a combination that works together.
REQUIREMENTS = {
    "numpy": "numpy==1.26.4",
    "pandas": "pandas==2.2.2",
    "sklearn": "scikit-learn==1.5.2",
    "matplotlib": "matplotlib==3.10.0",
    "seaborn": "seaborn==0.13.2",
    "joblib": "joblib>=1.3.0",
    "lightgbm": "lightgbm==4.2.0",
    "shap": "shap==0.45.1",
    "fairlearn": "fairlearn==0.9.0",
    "torch": "torch==2.3.1",
    "codecarbon": "codecarbon==2.3.4",
    "streamlit": "streamlit>=1.28.0,<1.50",
    "plotly": "plotly>=5.18.0",
    "psutil": "psutil>=5.9.0",
}


def ensure_packages():
    """pip-install only the packages that are missing (never uninstalls anything)."""
    if os.environ.get("FAIR_AI_SKIP_INSTALL") == "1":
        return
    missing = [spec for mod, spec in REQUIREMENTS.items()
               if importlib.util.find_spec(mod) is None]
    if missing:
        print("Installing missing packages:", ", ".join(missing))
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


ensure_packages()

# ----------------------------------------------------------------------------
# Imports
# (heavy / optional libraries - lightgbm, fairlearn, shap, torch, codecarbon -
#  are imported inside the functions that use them)
# ----------------------------------------------------------------------------
import warnings
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def _in_notebook():
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ in ("ZMQInteractiveShell", "Shell")
    except Exception:
        return False


IN_NOTEBOOK = _in_notebook()

import matplotlib
if not IN_NOTEBOOK:
    matplotlib.use("Agg")          # no GUI needed when running as a plain script
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
SEED = 42
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR = BASE_DIR / "data"

ADULT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
]
FEATURES = ["age", "education-num", "hours-per-week", "sex", "race"]
TARGET = "income_binary"
SENSITIVE = "sex"

DASHBOARD_REQUIREMENTS = """\
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.5.2
joblib>=1.3.0
lightgbm==4.2.0
shap==0.45.1
fairlearn==0.9.0
plotly>=5.18.0
codecarbon==2.3.4
psutil>=5.9.0
matplotlib==3.10.0
streamlit>=1.28.0,<1.50
"""


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def show(obj):
    """display() in notebooks, plain print otherwise."""
    if IN_NOTEBOOK:
        from IPython.display import display
        display(obj)
    elif hasattr(obj, "to_string"):
        print(obj.to_string())
    else:
        print(obj)


def finish_plot(fig, path):
    """Save a figure; also show it when running in a notebook."""
    fig.savefig(path, dpi=300, bbox_inches="tight")
    if IN_NOTEBOOK:
        plt.show()
    plt.close(fig)


def positive_class_shap(values):
    """Return a 2-D (samples x features) SHAP array for the positive class,
    whatever format the installed shap version returns."""
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    return values


# ----------------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------------
def load_data(source=ADULT_URL):
    """Load the Adult dataset. Uses a local cache; falls back to OpenML."""
    DATA_DIR.mkdir(exist_ok=True)
    cache = DATA_DIR / "adult.csv"
    if source == ADULT_URL and cache.exists():
        print(f"Using cached dataset: {cache}")
        return pd.read_csv(cache)

    print("Loading dataset...")
    try:
        # skipinitialspace=True already strips the leading space, so the
        # missing-value marker is "?" (NOT " ?").
        df = pd.read_csv(source, header=None, names=ADULT_COLUMNS,
                         na_values="?", skipinitialspace=True)
    except Exception as exc:
        print(f"Could not download from UCI ({exc}). Trying OpenML...")
        from sklearn.datasets import fetch_openml
        raw = fetch_openml("adult", version=2, as_frame=True).frame
        df = raw.rename(columns={"class": "income"})[ADULT_COLUMNS].copy()
        for col in df.columns:
            if str(df[col].dtype) == "category":
                df[col] = df[col].astype(object)
    if source == ADULT_URL:
        df.to_csv(cache, index=False)

    print("Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    return df


def preprocess(df):
    """Drop missing rows, strip whitespace and encode everything numerically."""
    print("Dataset overview:")
    print(f"  Total records:      {len(df)}")
    print(f"  Number of columns:  {len(df.columns)}")
    print(f"  Missing values:     {int(df.isnull().sum().sum())}")

    df_clean = df.dropna().copy()
    print(f"Records after removing missing values: {len(df_clean)}")

    for col in df_clean.select_dtypes(include=["object", "string"]).columns:
        df_clean[col] = df_clean[col].str.strip()

    df_fixed = pd.DataFrame({
        "age": df_clean["age"],
        "education-num": df_clean["education-num"],
        "hours-per-week": df_clean["hours-per-week"],
        "sex": (df_clean["sex"] == "Male").astype(int),
        "race": (df_clean["race"] == "White").astype(int),
        TARGET: (df_clean["income"].str.rstrip(".") == ">50K").astype(int),
    }).reset_index(drop=True)

    print("Encoding check:")
    print("  Sex   :", df_fixed["sex"].value_counts().to_dict())
    print("  Race  :", df_fixed["race"].value_counts().to_dict())
    print("  Income:", df_fixed[TARGET].value_counts().to_dict())
    print(f"  High-income rate: {df_fixed[TARGET].mean() * 100:.1f}%")
    return df_fixed


def analyze_bias(df_fixed):
    male_rate = df_fixed.loc[df_fixed["sex"] == 1, TARGET].mean()
    female_rate = df_fixed.loc[df_fixed["sex"] == 0, TARGET].mean()
    print("Bias in the data:")
    print(f"  High-income rate, males:   {male_rate:.3f}")
    print(f"  High-income rate, females: {female_rate:.3f}")
    print(f"  Difference:                {male_rate - female_rate:.3f}")
    if male_rate > 0:
        print(f"  Disparate impact ratio:    {female_rate / male_rate:.3f}")
    return male_rate, female_rate


def split_and_scale(df_fixed):
    X = df_fixed[FEATURES]
    y = df_fixed[TARGET]
    sensitive = df_fixed[SENSITIVE]

    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, sensitive, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f"Training set: {X_train.shape}   Test set: {X_test.shape}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return (X_train_scaled, X_test_scaled, y_train, y_test,
            sens_train, sens_test, scaler)


# ----------------------------------------------------------------------------
# 2. Models
# ----------------------------------------------------------------------------
def train_baseline(X_train, y_train):
    from lightgbm import LGBMClassifier
    model = LGBMClassifier(n_estimators=100, random_state=SEED, verbose=-1)
    model.fit(X_train, y_train)
    return model


def train_fair(X_train, y_train, sens_train):
    from fairlearn.reductions import DemographicParity, ExponentiatedGradient
    from lightgbm import LGBMClassifier
    model = ExponentiatedGradient(
        LGBMClassifier(n_estimators=100, random_state=SEED, verbose=-1),
        DemographicParity(),
    )
    model.fit(X_train, y_train, sensitive_features=sens_train)
    return model


def build_net(input_size):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(input_size, 64), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 1), nn.Sigmoid(),
    )


def train_neural_net(X_train, y_train, epochs=30, batch_size=256, lr=1e-3):
    """Mini-batch training (the original full-batch / 50-step loop under-fit)."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(SEED)
    net = build_net(X_train.shape[1])
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train.to_numpy(), dtype=torch.float32).unsqueeze(1)),
        batch_size=batch_size, shuffle=True,
    )
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    net.train()
    for epoch in range(epochs):
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(net(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, loss: {total / len(loader.dataset):.4f}")
    return net


def nn_predict(net, X):
    import torch
    net.eval()
    with torch.no_grad():
        proba = net(torch.tensor(X, dtype=torch.float32)).numpy().ravel()
    return (proba > 0.5).astype(int)


def evaluate(name, y_true, y_pred, sens):
    from fairlearn.metrics import demographic_parity_difference
    acc = accuracy_score(y_true, y_pred)
    gap = demographic_parity_difference(y_true, y_pred, sensitive_features=sens)
    print(f"{name} results:")
    print(f"  Accuracy:     {acc:.3f}")
    print(f"  Fairness gap: {gap:.3f}")
    print(f"  High-income predictions: {int(np.sum(y_pred))} of {len(y_pred)}")
    return acc, gap


# ----------------------------------------------------------------------------
# 3. Explainability, energy, outputs
# ----------------------------------------------------------------------------
def explain_baseline(model, X_test_scaled, max_samples=2000):
    import shap
    X_eval = pd.DataFrame(X_test_scaled[:max_samples], columns=FEATURES)
    explainer = shap.TreeExplainer(model)
    shap_values = positive_class_shap(explainer.shap_values(X_eval))
    print(f"SHAP values shape: {shap_values.shape}")

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_eval, show=False)
    plt.title("SHAP Feature Importance - Baseline Model")
    plt.tight_layout()
    finish_plot(plt.gcf(), RESULTS_DIR / "shap_summary.png")


def plot_comparison(final_comparison):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    final_comparison.plot(x="Model", y="Accuracy", kind="bar", ax=ax1,
                          color="skyblue", legend=False, rot=0)
    ax1.set_title("Model Accuracy Comparison")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1)

    gaps = final_comparison["Fairness_Gap"].abs()
    final_comparison.assign(Fairness_Gap_Abs=gaps).plot(
        x="Model", y="Fairness_Gap_Abs", kind="bar", ax=ax2,
        color="lightcoral", legend=False, rot=0)
    ax2.set_title("Fairness Gap Comparison (absolute values)")
    ax2.set_ylabel("Fairness gap")
    ax2.set_ylim(0, max(0.2, float(gaps.max()) * 1.15))

    plt.tight_layout()
    finish_plot(fig, RESULTS_DIR / "final_comparison.png")


def measure_inference_energy(model, X, n_calls=1000, batch=10):
    """CO2 (kg) for n_calls predict() calls on `batch` rows. NaN if unavailable."""
    try:
        from codecarbon import EmissionsTracker
        tracker = EmissionsTracker(save_to_file=False, log_level="error")
        print("Measuring inference energy consumption...")
        tracker.start()
        for _ in range(n_calls):
            model.predict(X[:batch])
        emissions = tracker.stop()
        return float(emissions) if emissions is not None else 0.0
    except Exception as exc:
        print(f"Energy tracking unavailable: {exc}")
        return float("nan")


def save_artifacts(baseline_model, fair_model, net, scaler, df_fixed,
                   X_train_scaled, X_test_scaled, y_test, sens_test):
    import torch
    joblib.dump(baseline_model, MODELS_DIR / "baseline_model.pkl")
    joblib.dump(fair_model, MODELS_DIR / "fair_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    torch.save(net.state_dict(), MODELS_DIR / "neural_network.pth")

    # Files the Streamlit dashboard needs
    rng = np.random.RandomState(SEED)
    background = X_train_scaled[rng.choice(len(X_train_scaled), size=100, replace=False)]
    joblib.dump(X_test_scaled, MODELS_DIR / "X_test_scaled.pkl")
    joblib.dump(np.asarray(y_test), MODELS_DIR / "y_test.pkl")
    joblib.dump(np.asarray(sens_test), MODELS_DIR / "sens_test.pkl")
    joblib.dump(FEATURES, MODELS_DIR / "feature_names.pkl")
    joblib.dump(background, MODELS_DIR / "shap_background.pkl")
    df_fixed.to_csv(RESULTS_DIR / "processed_data.csv", index=False)
    print("All models and results saved (see the 'models' and 'results' folders).")


def build_deployment_package():
    """Zip dashboard + requirements + README + model files for Streamlit Cloud."""
    dashboard = BASE_DIR / "fair_ai_dashboard.py"
    if not dashboard.exists():
        print("fair_ai_dashboard.py not found next to this script - skipping package.")
        return None

    readme = (
        "# Fair AI Dashboard with Explainability & Energy Tracking\n\n"
        "## Features\n"
        "- Fair model (demographic parity) and unconstrained baseline\n"
        "- SHAP explanations for every prediction\n"
        "- Inference time / CPU / memory / CO2 tracking\n"
        "- Fairness metrics (demographic parity difference)\n\n"
        "## Run locally\n"
        "```bash\n"
        "pip install -r requirements.txt\n"
        "streamlit run fair_ai_dashboard.py\n"
        "```\n\n"
        "## Deploy on Streamlit Cloud\n"
        "1. Push this folder to GitHub\n"
        "2. Connect the repo in Streamlit Cloud\n"
        "3. Set the main file to `fair_ai_dashboard.py`\n"
    )

    zip_path = BASE_DIR / "fair_ai_dashboard_deployment.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(dashboard, "fair_ai_dashboard.py")
        zf.writestr("requirements.txt", DASHBOARD_REQUIREMENTS)
        zf.writestr("README.md", readme)
        for pkl in sorted(MODELS_DIR.glob("*.pkl")):
            zf.write(pkl, f"models/{pkl.name}")
    print(f"Created deployment package: {zip_path}")

    try:  # only inside Google Colab
        from google.colab import files
        files.download(str(zip_path))
    except Exception:
        pass
    return zip_path


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    np.random.seed(SEED)
    for folder in (MODELS_DIR, RESULTS_DIR, DATA_DIR):
        folder.mkdir(exist_ok=True)

    # --- data -----------------------------------------------------------
    df = load_data()
    print("\nFirst 5 rows:")
    show(df.head())
    df_fixed = preprocess(df)
    print()
    male_rate, female_rate = analyze_bias(df_fixed)
    print()
    (X_train, X_test, y_train, y_test,
     sens_train, sens_test, scaler) = split_and_scale(df_fixed)

    # --- models ---------------------------------------------------------
    print("\nTraining baseline LightGBM...")
    baseline_model = train_baseline(X_train, y_train)
    acc_base, gap_base = evaluate("Baseline", y_test, baseline_model.predict(X_test), sens_test)

    print("\nTraining fair model (demographic parity)...")
    fair_model = train_fair(X_train, y_train, sens_train)
    # random_state makes the randomized fair classifier reproducible
    y_pred_fair = fair_model.predict(X_test, random_state=SEED)
    acc_fair, gap_fair = evaluate("Fair model", y_test, y_pred_fair, sens_test)

    print("\nTraining neural network...")
    net = train_neural_net(X_train, y_train)
    acc_nn, gap_nn = evaluate("Neural network", y_test, nn_predict(net, X_test), sens_test)

    # --- comparison -----------------------------------------------------
    final_comparison = pd.DataFrame({
        "Model": ["Baseline", "Fair Model", "Neural Network"],
        "Accuracy": [acc_base, acc_fair, acc_nn],
        "Fairness_Gap": [gap_base, gap_fair, gap_nn],
    })
    final_comparison["Overall_Score"] = final_comparison["Accuracy"] - final_comparison["Fairness_Gap"]
    best_idx = final_comparison["Overall_Score"].idxmax()
    best = final_comparison.loc[best_idx]

    print("\nFinal model comparison:")
    show(final_comparison)
    print(f"\nBest overall model: {best['Model']} "
          f"(accuracy {best['Accuracy']:.3f}, fairness gap {best['Fairness_Gap']:.3f})")
    print(f"Accuracy change (fair vs baseline):  {acc_fair - acc_base:+.3f}")
    print(f"Fairness improvement (gap reduction): {gap_base - gap_fair:+.3f}")

    # --- explainability, plots, energy ----------------------------------
    print("\nCreating SHAP explanations...")
    explain_baseline(baseline_model, X_test)
    plot_comparison(final_comparison)
    emissions = measure_inference_energy(baseline_model, X_test)
    print(f"CO2 for 1000 predict() calls (10 rows each): {emissions:.6f} kg")

    # --- save everything ------------------------------------------------
    final_comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    save_artifacts(baseline_model, fair_model, net, scaler, df_fixed,
                   X_train, X_test, y_test, sens_test)

    summary = (
        "FAIR AI PROJECT - FINAL RESULTS SUMMARY\n"
        + "=" * 50 + "\n\n"
        "DATASET\n"
        f"  Total records:      {len(df_fixed)}\n"
        f"  Features:           {len(FEATURES)}\n"
        f"  High-income rate:   {df_fixed[TARGET].mean() * 100:.1f}%\n"
        f"  Bias in data:       {male_rate - female_rate:.3f}\n\n"
        "MODEL PERFORMANCE\n"
        f"{final_comparison.to_string()}\n\n"
        f"BEST MODEL: {best['Model']}\n"
        f"  Accuracy:      {best['Accuracy']:.3f}\n"
        f"  Fairness gap:  {best['Fairness_Gap']:.3f}\n"
        f"  Overall score: {best['Overall_Score']:.3f}\n\n"
        "ENERGY\n"
        f"  CO2, 1000 predict() calls: {emissions:.6f} kg\n"
        f"  CO2 per call:              {emissions / 1000:.8f} kg\n"
    )
    (RESULTS_DIR / "summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)

    build_deployment_package()


if __name__ == "__main__":
    main()
