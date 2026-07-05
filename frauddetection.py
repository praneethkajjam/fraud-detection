import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (f1_score, roc_auc_score, roc_curve,classification_report, confusion_matrix,ConfusionMatrixDisplay, precision_recall_curve)

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

sns.set_style("whitegrid")
RANDOM_STATE = 42
DATA_PATH = r"C:\Users\prane\Downloads\creditcard.csv"
OUT_DIR = "outputsfrauddetection"
os.makedirs(OUT_DIR, exist_ok=True)

# 1. LOAD DATA

def load_data():
    if os.path.exists(DATA_PATH):
        print(f"Loading real dataset from {DATA_PATH} ...")
        return pd.read_csv(DATA_PATH)
    rng = np.random.default_rng(RANDOM_STATE)
    n_normal = 15000
    n_fraud = 80  # keep it heavily imbalanced like the real data (~0.5%)

    # "Normal" transactions cluster around 0 in PCA-space
    normal_v = rng.normal(0, 1, size=(n_normal, 28))
    normal_amount = np.abs(rng.gamma(2.0, 40, n_normal))
    normal_time = rng.integers(0, 172800, n_normal)  # 2 days in seconds

    # Fraudulent transactions are shifted / more spread out (as PCA
    # components tend to look for real fraud cases)
    fraud_v = rng.normal(0, 1, size=(n_fraud, 28)) + rng.normal(3, 1, size=(n_fraud, 28))
    fraud_amount = np.abs(rng.gamma(3.0, 120, n_fraud))
    fraud_time = rng.integers(0, 172800, n_fraud)

    v_cols = [f"V{i}" for i in range(1, 29)]
    df_normal = pd.DataFrame(normal_v, columns=v_cols)
    df_normal["Time"] = normal_time
    df_normal["Amount"] = normal_amount
    df_normal["Class"] = 0

    df_fraud = pd.DataFrame(fraud_v, columns=v_cols)
    df_fraud["Time"] = fraud_time
    df_fraud["Amount"] = fraud_amount
    df_fraud["Class"] = 1

    df = pd.concat([df_normal, df_fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return df

# 2. EDA - CLASS IMBALANCE

def plot_class_balance(df):
    counts = df["Class"].value_counts()
    plt.figure(figsize=(5, 4))
    sns.barplot(x=counts.index.map({0: "Normal", 1: "Fraud"}), y=counts.values,
                hue=counts.index.map({0: "Normal", 1: "Fraud"}), palette="Set2", legend=False)
    plt.title("Class Distribution (highly imbalanced)")
    plt.ylabel("Number of transactions")
    for i, v in enumerate(counts.values):
        plt.text(i, v, f"{v:,}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/class_balance.png", dpi=150)
    plt.close()
    fraud_pct = df["Class"].mean() * 100
    print(f"Fraud rate in dataset: {fraud_pct:.3f}%")

# 3. FEATURE ENGINEERING

def engineer_features(df):
    df = df.copy()
    # Scale Amount and Time - they're on a very different scale than
    # the PCA components V1-V28
    scaler = StandardScaler()
    df["ScaledAmount"] = scaler.fit_transform(df[["Amount"]])
    df["ScaledTime"] = scaler.fit_transform(df[["Time"]])
    df = df.drop(columns=["Amount", "Time"])

    # Hour of day (transactions at unusual hours are riskier)
    seconds_in_day = 24 * 3600
    df["HourOfDay"] = ((df["ScaledTime"] * 0) + (df.index * 0))  
    return df

# 4. HANDLE CLASS IMBALANCE (SMOTE)

def balance_training_data(X_train, y_train):
    if HAS_SMOTE:
        print("Applying SMOTE to balance the training set...")
        sm = SMOTE(random_state=RANDOM_STATE)
        X_res, y_res = sm.fit_resample(X_train, y_train)
    else:
        print("imbalanced-learn not installed -> using simple random "
              "oversampling instead of SMOTE.")
        X_train = X_train.copy()
        X_train["target"] = y_train.values
        majority = X_train[X_train["target"] == 0]
        minority = X_train[X_train["target"] == 1]
        minority_upsampled = minority.sample(len(majority), replace=True,
                                              random_state=RANDOM_STATE)
        upsampled = pd.concat([majority, minority_upsampled])
        y_res = upsampled["target"]
        X_res = upsampled.drop(columns="target")

    print(f"Class balance after resampling: {pd.Series(y_res).value_counts().to_dict()}")
    return X_res, y_res

# 5. MODELING

def train_isolation_forest(X_train, contamination):
    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=RANDOM_STATE,
    )
    iso.fit(X_train)
    return iso
def train_classifier(X_train, y_train):
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1
    )
    clf.fit(X_train, y_train)
    return clf

# 6. MAIN

def main():
    df = load_data()
    plot_class_balance(df)
    df = engineer_features(df).drop(columns=["HourOfDay"])  # drop placeholder col

    target = "Class"
    feature_cols = [c for c in df.columns if c != target]
    X = df[feature_cols]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    # ---- Model 1: Isolation Forest (unsupervised anomaly detection) ----
    contamination = max(y_train.mean(), 0.001)
    iso_model = train_isolation_forest(X_train, contamination)
    iso_pred_raw = iso_model.predict(X_test)          # -1 = anomaly, 1 = normal
    iso_pred = np.where(iso_pred_raw == -1, 1, 0)      # convert to 1 = fraud
    iso_scores = -iso_model.score_samples(X_test)      # higher = more anomalous

    print("\n--- Isolation Forest (unsupervised) ---")
    print(classification_report(y_test, iso_pred, digits=3))
    iso_auc = roc_auc_score(y_test, iso_scores)
    iso_f1 = f1_score(y_test, iso_pred)
    print(f"AUC-ROC: {iso_auc:.4f} | F1-score: {iso_f1:.4f}")

    # ---- Model 2: RandomForest classifier trained on SMOTE-balanced data ----
    X_train_bal, y_train_bal = balance_training_data(X_train, y_train)
    clf = train_classifier(X_train_bal, y_train_bal)
    clf_prob = clf.predict_proba(X_test)[:, 1]
    clf_pred = (clf_prob >= 0.5).astype(int)

    print("\n--- Random Forest (trained on SMOTE-balanced data) ---")
    print(classification_report(y_test, clf_pred, digits=3))
    clf_auc = roc_auc_score(y_test, clf_prob)
    clf_f1 = f1_score(y_test, clf_pred)
    print(f"AUC-ROC: {clf_auc:.4f} | F1-score: {clf_f1:.4f}")

    # ---- Plots ----
    plt.figure(figsize=(7, 6))
    fpr, tpr, _ = roc_curve(y_test, iso_scores)
    plt.plot(fpr, tpr, label=f"Isolation Forest (AUC={iso_auc:.3f})")
    fpr, tpr, _ = roc_curve(y_test, clf_prob)
    plt.plot(fpr, tpr, label=f"Random Forest + SMOTE (AUC={clf_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Fraud Detection Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/roc_curve.png", dpi=150)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_test, clf_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, color="darkorange")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - Random Forest + SMOTE")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/precision_recall_curve.png", dpi=150)
    plt.close()

    cm = confusion_matrix(y_test, clf_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Normal", "Fraud"]).plot(cmap="Reds")
    plt.title("Confusion Matrix - Random Forest + SMOTE")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=150)
    plt.close()

    print(f"\nPlots saved to '{OUT_DIR}/' folder.")
    print("\nSummary:")
    print(f"  Isolation Forest : AUC={iso_auc:.4f}, F1={iso_f1:.4f}")
    print(f"  Random Forest+SMOTE: AUC={clf_auc:.4f}, F1={clf_f1:.4f}")


if __name__ == "__main__":
    main()