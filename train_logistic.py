import argparse
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import pickle


# ============================================================
# PWM LOADING
# ============================================================

def load_pwm(path, target_tf="CTCF"):
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")

            if parts[0] != target_tf:
                continue

            L = int(parts[1])

            A = [float(x) for x in parts[2].strip().rstrip(",").split(",")]
            C = [float(x) for x in parts[3].strip().rstrip(",").split(",")]
            G = [float(x) for x in parts[4].strip().rstrip(",").split(",")]
            T = [float(x) for x in parts[5].strip().rstrip(",").split(",")]

            pwm = np.array([A, C, G, T], dtype=float)
            assert pwm.shape == (4, L)
            return pwm

    raise ValueError(f"TF {target_tf} not found in PWM file {path}!")


def score_pwm(onehot_seq, pwm):
    L_seq = onehot_seq.shape[1]
    L_pwm = pwm.shape[1]

    max_score = -np.inf

    for start in range(L_seq - L_pwm + 1):
        window = onehot_seq[:, start:start + L_pwm]
        score = np.sum(window * pwm)
        if score > max_score:
            max_score = score

    return max_score


# ============================================================
# MAIN TRAINING LOGIC
# ============================================================

def train_model(use_pwm, data_path, pwm_path):

    print("\n=======================================")
    print(f"Training Logistic Regression (use_pwm={use_pwm})")
    print("=======================================\n")

    # --------------------------
    # Load dataset
    # --------------------------
    data = np.load(data_path)

    X_struct = data["X_struct"]      # (N, 6) - MGW, ProT, Roll only
    X_seq = data["X_seq"]            # (N, 4, L)
    y = data["y"]                    # (N,)

    print("Dataset loaded.")
    print("Structure shape:", X_struct.shape)
    print("Sequence shape:", X_seq.shape)
    print("Labels shape:", y.shape)

    # Check for NaN values
    nan_count = np.isnan(X_struct).sum()
    print(f"NaN values in structural features: {nan_count} ({100*nan_count/X_struct.size:.1f}%)")

    # --------------------------
    # Load & compute PWM
    # --------------------------
    if use_pwm:
        pwm = load_pwm(pwm_path)
        print("PWM loaded:", pwm.shape)

        pwm_scores = np.array([
            score_pwm(X_seq[i], pwm)
            for i in range(len(X_seq))
        ]).reshape(-1, 1)

        print("PWM scores shape:", pwm_scores.shape)

        X = np.concatenate([X_struct, pwm_scores], axis=1)
        model_name = "logistic_model_with_pwm.pkl"
        scaler_name = "logistic_scaler_with_pwm.pkl"
        imputer_name = "logistic_imputer_with_pwm.pkl"
        print(f"Using 6 structural features + 1 PWM feature = 7 total features")

    else:
        print("Training WITHOUT PWM features.")
        X = X_struct.copy()
        model_name = "logistic_model_no_pwm.pkl"
        scaler_name = "logistic_scaler_no_pwm.pkl"
        imputer_name = "logistic_imputer_no_pwm.pkl"
        print(f"Using 6 structural features only")

    print("Final feature matrix:", X.shape)

    # ============================================================
    # 5-FOLD CROSS VALIDATION
    # ============================================================

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    accs, aurocs, auprcs = [], [], []

    fold = 1

    for train_idx, test_idx in kf.split(X):

        print(f"\n===== Fold {fold} =====")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Impute missing values FIRST (replace NaN with mean)
        imputer = SimpleImputer(strategy='mean')
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)

        # Then standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_test_scaled = scaler.transform(X_test_imputed)

        # Train logistic regression
        model = LogisticRegression(
            penalty='l2',
            C=1.0,
            solver='lbfgs',
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train_scaled, y_train)

        # Predictions
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        auroc = roc_auc_score(y_test, y_prob)
        auprc = average_precision_score(y_test, y_prob)

        accs.append(acc)
        aurocs.append(auroc)
        auprcs.append(auprc)

        print(f"Fold {fold} Accuracy: {acc:.4f}")
        print(f"Fold {fold} AUROC:    {auroc:.4f}")
        print(f"Fold {fold} AUPRC:    {auprc:.4f}")

        # --------------------------------------
        # Save ROC curve
        # --------------------------------------
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={auroc:.3f}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Logistic Regression ROC – Fold {fold}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"logistic_roc_fold{fold}_usePWM_{use_pwm}.png", dpi=200)
        plt.close()

        # --------------------------------------
        # Save PR curve
        # --------------------------------------
        prec, recall, _ = precision_recall_curve(y_test, y_prob)
        plt.figure()
        plt.plot(recall, prec, label=f"AUPRC={auprc:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Logistic Regression PR – Fold {fold}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"logistic_pr_fold{fold}_usePWM_{use_pwm}.png", dpi=200)
        plt.close()

        fold += 1

    # ============================================================
    # SUMMARY
    # ============================================================

    print("\n===== 5-FOLD CV SUMMARY =====")
    print(f"Accuracy: mean={np.mean(accs):.4f}, std={np.std(accs):.4f}")
    print(f"AUROC:    mean={np.mean(aurocs):.4f}, std={np.std(aurocs):.4f}")
    print(f"AUPRC:    mean={np.mean(auprcs):.4f}, std={np.std(auprcs):.4f}")

    # ============================================================
    # Train FINAL model on full dataset
    # ============================================================

    print("\n===== Training Final Model =====")

    # Impute missing values on full dataset
    final_imputer = SimpleImputer(strategy='mean')
    X_imputed = final_imputer.fit_transform(X)

    # Standardize
    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X_imputed)

    # Train final model
    final_model = LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='lbfgs',
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )

    final_model.fit(X_scaled, y)

    # Save model, scaler, and imputer
    with open(model_name, 'wb') as f:
        pickle.dump(final_model, f)
    
    with open(scaler_name, 'wb') as f:
        pickle.dump(final_scaler, f)
    
    with open(imputer_name, 'wb') as f:
        pickle.dump(final_imputer, f)

    print(f"\nSaved final model → {model_name}")
    print(f"Saved scaler → {scaler_name}")
    print(f"Saved imputer → {imputer_name}")

    # Print feature coefficients
    print("\n===== Feature Importance (Coefficients) =====")
    if use_pwm:
        feature_names = [
            "MGW_mean", "MGW_std",
            "ProT_mean", "ProT_std",
            "Roll_mean", "Roll_std",
            "PWM_score"
        ]
    else:
        feature_names = [
            "MGW_mean", "MGW_std",
            "ProT_mean", "ProT_std",
            "Roll_mean", "Roll_std"
        ]
    
    coefs = final_model.coef_[0]
    for name, coef in zip(feature_names, coefs):
        print(f"  {name:15s}: {coef:+.4f}")


# ============================================================
# CLI ARGPARSE ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--use-pwm",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Whether to include PWM score as a feature."
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="datasets_chr1/ctcf_chr1_dataset_struct.npz",
        help="Path to dataset file."
    )

    parser.add_argument(
        "--pwm-path",
        type=str,
        default="data/factorbook/factorbookMotifPwm.txt",
        help="Path to PWM file."
    )

    args = parser.parse_args()

    train_model(
        use_pwm=args.use_pwm,
        data_path=args.data_path,
        pwm_path=args.pwm_path
    )