import argparse
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt


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

def train_model(mode, data_path, pwm_path):

    print("\n=======================================")
    print(f"Training XGBoost model (mode={mode})")
    print("=======================================\n")

    # --------------------------
    # Load dataset
    # --------------------------
    data = np.load(data_path)

    X_struct = data["X_struct"]      # (N, 10)
    X_seq = data["X_seq"]            # (N, 4, 20)
    y = data["y"]                    # (N,)

    print("Dataset loaded.")
    print("Structure shape:", X_struct.shape)
    print("Sequence shape:", X_seq.shape)
    print("Labels shape:", y.shape)
    
    # Check for NaN
    nan_count = np.isnan(X_struct).sum()
    print(f"NaN values in structural features: {nan_count} ({100*nan_count/X_struct.size:.1f}%)")

    # ============================================================
    # PWM feature construction
    # ============================================================

    if mode in ["pwm", "struct_pwm"]:
        pwm = load_pwm(pwm_path)
        print("PWM loaded:", pwm.shape)

        pwm_scores = np.array([
            score_pwm(X_seq[i], pwm)
            for i in range(len(X_seq))
        ]).reshape(-1, 1)

        print("PWM scores shape:", pwm_scores.shape)

    # ============================================================
    # MODE SELECTION
    # ============================================================

    print(f"\nSelected MODE: {mode}")

    # STRUCT features (10)
    if mode == "struct":
        X = X_struct.copy()
        print("Using STRUCT features only:", X.shape)

    # PWM only
    elif mode == "pwm":
        X = pwm_scores.copy()
        print("Using PWM feature only:", X.shape)

    # STRUCT + PWM
    elif mode == "struct_pwm":
        X = np.concatenate([X_struct, pwm_scores], axis=1)
        print("Using STRUCT + PWM features:", X.shape)

    else:
        raise ValueError(f"Unknown mode: {mode}")

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

        model = XGBClassifier(
            max_depth=4,
            learning_rate=0.05,
            n_estimators=500,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=1,
            objective="binary:logistic",
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=42
        )

        model.fit(X_train, y_train)

        # predictions
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)

        # metrics
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
        plt.title(f"XGBoost ROC Curve – Fold {fold}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"xgb_roc_fold{fold}_mode_{mode}.png", dpi=200)
        plt.close()

        # --------------------------------------
        # Save PR curve
        # --------------------------------------
        prec, recall, _ = precision_recall_curve(y_test, y_prob)
        plt.figure()
        plt.plot(recall, prec, label=f"AUPRC={auprc:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"XGBoost PR Curve – Fold {fold}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"xgb_pr_fold{fold}_mode_{mode}.png", dpi=200)
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

    final_model = XGBClassifier(
        max_depth=4,
        learning_rate=0.05,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1,
        objective="binary:logistic",
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=42
    )

    final_model.fit(X, y)
    final_model.save_model(f"xgb_final_model_mode_{mode}.json")

    print(f"\nSaved final model → xgb_final_model_mode_{mode}.json")
    
    # ============================================================
    # Feature importance
    # ============================================================
    
    print("\n===== Feature Importance =====")

    feature_names_struct = [
        "Buckle_mean","Buckle_std",
        "MGW_mean","MGW_std",
        "Opening_mean","Opening_std",
        "ProT_mean","ProT_std",
        "Roll_mean","Roll_std"
    ]

    if mode == "struct":
        feature_names = feature_names_struct

    elif mode == "pwm":
        feature_names = ["PWM_score"]

    elif mode == "struct_pwm":
        feature_names = feature_names_struct + ["PWM_score"]

    importances = final_model.feature_importances_

    for name, imp in zip(feature_names, importances):
        print(f"  {name:15s}: {imp:.4f}")


# ============================================================
# CLI ARGPARSE ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        choices=["struct", "pwm", "struct_pwm"],
        default="struct",
        help="Feature mode: struct, pwm, or struct_pwm."
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="datasets_chr1/ctcf_chr1_dataset_struct.npz"
    )

    parser.add_argument(
        "--pwm-path",
        type=str,
        default="data/factorbook/factorbookMotifPwm.txt"
    )

    args = parser.parse_args()

    train_model(
        mode=args.mode,
        data_path=args.data_path,
        pwm_path=args.pwm_path
    )
