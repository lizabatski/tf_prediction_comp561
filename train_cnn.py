#!/usr/bin/env python

"""
CNN training + full TF-MoDISco interpretation for TF binding.

Expected dataset (.npz):
    - X_seq: (N, 4, L) one-hot DNA
    - y:     (N,)      binary labels (0/1)
"""

# =====================================================================
# IMPORTS (order matters for hdf5plugin / modiscolite)
# =====================================================================

# Must come BEFORE anything that uses h5py inside modiscolite
import hdf5plugin            # needed so h5py (inside modiscolite) can read compressed HDF5
import h5py                  # safe to import now

from modiscolite.tfmodisco import TFMoDISco

# Standard imports
import argparse
import pickle
import numpy as np

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# =====================================================================
# Utility
# =====================================================================

def sigmoid_np(x: np.ndarray) -> np.ndarray:
    """Stable-ish sigmoid in numpy."""
    return 1.0 / (1.0 + np.exp(-x))


# =====================================================================
# CNN MODEL
# =====================================================================

class CNN(nn.Module):
    """
    Simple 1D CNN for binary classification on (B, C, L) inputs.
    Returns logits of shape (B,) (no sigmoid inside).
    """

    def __init__(self, input_length: int, num_channels: int) -> None:
        super().__init__()

        # First motif-learning convolution
        self.motif_conv = nn.Conv1d(
            in_channels=num_channels,
            out_channels=32,
            kernel_size=8,
            padding=0,      # valid
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # Second conv (roughly "same" via padding=2, kernel_size=4)
        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=32,
            kernel_size=4,
            padding=2,
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # Compute flattened size after conv/pool stack
        temp_size = input_length - 8 + 1   # conv1 (valid)
        temp_size = temp_size // 2         # pool1
        temp_size = temp_size              # conv2 ("same")
        temp_size = temp_size // 2         # pool2

        self.flat_size = temp_size * 32

        # Fully connected head
        self.fc1 = nn.Linear(self.flat_size, 64)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc_out = nn.Linear(64, 1)     # logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, L)
        returns logits: (B,)
        """
        x = self.relu1(self.motif_conv(x))
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc_out(x).squeeze(1)
        return logits

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """
        Convenience: return probabilities as numpy array.
        x: (B, C, L) tensor on correct device
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
        return probs.cpu().numpy()


# =====================================================================
# IMPORTANCE SCORES (grad * input)
# =====================================================================

def compute_importance_scores(
    model: nn.Module,
    X_seq: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Compute importance scores using gradient * input.

    Parameters
    ----------
    model : trained CNN
    X_seq : np.ndarray, shape (N, 4, L)
    device : torch.device

    Returns
    -------
    importance_scores : np.ndarray, shape (N, 4, L)
    """
    print("\n===== Computing Importance Scores (grad * input) =====")
    model.eval()

    N = X_seq.shape[0]
    importance_chunks = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)

        batch = torch.tensor(
            X_seq[start:end],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )

        logits = model(batch)       # (B,)
        loss = logits.sum()         # scalar for backprop
        model.zero_grad()
        loss.backward()

        grads = batch.grad.detach().cpu().numpy()   # (B, 4, L)
        importance_chunks.append(grads * X_seq[start:end])

    importance_scores = np.concatenate(importance_chunks, axis=0)
    print(f"Importance scores shape: {importance_scores.shape}")
    return importance_scores


# =====================================================================
# FULL TF-MoDISco WRAPPER
# =====================================================================

def run_full_tfmodisco(
    one_hot_seqs: np.ndarray,
    importance_scores: np.ndarray,
    output_prefix: str,
    sliding_window_size: int = 21,
    flank_size: int = 10,
    min_metacluster_size: int = 100,
) -> dict:
    """
    Run full TFMoDISco on provided sequences + importance scores.

    Parameters
    ----------
    one_hot_seqs : (N, 4, L) one-hot DNA
    importance_scores : (N, 4, L) hypothetical contribs (grad*input)
    output_prefix : prefix for output files

    Returns
    -------
    results : dict with 'pos_patterns' and 'neg_patterns'
    """
    print("\n===== Running FULL TF-MoDISco =====")
    print(f"one_hot shape:            {one_hot_seqs.shape}")
    print(f"hypothetical_contribs:    {importance_scores.shape}")

    pos_patterns, neg_patterns = TFMoDISco(
    one_hot=one_hot_seqs.astype(np.float32),
    hypothetical_contribs=importance_scores.astype(np.float32),
    sliding_window_size=20,
    flank_size=flank_size,
    min_metacluster_size=50,  # Lower from 100
    target_seqlet_fdr=0.5,     # Increase from 0.2
    max_seqlets_per_metacluster=20000,
    verbose=True,
)

    out_pkl = f"{output_prefix}_tfmodisco_patterns.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(
            {
                "pos_patterns": pos_patterns,
                "neg_patterns": neg_patterns,
            },
            f,
        )

    print(f"Saved TF-MoDISco patterns to {out_pkl}")
    if pos_patterns is not None:
        print(f"  # positive patterns: {len(pos_patterns)}")
    if neg_patterns is not None:
        print(f"  # negative patterns: {len(neg_patterns)}")

    return {
        "pos_patterns": pos_patterns,
        "neg_patterns": neg_patterns,
    }


# =====================================================================
# TRAINING + CV + FINAL MODEL + MODISCO
# =====================================================================

def train_cnn_with_validation(
    data_path: str,
    output_prefix: str,
    use_modisco: bool = True,
    max_pos_for_modisco: int = 5000,
) -> dict:
    """
    Full pipeline:
      - load data
      - 5-fold stratified CV with early stopping on AUROC
      - train final model on all data
      - run full TF-MoDISco on positives (optionally subsampled)
    """

    # Reproducibility-ish
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = np.load(data_path)
    X_seq = data["X_seq"].astype(np.float32)  # (N, 4, L)
    y = data["y"].astype(np.int64)

    N, C, L = X_seq.shape
    print(f"Sequence shape for CNN: {X_seq.shape}")
    print(f"Number of samples:       {N}")
    print(f"Sequence length:         {L} bp")
    print(f"Positive samples:        {y.sum()}")
    print(f"Negative samples:        {(y == 0).sum()}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accs, aurocs, auprcs = [], [], []
    train_aurocs, test_aurocs = [], []
    all_training_histories = []
    fold = 1

    for train_idx, test_idx in skf.split(X_seq, y):
        print("\n" + "=" * 60)
        print(f"Fold {fold}")
        print("=" * 60)

        X_train_full, X_test = X_seq[train_idx], X_seq[test_idx]
        y_train_full, y_test = y[train_idx], y[test_idx]

        # train/val split within fold
        train_sub_idx, val_idx = train_test_split(
            np.arange(len(X_train_full)),
            test_size=0.1,
            stratify=y_train_full,
            random_state=42,
        )

        X_train = X_train_full[train_sub_idx]
        y_train = y_train_full[train_sub_idx]
        X_val = X_train_full[val_idx]
        y_val = y_train_full[val_idx]

        print(
            f"Train size: {len(X_train)}, "
            f"Val size: {len(X_val)}, "
            f"Test size: {len(X_test)}"
        )

        # Tensors
        X_train_t = torch.from_numpy(X_train)
        y_train_t = torch.from_numpy(y_train.astype(np.float32))
        X_val_t = torch.from_numpy(X_val).to(device)
        y_val_t = torch.from_numpy(y_val.astype(np.float32)).to(device)
        X_test_t = torch.from_numpy(X_test).to(device)
        y_test_t = torch.from_numpy(y_test.astype(np.float32)).to(device)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        model = CNN(L, C).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        best_val_auroc = -np.inf
        patience = 5
        no_improve = 0
        best_model_state = None

        history = {
            "train_loss": [],
            "val_loss": [],
            "val_auroc": [],
            "val_auprc": [],
        }

        print("\nStarting training...")
        for epoch in range(50):
            model.train()
            epoch_train_loss = 0.0

            for batch_X_cpu, batch_y_cpu in train_loader:
                batch_X = batch_X_cpu.to(device)
                batch_y = batch_y_cpu.to(device)

                optimizer.zero_grad()
                logits = model(batch_X)             # (B,)
                loss = criterion(logits, batch_y)   # BCEWithLogits
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()

            avg_train_loss = epoch_train_loss / len(train_loader)

            # Validation
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_t)
                val_loss = criterion(val_logits, y_val_t).item()
                val_probs = sigmoid_np(val_logits.cpu().numpy())

                val_auroc = roc_auc_score(y_val, val_probs)
                val_auprc = average_precision_score(y_val, val_probs)

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)
            history["val_auroc"].append(val_auroc)
            history["val_auprc"].append(val_auprc)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"Epoch {epoch + 1:2d}: "
                    f"Train Loss={avg_train_loss:.4f}, "
                    f"Val Loss={val_loss:.4f}, "
                    f"Val AUROC={val_auroc:.4f}"
                )

            # Early stopping on AUROC
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_model_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                print(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(no improvement for {patience} epochs)"
                )
                break

        # Load best model for this fold
        model.load_state_dict(best_model_state)
        model.to(device)
        all_training_histories.append(history)

        # Plot training curves
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(history["train_loss"], label="Train Loss", linewidth=2)
        axes[0].plot(history["val_loss"], label="Val Loss", linewidth=2)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training and Validation Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(history["val_auroc"], label="Val AUROC", linewidth=2)
        axes[1].axhline(0.5, linestyle="--", alpha=0.5, label="Random")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("AUROC")
        axes[1].set_title("Validation AUROC")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(history["val_auprc"], label="Val AUPRC", linewidth=2)
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("AUPRC")
        axes[2].set_title("Validation AUPRC")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        curves_path = f"{output_prefix}_training_curves_fold{fold}.png"
        plt.savefig(curves_path, dpi=200)
        plt.close()
        print(f"Saved training curves: {curves_path}")

        # Evaluation on test set + overfitting check
        model.eval()
        with torch.no_grad():
            test_logits = model(X_test_t)
            train_logits = model(torch.from_numpy(X_train).to(device))

        test_probs = sigmoid_np(test_logits.cpu().numpy())
        train_probs = sigmoid_np(train_logits.cpu().numpy())

        test_pred = (test_probs > 0.5).astype(int)

        acc = accuracy_score(y_test, test_pred)
        auroc_test = roc_auc_score(y_test, test_probs)
        auprc = average_precision_score(y_test, test_probs)
        auroc_train = roc_auc_score(y_train, train_probs)

        accs.append(acc)
        aurocs.append(auroc_test)
        auprcs.append(auprc)
        train_aurocs.append(auroc_train)
        test_aurocs.append(auroc_test)

        print("\n" + "-" * 60)
        print(f"Fold {fold} Results:")
        print(f"  Train AUROC: {auroc_train:.4f}")
        print(f"  Test AUROC:  {auroc_test:.4f}")
        print(f"  Overfitting gap: {auroc_train - auroc_test:.4f}")
        print(f"  Test Accuracy: {acc:.4f}")
        print(f"  Test AUPRC:   {auprc:.4f}")
        print("-" * 60)

        if auroc_train - auroc_test > 0.15:
            print("⚠️  WARNING: Significant overfitting detected")

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, test_probs)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"Test AUC={auroc_test:.3f}", linewidth=2)
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve – Fold {fold}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        roc_path = f"{output_prefix}_roc_fold{fold}.png"
        plt.savefig(roc_path, dpi=200)
        plt.close()

        # PR curve
        prec, recall, _ = precision_recall_curve(y_test, test_probs)
        plt.figure(figsize=(6, 6))
        plt.plot(recall, prec, label=f"AUPRC={auprc:.3f}", linewidth=2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR Curve – Fold {fold}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        pr_path = f"{output_prefix}_pr_fold{fold}.png"
        plt.savefig(pr_path, dpi=200)
        plt.close()

        fold += 1

    # Summary
    train_aurocs = np.array(train_aurocs)
    test_aurocs = np.array(test_aurocs)
    overfit_gap = train_aurocs - test_aurocs

    print("\n" + "=" * 60)
    print("5-FOLD CROSS-VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Accuracy:         {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"AUROC (test):     {np.mean(test_aurocs):.4f} ± {np.std(test_aurocs):.4f}")
    print(f"AUROC (train):    {np.mean(train_aurocs):.4f} ± {np.std(train_aurocs):.4f}")
    print(f"Avg overfit gap:  {np.mean(overfit_gap):.4f}")
    print(f"AUPRC (test):     {np.mean(auprcs):.4f} ± {np.std(auprcs):.4f}")
    print("=" * 60 + "\n")

    # Train final model on ALL data for MoDISco
    print("===== Training Final Model on All Data =====")

    X_all_t = torch.from_numpy(X_seq).to(device)
    y_all_t = torch.from_numpy(y.astype(np.float32)).to(device)

    final_dataset = TensorDataset(X_all_t, y_all_t)
    final_loader = DataLoader(final_dataset, batch_size=64, shuffle=True)

    final_model = CNN(L, C).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(final_model.parameters(), lr=1e-3)

    avg_epochs = int(np.mean([len(h["train_loss"]) for h in all_training_histories]))
    avg_epochs = max(avg_epochs, 1)
    print(f"Training final model for {avg_epochs} epochs")

    final_model.train()
    for epoch in range(avg_epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in final_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = final_model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 5 == 0 or epoch == avg_epochs:
            print(
                f"Epoch {epoch + 1}/{avg_epochs}, "
                f"Loss: {epoch_loss / len(final_loader):.4f}"
            )

    model_path = f"{output_prefix}_final_model.pt"
    torch.save(final_model.state_dict(), model_path)
    print(f"\n✓ Saved final CNN model: {model_path}")

    # MoDISco on positives
    if use_modisco:
        pos_idx = np.where(y == 1)[0]
        X_pos = X_seq[pos_idx]

        if len(X_pos) > max_pos_for_modisco:
            print(f"Subsampling {len(X_pos)} positives to {max_pos_for_modisco} for MoDISco")
            subset = np.random.choice(len(X_pos), max_pos_for_modisco, replace=False)
            X_pos = X_pos[subset]

        importance_scores = compute_importance_scores(final_model, X_pos, device)
        try:
            _ = run_full_tfmodisco(X_pos, importance_scores, output_prefix)
        except Exception as e:
            print(f"⚠️ TF-MoDISco failed with error:\n{e}")
            print("Continuing without MoDISco results...")

    print("\n✓ CNN training complete!")

    return {
        "mean_auroc": float(np.mean(test_aurocs)),
        "std_auroc": float(np.std(test_aurocs)),
        "mean_auprc": float(np.mean(auprcs)),
        "training_histories": all_training_histories,
    }


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-path",
        type=str,
        default="datasets_chr1_1000bp/ctcf_chr1_dataset_struct.npz",
        help="Path to dataset file (.npz with X_seq and y)",
    )

    parser.add_argument(
        "--output-prefix",
        type=str,
        default="cnn",
        help="Prefix for output files",
    )

    parser.add_argument(
        "--use-modisco",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Whether to run full TF-MoDISco motif discovery (True/False)",
    )

    parser.add_argument(
        "--max-pos-for-modisco",
        type=int,
        default=5000,
        help="Max # positive sequences to use for TF-MoDISco",
    )

    args = parser.parse_args()

    # If user left default 'cnn', infer TF name from data path
    if args.output_prefix == "cnn":
        # e.g. "ctcf_chr1_dataset_struct.npz" -> "ctcf"
        tf_name = args.data_path.split("/")[-1].split("_")[0]
        args.output_prefix = f"cnn_{tf_name}"

    results = train_cnn_with_validation(
        data_path=args.data_path,
        output_prefix=args.output_prefix,
        use_modisco=args.use_modisco,
        max_pos_for_modisco=args.max_pos_for_modisco,
    )

    print("\nFinal Results:")
    print(f"  Mean AUROC: {results['mean_auroc']:.4f} ± {results['std_auroc']:.4f}")
    print(f"  Mean AUPRC: {results['mean_auprc']:.4f}")
