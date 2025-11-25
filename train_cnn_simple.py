#!/usr/bin/env python

"""
CNN + 60/20/20 + Full TF-MoDISco
FULLY DEBUG-INSTRUMENTED VERSION
"""

# ============================================================
#  IMPORT ORDER — MUST IMPORT hdf5plugin BEFORE h5py
# ============================================================
import hdf5plugin
import h5py

from modiscolite.tfmodisco import TFMoDISco

# Standard libs
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
#  CNN MODEL (simple, debug-safe)
# ============================================================
class CNN(nn.Module):
    def __init__(self, seq_len, num_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(num_channels, 32, kernel_size=8)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)

        out_len = (seq_len - 8 + 1) // 2
        self.fc = nn.Linear(out_len * 32, 1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x).squeeze(1)

    def predict_proba(self, X, device):
        self.eval()
        with torch.no_grad():
            logits = self.forward(X.to(device))
            return torch.sigmoid(logits).cpu().numpy()


# ============================================================
#  IMPORTANCE SCORES (Grad × Input)
# ============================================================
def compute_importance(model, X, device, batch_size=64):
    print("\n=== Computing Grad×Input ===")
    model.eval()

    N = X.shape[0]
    out = []

    for i in range(0, N, batch_size):
        xb = torch.tensor(
            X[i:i+batch_size],
            dtype=torch.float32,
            device=device,
            requires_grad=True
        )

        logits = model(xb)
        loss = logits.sum()
        model.zero_grad()
        loss.backward()

        grads = xb.grad.detach().cpu().numpy()
        out.append(grads * X[i:i+batch_size])

    scores = np.concatenate(out, axis=0)
    print("Importance shape:", scores.shape)

    return scores


# ============================================================
#  FULL TF-MoDISco (with full debug tracing)
# ============================================================
def run_tfmodisco(one_hot, contribs, prefix):
    print("\n===== Running TF-MoDISco =====")
    print("one_hot (CNN convention):", one_hot.shape)      # (N, 4, L)
    print("contribs (CNN convention):", contribs.shape)     # (N, 4, L)

    # ----------------------------------------------------
    # Reorder axes for TF-MoDISco: (N, L, 4)
    # ----------------------------------------------------
    one_hot_tf = np.transpose(one_hot, (0, 2, 1))      # (N, L, 4)
    contribs_tf = np.transpose(contribs, (0, 2, 1))    # (N, L, 4)

    print("one_hot (TF-MoDISco):", one_hot_tf.shape)
    print("contribs (TF-MoDISco):", contribs_tf.shape)

    # DEBUG: attribution stats BEFORE TF-MoDISco
    flat = np.abs(contribs_tf).ravel()
    print("\n[DEBUG] Attribution score stats BEFORE TF-MoDISco:")
    print("  mean abs:", float(flat.mean()))
    print("  median abs:", float(np.median(flat)))
    print("  99th percentile:", float(np.percentile(flat, 99)))
    print("  max abs:", float(flat.max()))

    # Seqlet extraction sanity check
    sliding_window_size = 21
    flank_size = 10
    seqlet_span = sliding_window_size + 2 * flank_size
    seq_len = one_hot_tf.shape[1]

    print("\n[DEBUG] Seqlet Extraction Check:")
    print("  sliding_window_size:", sliding_window_size)
    print("  flank_size:", flank_size)
    print("  total span:", seqlet_span)
    print("  sequence length:", seq_len)

    if seq_len < seqlet_span:
        raise ValueError(
            f"Seqlet span {seqlet_span} > sequence length {seq_len}; "
            "reduce sliding_window_size or flank_size."
        )

    # ----------------------------------------------------
    # Call TFMoDISco with correctly-shaped arrays
    # ----------------------------------------------------
    try:
        print("\n[DEBUG] Calling TFMoDISco...")

        pos_patterns, neg_patterns = TFMoDISco(
            one_hot=one_hot_tf.astype(np.float32),
            hypothetical_contribs=contribs_tf.astype(np.float32),
            sliding_window_size=sliding_window_size,
            flank_size=flank_size,
            min_metacluster_size=50,
            target_seqlet_fdr=0.5,
            max_seqlets_per_metacluster=20000,
            verbose=True,
        )

        print("\n[DEBUG] TFMoDISco returned successfully.")
        print("[DEBUG] pos_patterns:", None if pos_patterns is None else len(pos_patterns))
        print("[DEBUG] neg_patterns:", None if neg_patterns is None else len(neg_patterns))

    except Exception as e:
        import traceback
        print("\n❌ TF-MoDISco CRASHED — FULL TRACEBACK:")
        traceback.print_exc()
        print("\n=== DEBUG SNAPSHOT ===")
        print("one_hot_tf shape:", one_hot_tf.shape)
        print("contribs_tf shape:", contribs_tf.shape)
        print("dtype:", contribs_tf.dtype)
        print("min contrib:", contribs_tf.min())
        print("max contrib:", contribs_tf.max())
        print("====================================")
        raise

    # Save patterns
    out_file = prefix + "_tfmodisco.pkl"
    with open(out_file, "wb") as f:
        pickle.dump(
            {
                "pos_patterns": pos_patterns,
                "neg_patterns": neg_patterns,
            },
            f,
        )

    print("✓ Saved MoDISco:", out_file)
    return pos_patterns, neg_patterns



# ============================================================
#  MAIN PIPELINE
# ============================================================
def main(data_path, output_prefix):

    # -------------------------
    # Load data
    # -------------------------
    data = np.load(data_path)
    X = data["X_seq"].astype(np.float32)
    y = data["y"].astype(np.float32)

    N, C, L = X.shape
    print("Loaded:", data_path)
    print("X:", X.shape, " y:", y.shape)

    # ---------------------------------------------------------------------
    # DEBUG BLOCK 4 — class distribution
    # ---------------------------------------------------------------------
    print("\n[DEBUG] Label Distribution:")
    print("  positives:", int((y == 1).sum()))
    print("  negatives:", int((y == 0).sum()))

    # -------------------------
    # Split data
    # -------------------------
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp
    )

    print("\n=== Data Split ===")
    print("Train:", X_train.shape)
    print("Val:  ", X_val.shape)
    print("Test: ", X_test.shape)

    # -------------------------
    # DataLoader
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nUsing device:", device)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=64,
        shuffle=True,
    )

    # -------------------------
    # Train CNN
    # -------------------------
    model = CNN(L, C).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    print("\n=== Training CNN ===")

    for epoch in range(10):
        model.train()
        epoch_loss = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss={epoch_loss/len(train_loader):.4f}")

    # -------------------------
    # Evaluate
    # -------------------------
    model.eval()
    test_probs = model.predict_proba(
        torch.tensor(X_test, dtype=torch.float32), device
    )

    print("\n=== Test Performance ===")
    print("AUROC:", roc_auc_score(y_test, test_probs))
    print("AUPRC:", average_precision_score(y_test, test_probs))

    # -------------------------
    # TF-MoDISco on POSITIVES
    # -------------------------
    print("\n=== Preparing positives for TF-MoDISco ===")
    pos_idx = np.where(y == 1)[0]
    X_pos = X[pos_idx]
    print("Positive samples:", len(X_pos))

    # Importance scores
    scores = compute_importance(model, X_pos, device)

    # ---------------------------------------------------------------------
    # DEBUG BLOCK 5 — Gradient statistics
    # ---------------------------------------------------------------------
    print("\n[DEBUG] Gradient×Input statistics:")
    flat_imp = np.abs(scores).ravel()
    print("  mean abs:", float(flat_imp.mean()))
    print("  99th pct:", float(np.percentile(flat_imp, 99)))
    print("  max abs:", float(flat_imp.max()))

    # MoDISco
    run_tfmodisco(X_pos, scores, output_prefix)


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--data", type=str,
                  default="datasets_chr1_1000bp/ctcf_chr1_dataset_struct.npz")
    p.add_argument("--prefix", type=str, default="cnn_debug")

    args = p.parse_args()
    main(args.data, args.prefix)
