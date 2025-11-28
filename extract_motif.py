#!/usr/bin/env python3
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import logomaker


# ------------------------------
#   CNN model definition
# ------------------------------
class SeqCNN(nn.Module):
    def __init__(self, motif_k=10, num_filters=16):
        super().__init__()
        self.conv = nn.Conv1d(4, num_filters, kernel_size=motif_k)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(num_filters, 1)

    def forward(self, x_seq):
        x = torch.relu(self.conv(x_seq))
        x = self.pool(x).squeeze(-1)
        return torch.sigmoid(self.fc(x)).view(-1)


# ------------------------------
#  Convert filter → PWM
# ------------------------------
def normalize_filter_to_pwm(w):
    """
    w: numpy array of shape (4, motif_k)
       values are convolution weights.

    We convert each column to probability distribution using softmax.
    """
    pwm = np.exp(w)  # softmax numerator
    pwm = pwm / pwm.sum(axis=0, keepdims=True)
    return pwm


# ------------------------------
#   Plot PWM using logomaker
# ------------------------------
def plot_pwm(pwm, out_png):
    """
    pwm: numpy array (4, L), rows = A,C,G,T
    """

    bases = ["A", "C", "G", "T"]
    L = pwm.shape[1]

    # Convert to DataFrame in correct orientation for logomaker:
    # columns = bases, rows = positions
    df = pd.DataFrame(pwm.T, columns=bases)

    # Compute Information Content manually:
    # IC = log2(4) - sum(p * log2(p))
    eps = 1e-9
    entropy = -(df * np.log2(df + eps)).sum(axis=1)
    ic = 2 - entropy            # 2 bits max for DNA
    df_ic = df.multiply(ic, axis=0)

    # Plot
    plt.figure(figsize=(max(6, L / 2), 2))
    logomaker.Logo(df_ic)
    plt.title("Filter PWM")
    plt.xlabel("Position")
    plt.ylabel("Information (bits)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ------------------------------
#   MAIN EXTRACTION LOGIC
# ------------------------------
def main(args):
    os.makedirs(args.outdir, exist_ok=True)

    # Load model skeleton
    model = SeqCNN(motif_k=args.kernel, num_filters=args.filters)
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    print(f"Loaded model: {args.model}")
    print(f"Extracting {args.filters} filters (kernel={args.kernel})")

    # Extract convolution weights
    conv_w = model.conv.weight.detach().cpu().numpy()   # shape: (F, 4, K)

    for i in range(args.filters):
        print(f"Processing filter {i}")

        w = conv_w[i]            # (4, kernel_size)
        pwm = normalize_filter_to_pwm(w)

        out_npy = os.path.join(args.outdir, f"filter_{i}_pwm.npy")
        out_png = os.path.join(args.outdir, f"filter_{i}_logo.png")

        np.save(out_npy, pwm)
        plot_pwm(pwm, out_png)

        print(f"  Saved PWM → {out_npy}")
        print(f"  Saved logo → {out_png}")

    print("\nDone! Motifs saved in:", args.outdir)


# ------------------------------
#   CLI
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained SeqCNN .pt file")

    parser.add_argument("--kernel", type=int, default=10,
                        help="Motif length used during training")

    parser.add_argument("--filters", type=int, default=16,
                        help="Number of convolution filters")

    parser.add_argument("--outdir", type=str, required=True,
                        help="Output directory for motifs")

    args = parser.parse_args()
    main(args)
