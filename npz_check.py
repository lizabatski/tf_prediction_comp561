import sys
import numpy as np
import matplotlib.pyplot as plt

def onehot_to_seq(onehot):
    mapping = {0: "A", 1: "C", 2: "G", 3: "T"}
    return "".join(mapping[i] for i in onehot.argmax(axis=0))

def main(path):
    print(f"\n=== Inspecting: {path} ===\n")

    data = np.load(path)
    print("Keys in file:", data.files)

    # Basic shapes
    X_seq = data["X_seq"]
    X_struct = data["X_struct"]
    y = data["y"]

    print("\n--- Shape Summary ---")
    print("X_seq shape:   ", X_seq.shape)     # (N, 4, L)
    print("X_struct shape:", X_struct.shape)  # (N, num_struct_features)
    print("y shape:        ", y.shape)

    # Check class balance
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    print("\n--- Class Balance ---")
    print(f"Positive samples: {n_pos}")
    print(f"Negative samples: {n_neg}")

    # Inspect first sample
    print("\n--- First Sample ---")
    print("Label y[0]:", y[0])
    print("Structural features:", X_struct[0])
    print("Sequence (one-hot shape):", X_seq[0].shape)

    seq = onehot_to_seq(X_seq[0])
    print("Reconstructed sequence:")
    print(seq)

    # Check for NaNs
    print("\n--- Data Quality ---")
    print("NaNs in X_seq:", np.isnan(X_seq).any())
    print("NaNs in X_struct:", np.isnan(X_struct).any())
    print("NaNs in y:", np.isnan(y).any())

    # Structural feature statistics
    print("\n--- Structural Feature Statistics ---")
    print("Means :", X_struct.mean(axis=0))
    print("Stds  :", X_struct.std(axis=0))

    # Quick visualization
    try:
        plt.figure(figsize=(8, 4))
        plt.hist(X_struct[:,0], bins=50)
        plt.title("Distribution of first structural feature")
        plt.xlabel("Value")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Plotting skipped:", e)

    print("\nDone.\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_npz.py <path_to_npz>")
        sys.exit(1)

    main(sys.argv[1])
