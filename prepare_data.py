import numpy as np
import pandas as pd
from pyfaidx import Fasta
import random
import os
import time

# ============================================================
# CONFIGURATION
# ============================================================

FACTORBOOK = "data/factorbook/factorbookMotifPos.txt"
REGULATORY_BED = "data/regulatory/GM12878_regulatory.bed"
CHR1_FASTA = "data/reference/chr1.fa"
CHR_NAME = "chr1"

# structural tracks for chr1 from RohsDB
MGW_WIG  = "data/structural/hg19.chr1.MGW.2nd.wig"
PROT_WIG = "data/structural/hg19.chr1.ProT.2nd.wig"
ROLL_WIG = "data/structural/hg19.chr1.Roll.2nd.wig"

OUTPUT_DIR = "datasets_chr1_1000bp"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_TFS = ["CTCF", "EGR1", "GATA1"]
WINDOW_SIZE = 1000
HALF = WINDOW_SIZE // 2


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def reverse_complement(seq):
    comp = str.maketrans("ACGTN", "TGCAN")
    return seq.translate(comp)[::-1]


def load_fasta():
    return Fasta(CHR1_FASTA)[CHR_NAME]


def load_regulatory_bed():
    df = pd.read_csv(
        REGULATORY_BED,
        sep="\t",
        header=None
    )

    # BED format: chrom, start, end, name
    df = df.iloc[:, :3]   # keep first 3 columns only
    df.columns = ["chrom", "start", "end"]

    # Normalize chromosome names: remove spaces, ensure "chr" prefix
    df["chrom"] = df["chrom"].astype(str).str.strip()

    # Match exact chromosome
    df = df[df["chrom"] == CHR_NAME]

    return df.reset_index(drop=True)


def one_hot_encode(seq):
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    arr = np.zeros((4, len(seq)), dtype=np.float32)
    for i, b in enumerate(seq):
        if b in mapping:
            arr[mapping[b], i] = 1.0
    return arr


def load_wig_track_to_array(wig_path, chrom_name, chrom_len):
    print(f"  Loading {wig_path}...")
    arr = np.full(chrom_len, np.nan, dtype=np.float32)

    current_chrom = None
    mode = None  # "fixed" or "variable"
    pos = None
    step = 1
    values_loaded = 0

    with open(wig_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("fixedStep"):
                mode = "fixed"
                current_chrom = None
                start = None
                step = 1
                # parse like: fixedStep chrom=chr1 start=1 step=1
                for token in line.split()[1:]:
                    if "=" not in token:
                        continue
                    key, val = token.split("=")
                    if key == "chrom":
                        current_chrom = val
                    elif key == "start":
                        start = int(val)
                    elif key == "step":
                        step = int(val)
                
                if start is not None:
                    pos = (start - 1)  # convert to 0-based index

            elif line.startswith("variableStep"):
                mode = "variable"
                current_chrom = None
                # parse like: variableStep chrom=chr1
                for token in line.split()[1:]:
                    if "=" not in token:
                        continue
                    key, val = token.split("=")
                    if key == "chrom":
                        current_chrom = val
                pos = None

            else:
                # data line
                if current_chrom != chrom_name:
                    # still need to advance pos for fixedStep, but we ignore values
                    if mode == "fixed" and pos is not None:
                        pos += step
                    continue

                if mode == "fixed":
                    try:
                        value = float(line)
                        if 0 <= pos < chrom_len:
                            arr[pos] = value
                            values_loaded += 1
                        pos += step
                    except ValueError:
                        pass

                elif mode == "variable":
                    # "position value"
                    parts = line.split()
                    if len(parts) != 2:
                        continue
                    try:
                        position = int(parts[0]) - 1
                        value = float(parts[1])
                        if 0 <= position < chrom_len:
                            arr[position] = value
                            values_loaded += 1
                    except ValueError:
                        pass

    print(f"    Loaded {values_loaded} values, non-NaN: {(~np.isnan(arr)).sum()}")
    return arr


# ============================================================
# LOAD INPUT DATA
# ============================================================

print("Loading FASTA...")
genome = load_fasta()
chrom_len = len(genome)
print(f"Chromosome length: {chrom_len:,} bp")

print("Loading Factorbook motifs...")
cols = ["id", "chrom", "start", "end", "tf", "score", "strand"]
fb = pd.read_csv(FACTORBOOK, sep="\t", header=None, names=cols)
fb = fb[fb["chrom"] == CHR_NAME].reset_index(drop=True)
print(f"Found {len(fb)} TF binding sites on {CHR_NAME}")

print("Loading regulatory regions...")
reg = load_regulatory_bed()
print(f"Found {len(reg)} regulatory regions on {CHR_NAME}")

print(f"\nLoading structural tracks for {CHR_NAME}...")
start_time = time.time()
mgw_arr = load_wig_track_to_array(MGW_WIG, CHR_NAME, chrom_len)
print(f"  MGW loaded in {time.time() - start_time:.2f} seconds")

start_time = time.time()
prot_arr = load_wig_track_to_array(PROT_WIG, CHR_NAME, chrom_len)
print(f"  ProT loaded in {time.time() - start_time:.2f} seconds")

start_time = time.time()
roll_arr = load_wig_track_to_array(ROLL_WIG, CHR_NAME, chrom_len)
print(f"  Roll loaded in {time.time() - start_time:.2f} seconds")


# ============================================================
# NEGATIVE SAMPLING FUNCTIONS
# ============================================================

def overlaps(pos_start, pos_end, tf_intervals):
    """Check if [pos_start, pos_end] overlaps ANY TF interval."""
    if tf_intervals.size == 0:
        return False
    return np.any((tf_intervals[:, 0] < pos_end) & (tf_intervals[:, 1] > pos_start))


def sample_negatives(n_pos, tf_intervals):
    """
    Negative sampling inside regulatory regions on chr1.
    Reject if overlapping any positive TF site.
    """
    negatives = []
    tf_intervals = np.array(tf_intervals)

    while len(negatives) < n_pos:
        # 1. Pick a random regulatory region
        row = reg.sample(1).iloc[0]
        reg_start, reg_end = int(row.start), int(row.end)

        if reg_end - reg_start < WINDOW_SIZE:
            continue

        # 2. Pick random valid start inside the region
        start = random.randint(reg_start, reg_end - WINDOW_SIZE)
        end = start + WINDOW_SIZE

        # 3. Reject if overlap with TF positive sites
        if overlaps(start, end, tf_intervals):
            continue

        # strand for negatives not important; use "+"
        negatives.append((start, end, "+", 0))

    return negatives


# ============================================================
# SEQUENCE + STRUCTURE EXTRACTION
# ============================================================

def extract_seq(start, end, strand):
    seq = genome[start:end].seq.upper()
    if "N" in seq:
        return None
    if strand == "-":
        seq = reverse_complement(seq)
    return seq


def extract_struct_features(start, end):
    """
    For window [start, end) on chr1, compute mean + std
    for each structural track (ignoring NaNs).

    Returns a vector of length 6:
        [MGW_mean, MGW_std,
         ProT_mean, ProT_std,
         Roll_mean, Roll_std]
    """
    s = slice(start, end)

    window_mgw  = mgw_arr[s]
    window_prot = prot_arr[s]
    window_roll = roll_arr[s]

    tracks = [window_mgw, window_prot, window_roll]
    feats = []

    for arr in tracks:
        finite = np.isfinite(arr)
        if not np.any(finite):
            feats.extend([np.nan, np.nan])
        else:
            feats.append(arr[finite].mean())
            feats.append(arr[finite].std())

    return np.array(feats, dtype=np.float32)


# ============================================================
# MAIN LOOP – ONE DATASET PER TF
# ============================================================

for TF in TARGET_TFS:

    print("\n" + "="*60)
    print(f"Building dataset for {TF} on {CHR_NAME}")
    print("="*60)

    # --------------------------
    # POSITIVE EXAMPLES
    # --------------------------
    tf_sites = fb[fb["tf"] == TF]

    positives = []
    tf_intervals = []

    for _, row in tf_sites.iterrows():
        midpoint = (int(row.start) + int(row.end)) // 2
        start = midpoint - HALF
        end = midpoint + HALF

        if start < 0 or end > chrom_len:
            continue

        seq = extract_seq(start, end, row.strand)
        if seq is None:  # skip sequences with "N"
            continue

        positives.append((start, end, row.strand, 1))
        tf_intervals.append((start, end))

    print(f"Positives collected: {len(positives)}")

    tf_intervals = np.array(tf_intervals, dtype=np.int64)

    # --------------------------
    # NEGATIVE EXAMPLES
    # --------------------------
    negatives = sample_negatives(len(positives), tf_intervals)
    print(f"Negatives sampled: {len(negatives)}")

    # --------------------------
    # BUILD FINAL ARRAYS
    # --------------------------
    all_examples = positives + negatives

    X_seq_list = []
    X_struct_list = []
    y_list = []

    for (start, end, strand, label) in all_examples:
        seq = extract_seq(start, end, strand)
        if seq is None:
            continue

        X_seq_list.append(one_hot_encode(seq))
        X_struct_list.append(extract_struct_features(start, end))
        y_list.append(label)

    X_seq = np.stack(X_seq_list, axis=0)        # (N, 4, 20)
    X_struct = np.stack(X_struct_list, axis=0)  # (N, 6) - NOW 6 features, not 8!
    y = np.array(y_list, dtype=np.int64)        # (N,)

    print(f"Final dataset size for {TF}: {X_seq.shape[0]} windows")
    print(f"Sequence feature shape: {X_seq.shape}")
    print(f"Structural feature shape: {X_struct.shape}")

    # Check for NaN in structural features
    nan_count = np.isnan(X_struct).sum()
    print(f"NaN values in structural features: {nan_count} ({100*nan_count/X_struct.size:.1f}%)")

    # --------------------------
    # SAVE .npz FILE
    # --------------------------
    outpath = os.path.join(OUTPUT_DIR, f"{TF.lower()}_chr1_dataset_struct.npz")

    np.savez(
        outpath,
        X_seq=X_seq,
        X_struct=X_struct,
        y=y,
    )

    print(f"Saved: {outpath}")

print("\n" + "="*60)
print("All three datasets complete!")
print("="*60)