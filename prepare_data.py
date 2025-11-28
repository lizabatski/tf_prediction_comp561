import numpy as np
import pandas as pd
from pyfaidx import Fasta
import pyBigWig
import random
import os
from multiprocessing import Pool, cpu_count

# ============================================================
# CONFIGURATION
# ============================================================

FACTORBOOK = "data/factorbook/factorbookMotifPos.txt"
REGULATORY_BED = "data/regulatory/GM12878_regulatory.bed"
CHR1_FASTA = "data/reference/chr1.fa"
PWM_FILE = "data/factorbook/factorbookMotifPwm.txt"
CHR_NAME = "chr1"

STRUCT_DIR = "/scratch/ekourb/wigconvert"
BUCKLE_BW = f"{STRUCT_DIR}/hg19.Buckle.chr1.bw"
MGW_BW = f"{STRUCT_DIR}/hg19.MGW.2nd.chr1.bw"
OPENING_BW = f"{STRUCT_DIR}/hg19.Opening.chr1.bw"
PROT_BW = f"{STRUCT_DIR}/hg19.ProT.2nd.chr1.bw"
ROLL_BW = f"{STRUCT_DIR}/hg19.Roll.2nd.chr1.bw"
BW_PATHS = [BUCKLE_BW, MGW_BW, OPENING_BW, PROT_BW, ROLL_BW]

OUTPUT_DIR = "datasets_chr1_optionC_fast"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_TFS = ["CTCF", "EGR1", "GATA1"]
PWM_SCORE_THRESHOLD_PERCENTILE = 80
WINDOW_SIZE = 30
HALF = WINDOW_SIZE // 2

random.seed(42)
np.random.seed(42)

# ============================================================
# WORKER-LOCAL BIGWIG HANDLES
# ============================================================

_worker_bws = None

def init_worker():
    global _worker_bws
    _worker_bws = [pyBigWig.open(p) for p in BW_PATHS]

def extract_struct_features(start, end):
    feats = []
    for bw in _worker_bws:
        vals = np.array(bw.values(CHR_NAME, start, end))
        finite = np.isfinite(vals)
        if not np.any(finite):
            feats.extend([np.nan, np.nan])
        else:
            feats.append(float(vals[finite].mean()))
            feats.append(float(vals[finite].std()))
    return np.array(feats, dtype=np.float32)

# ============================================================
# HELPERS
# ============================================================

def reverse_complement(seq):
    return seq.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]

def one_hot_encode(seq):
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    arr = np.zeros((4, len(seq)), dtype=np.float32)
    for i, b in enumerate(seq):
        if b in mapping:
            arr[mapping[b], i] = 1.0
    return arr

def load_pwms():
    pwms = {}
    with open(PWM_FILE) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 6:
                continue
            tf, L = parts[0], int(parts[1])
            a = [float(x) for x in parts[2].rstrip(",").split(",") if x]
            c = [float(x) for x in parts[3].rstrip(",").split(",") if x]
            g = [float(x) for x in parts[4].rstrip(",").split(",") if x]
            t = [float(x) for x in parts[5].rstrip(",").split(",") if x]
            pwm = np.array([a, c, g, t], dtype=np.float32)
            if pwm.shape[1] == L:
                pwms[tf] = pwm
    return pwms

def vectorized_pwm_scan(seq, log_pwm, threshold):
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    L = log_pwm.shape[1]
    seq_upper = seq.upper()
    if len(seq_upper) < L:
        return []
    
    idxs = np.array([mapping.get(b, -1) for b in seq_upper], dtype=np.int32)
    bad = np.where(idxs == -1)[0]
    
    matches = []
    for i in range(len(seq_upper) - L + 1):
        if np.any((bad >= i) & (bad < i + L)):
            continue
        score = log_pwm[idxs[i:i+L], np.arange(L)].sum()
        if score >= threshold:
            matches.append(i)
    return matches

# ============================================================
# LOAD SHARED DATA
# ============================================================

print("Loading data...")
genome_seq = str(Fasta(CHR1_FASTA)[CHR_NAME]).upper()
chrom_len = len(genome_seq)
print(f"Chr1 length = {chrom_len:,}")

reg = pd.read_csv(REGULATORY_BED, sep="\t", header=None, usecols=[0, 1, 2], names=["chrom", "start", "end"])
reg = reg[reg["chrom"] == CHR_NAME].reset_index(drop=True)

fb = pd.read_csv(FACTORBOOK, sep="\t", header=None, names=["id", "chrom", "start", "end", "tf", "score", "strand"])
fb = fb[fb["chrom"] == CHR_NAME]

pwms = load_pwms()

# ============================================================
# PROCESS TF
# ============================================================

def process_tf(TF):
    print(f"[{TF}] starting...")
    if TF not in pwms:
        print(f"[{TF}] No PWM found")
        return

    pwm = pwms[TF]
    log_pwm = np.log(pwm + 1e-10)
    L = pwm.shape[1]

    tf_chip = fb[fb["tf"] == TF]
    chip_intervals = tf_chip[["start", "end"]].to_numpy(int)
    if len(chip_intervals) == 0:
        print(f"[{TF}] No chip-seq peaks")
        return

    # Compute threshold from random sequences
    random_scores = []
    bases = ["A", "C", "G", "T"]
    for _ in range(3000):
        seq = "".join(random.choice(bases) for _ in range(L))
        score = sum(log_pwm["ACGT".index(b), i] for i, b in enumerate(seq))
        random_scores.append(score)
    threshold = np.percentile(random_scores, PWM_SCORE_THRESHOLD_PERCENTILE)

    # Scan regulatory regions
    print(f"[{TF}] scanning...")
    hits = []
    for _, row in reg.iterrows():
        s, e = int(row.start), int(row.end)
        matches = vectorized_pwm_scan(genome_seq[s:e], log_pwm, threshold)
        for m in matches:
            hits.append((s + m, s + m + L))
    print(f"[{TF}] found {len(hits)} PWM hits")

    # Classify as bound/unbound
    pos, neg = [], []
    for s, e in hits:
        center = (s + e) // 2
        ws, we = center - HALF, center + HALF
        if ws < 0 or we > chrom_len:
            continue
        bound = np.any((chip_intervals[:, 0] < e) & (chip_intervals[:, 1] > s))
        if bound:
            pos.append((ws, we, 1))
        else:
            neg.append((ws, we, 0))

    if len(pos) == 0 or len(neg) == 0:
        print(f"[{TF}] insufficient samples")
        return

    # Balance and shuffle
    n = min(len(pos), len(neg))
    samples = random.sample(pos, n) + random.sample(neg, n)
    random.shuffle(samples)

    # Extract features
    Xs, Xstruct, Ys = [], [], []
    for ws, we, y in samples:
        seq = genome_seq[ws:we]
        feats = extract_struct_features(ws, we)
        if np.any(np.isnan(feats)):
            continue
        Xs.append(one_hot_encode(seq))
        Xstruct.append(feats)
        Ys.append(y)

    X_seq = np.stack(Xs)
    X_struct = np.stack(Xstruct)
    y = np.array(Ys)

    out = os.path.join(OUTPUT_DIR, f"{TF.lower()}_chr1_dataset_optionC.npz")
    np.savez(out, X_seq=X_seq, X_struct=X_struct, y=y)
    print(f"[{TF}] saved: {out} ({len(y)} samples)")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # n_workers = min(len(TARGET_TFS), cpu_count())
    # print(f"Using {n_workers} workers")
    # with Pool(n_workers, initializer=init_worker) as pool:
    #     pool.map(process_tf, TARGET_TFS)
    # print("Done.")
    init_worker() 
    process_tf("CTCF")