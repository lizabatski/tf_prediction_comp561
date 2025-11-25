import numpy as np
import os

print("="*60)
print("DIAGNOSTIC: Checking structural feature data")
print("="*60)

# Check what's actually in the saved dataset
data = np.load('/home/ekourb/tf/datasets_chr15/ctcf_chr15_dataset_struct.npz')
X_struct = data['X_struct']

print("\nStructural features analysis:")
print(f"Total samples: {X_struct.shape[0]}")
print(f"Total values: {X_struct.size}")
print(f"Total NaN: {np.isnan(X_struct).sum()}")
print(f"Total valid: {(~np.isnan(X_struct)).sum()}")

print("\nPer-feature breakdown:")
feature_names = ["MGW_mean", "MGW_std", "ProT_mean", "ProT_std", 
                 "Roll_mean", "Roll_std", "HelT_mean", "HelT_std"]
for i, name in enumerate(feature_names):
    col = X_struct[:, i]
    valid = ~np.isnan(col)
    print(f"{name:15s}: {valid.sum():5d} valid ({100*valid.sum()/len(col):5.1f}%), "
          f"mean={np.nanmean(col):7.3f}, std={np.nanstd(col):7.3f}")

# Check if WIG files exist
print("\n" + "="*60)
print("Checking WIG file existence:")
print("="*60)

wig_files = {
    "MGW": "data/structural/hg19.chr15.MGW.2nd.wig",
    "ProT": "data/structural/hg19.chr15.ProT.2nd.wig",
    "Roll": "data/structural/hg19.chr15.Roll.2nd.wig",
    "Buckle": "data/structural/hg19.chr15.Buckle.wig"
}

for name, path in wig_files.items():
    exists = os.path.exists(path)
    status = "✓ EXISTS" if exists else "✗ MISSING"
    print(f"{name:10s}: {status:10s} {path}")
    
    if exists:
        # Check file size and first few lines
        size = os.path.getsize(path) / (1024 * 1024)  # MB
        print(f"             Size: {size:.2f} MB")
        
        # Read first 10 lines
        with open(path, 'r') as f:
            print(f"             First lines:")
            for i, line in enumerate(f):
                if i >= 5:
                    break
                print(f"               {i+1}: {line.strip()}")

print("\n" + "="*60)
print("DIAGNOSIS:")
print("="*60)

# Diagnosis
valid_features = sum(~np.isnan(X_struct[:, i]).all() for i in range(8))
print(f"Features with valid data: {valid_features}/8")

if valid_features < 6:
    print("\n⚠️  PROBLEM DETECTED:")
    print("   Most structural features are completely NaN.")
    print("   Possible causes:")
    print("   1. WIG files don't exist at specified paths")
    print("   2. WIG files use different chromosome names (e.g., '15' vs 'chr15')")
    print("   3. WIG file format is different than expected")
    print("   4. Chromosome coordinates don't overlap with CTCF binding sites")
    print("\nRECOMMENDED ACTION:")
    print("   Run: head -20 data/structural/hg19.chr15.MGW.2nd.wig")
    print("   Check the chromosome name in the fixedStep/variableStep lines")
else:
    print("\n✓ Structural features loaded successfully!")