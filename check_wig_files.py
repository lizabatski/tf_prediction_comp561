import os
import gzip

print("="*70)
print("CHECKING WIG FILES FOR CHROMOSOME CONTENT")
print("="*70)

wig_dir = "data/structural"
wig_files = [
    "hg19.chr15.Buckle.wig",
    "hg19.chr15.MGW.2nd.wig",
    "hg19.chr15.ProT.2nd.wig",
    "hg19.chr15.Roll.2nd.wig"
]

for filename in wig_files:
    filepath = os.path.join(wig_dir, filename)
    
    print(f"\n{'='*70}")
    print(f"File: {filename}")
    print(f"{'='*70}")
    
    if not os.path.exists(filepath):
        print(f"❌ FILE NOT FOUND: {filepath}")
        continue
    
    # Check file size
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"Size: {size_mb:.2f} MB")
    
    # Read file and find all chromosome declarations
    chromosomes_found = set()
    line_count = 0
    chr15_found = False
    first_chrom = None
    
    try:
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                line_count += 1
                
                # Show first 5 lines
                if i < 5:
                    print(f"Line {i+1}: {line.strip()}")
                
                # Look for chromosome declarations
                if "chrom=" in line:
                    # Extract chromosome name
                    chrom = line.split("chrom=")[1].split()[0]
                    chromosomes_found.add(chrom)
                    
                    if first_chrom is None:
                        first_chrom = chrom
                    
                    if chrom == "chr15":
                        chr15_found = True
                        if i > 10:  # Only print if not in first few lines
                            print(f"\n✓ Found chr15 at line {i+1}: {line.strip()}")
                
                # Stop after reasonable number of lines
                if i > 1000000:
                    break
    
    except Exception as e:
        print(f"❌ ERROR reading file: {e}")
        continue
    
    # Summary
    print(f"\nTotal lines read: {line_count:,}")
    print(f"Chromosomes found: {sorted(chromosomes_found)}")
    print(f"First chromosome: {first_chrom}")
    print(f"Contains chr15: {'✓ YES' if chr15_found else '❌ NO'}")
    
    if not chr15_found:
        print(f"\n⚠️  PROBLEM: File named 'chr15' but does NOT contain chr15 data!")
        print(f"   This explains why your features are all NaN.")
    elif first_chrom != "chr15":
        print(f"\n⚠️  WARNING: File starts with {first_chrom}, not chr15")
        print(f"   You may need to extract chr15 data specifically.")

print("\n" + "="*70)
print("DIAGNOSIS SUMMARY")
print("="*70)

print("\nIf files are named 'chr15' but contain other chromosomes:")
print("  → Your preprocessing script is loading the wrong data")
print("  → Need to either:")
print("     1. Get chr15-only WIG files from RohsDB")
print("     2. Extract chr15 sections from genome-wide files")
print("     3. Fix the load_wig_track_to_array() function")