#!/usr/bin/env python3

import pandas as pd
from pathlib import Path
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path("data")
FACTORBOOK_DIR = DATA_DIR / "factorbook"
MOTIF_POS_FILE = FACTORBOOK_DIR / "factorbookMotifPos.txt"

# Target TFs for this study
TARGET_TFS = ["CTCF", "EGR1", "GATA1"]

# We'll focus on chromosome 15
TARGET_CHROM = "chr15"

# ============================================================================
# CHECK TF AVAILABILITY
# ============================================================================

def check_tf_availability(motif_file, target_tfs, target_chrom=None):
    """
    Check if target TFs exist in the factorbook data.
    Optionally filter by chromosome.
    """
    print("="*80)
    print("TRANSCRIPTION FACTOR AVAILABILITY CHECK")
    print("="*80)
    print(f"\nTarget TFs: {', '.join(target_tfs)}")
    if target_chrom:
        print(f"Target chromosome: {target_chrom}")
    print(f"\nReading: {motif_file}")
    print("-"*80)
    
    # Count all TFs in the file
    all_tf_counts = defaultdict(int)
    chrom_specific_counts = defaultdict(lambda: defaultdict(int))
    
    # Also collect detailed info for target TFs
    target_tf_data = {tf: [] for tf in target_tfs}
    
    with open(motif_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split('\t')
            
            if len(parts) < 7:
                continue
            
            tf_name = parts[4]
            chrom = parts[1]
            start = int(parts[2])
            end = int(parts[3])
            strand = parts[6]
            score = float(parts[5]) if parts[5] != '.' else 0.0
            
            # Count all TFs
            all_tf_counts[tf_name] += 1
            chrom_specific_counts[tf_name][chrom] += 1
            
            # Collect detailed data for target TFs
            if tf_name in target_tfs:
                target_tf_data[tf_name].append({
                    'chrom': chrom,
                    'start': start,
                    'end': end,
                    'strand': strand,
                    'score': score,
                    'length': end - start
                })
    
    print(f"\n✅ Processed {line_num:,} lines from factorbook file")
    print(f"✅ Found {len(all_tf_counts)} unique transcription factors")
    
    # ========================================================================
    # REPORT ON TARGET TFS
    # ========================================================================
    
    print("\n" + "="*80)
    print("TARGET TRANSCRIPTION FACTORS - DETAILED REPORT")
    print("="*80)
    
    for tf in target_tfs:
        print(f"\n{'─'*80}")
        print(f"📊 {tf}")
        print(f"{'─'*80}")
        
        if tf not in all_tf_counts:
            print(f"❌ NOT FOUND in factorbook data!")
            print(f"   This TF does not exist in the dataset.")
            continue
        
        # Overall statistics
        total_sites = all_tf_counts[tf]
        print(f"✅ FOUND: {total_sites:,} total binding sites across all chromosomes")
        
        # Convert to dataframe for easier analysis
        df = pd.DataFrame(target_tf_data[tf])
        
        if len(df) == 0:
            print(f"   Warning: No data collected for {tf}")
            continue
        
        # Chromosome-specific statistics
        print(f"\n   Binding sites per chromosome:")
        chrom_counts = df['chrom'].value_counts().sort_index()
        
        # Show all chromosomes with counts
        for chrom in sorted(chrom_counts.index):
            count = chrom_counts[chrom]
            percentage = 100 * count / total_sites
            marker = " ⭐" if chrom == target_chrom else ""
            print(f"      {chrom}: {count:>6,} sites ({percentage:>5.2f}%){marker}")
        
        # Highlight target chromosome
        if target_chrom:
            chrom_df = df[df['chrom'] == target_chrom]
            print(f"\n   🎯 {target_chrom} STATISTICS:")
            print(f"      Sites on {target_chrom}: {len(chrom_df):,}")
            
            if len(chrom_df) > 0:
                print(f"      Percentage of total: {100*len(chrom_df)/total_sites:.2f}%")
                print(f"      Strand distribution:")
                print(f"         Plus strand  (+): {(chrom_df['strand'] == '+').sum():,}")
                print(f"         Minus strand (-): {(chrom_df['strand'] == '-').sum():,}")
                
                # Length statistics
                lengths = chrom_df['length']
                print(f"\n      Binding site lengths on {target_chrom}:")
                print(f"         Mean:   {lengths.mean():.1f} bp")
                print(f"         Median: {lengths.median():.1f} bp")
                print(f"         Min:    {lengths.min()} bp")
                print(f"         Max:    {lengths.max()} bp")
                print(f"         Std:    {lengths.std():.1f} bp")
                
                # Score statistics
                scores = chrom_df['score']
                print(f"\n      Binding scores on {target_chrom}:")
                print(f"         Mean:   {scores.mean():.2f}")
                print(f"         Median: {scores.median():.2f}")
                print(f"         Min:    {scores.min():.2f}")
                print(f"         Max:    {scores.max():.2f}")
                print(f"         Std:    {scores.std():.2f}")
            else:
                print(f"      ⚠️  WARNING: No sites found on {target_chrom}!")
    
    # ========================================================================
    # COMPARISON TABLE
    # ========================================================================
    
    print("\n" + "="*80)
    print("COMPARATIVE SUMMARY")
    print("="*80)
    print(f"\n{'TF':<12} {'Total Sites':>15} {f'{target_chrom} Sites':>15} {'% on {}'.format(target_chrom):>15}")
    print("-"*80)
    
    for tf in target_tfs:
        if tf in all_tf_counts:
            total = all_tf_counts[tf]
            chrom_count = chrom_specific_counts[tf].get(target_chrom, 0)
            percentage = 100 * chrom_count / total if total > 0 else 0
            
            print(f"{tf:<12} {total:>15,} {chrom_count:>15,} {percentage:>14.2f}%")
        else:
            print(f"{tf:<12} {'NOT FOUND':>15} {'N/A':>15} {'N/A':>15}")
    
    # ========================================================================
    # RECOMMENDATIONS
    # ========================================================================
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR DATASET CREATION")
    print("="*80)
    
    all_found = all(tf in all_tf_counts for tf in target_tfs)
    
    if all_found:
        print("\n✅ All target TFs found in the dataset!")
        
        if target_chrom:
            chrom_viable = all(
                chrom_specific_counts[tf].get(target_chrom, 0) > 100 
                for tf in target_tfs
            )
            
            if chrom_viable:
                print(f"✅ All TFs have sufficient sites on {target_chrom} (>100 each)")
                print(f"\n📋 Recommended next steps:")
                print(f"   1. Extract positive samples from {target_chrom} for each TF")
                print(f"   2. Generate matched negative samples from {target_chrom} regulatory regions")
                print(f"   3. Use remaining chromosomes (chr1-chr14, chr16-chr22, chrX) for testing")
                print(f"   4. Consider using chr1-chr14 for training, {target_chrom} for validation,")
                print(f"      and chr16-chr22 for testing")
            else:
                print(f"⚠️  WARNING: Some TFs have few sites on {target_chrom}")
                print(f"   Consider using a different chromosome or multiple chromosomes")
                
                # Find best chromosome
                best_chrom = None
                best_count = 0
                
                for chrom in [f'chr{i}' for i in range(1, 23)] + ['chrX']:
                    min_sites = min(
                        chrom_specific_counts[tf].get(chrom, 0) 
                        for tf in target_tfs
                    )
                    if min_sites > best_count:
                        best_count = min_sites
                        best_chrom = chrom
                
                print(f"\n   💡 Better chromosome option: {best_chrom}")
                print(f"      Minimum sites across all TFs: {best_count:,}")
    else:
        missing = [tf for tf in target_tfs if tf not in all_tf_counts]
        print(f"\n❌ Missing TFs: {', '.join(missing)}")
        print(f"   These TFs are not available in the factorbook data.")
        print(f"\n   Available TFs with most binding sites:")
        
        top_tfs = sorted(all_tf_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        for i, (tf, count) in enumerate(top_tfs, 1):
            print(f"      {i:2d}. {tf:<15} {count:>8,} sites")
    
    print("\n" + "="*80)
    
    return target_tf_data, all_tf_counts, chrom_specific_counts


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main function to check TF availability."""
    
    if not MOTIF_POS_FILE.exists():
        print(f"❌ ERROR: {MOTIF_POS_FILE} not found!")
        print(f"   Please ensure the factorbook data is in the correct location.")
        return
    
    # Check TF availability
    tf_data, all_counts, chrom_counts = check_tf_availability(
        MOTIF_POS_FILE, 
        TARGET_TFS,
        target_chrom=TARGET_CHROM
    )
    
    # Save detailed data for target TFs if needed
    print(f"\n💾 Saving detailed data for target TFs...")
    output_dir = Path("tf_check_results")
    output_dir.mkdir(exist_ok=True)
    
    for tf in TARGET_TFS:
        if tf in all_counts and len(tf_data[tf]) > 0:
            df = pd.DataFrame(tf_data[tf])
            output_file = output_dir / f"{tf}_all_sites.csv"
            df.to_csv(output_file, index=False)
            print(f"   Saved {tf} data to {output_file}")
            
            # Also save chromosome-specific data
            if TARGET_CHROM:
                chrom_df = df[df['chrom'] == TARGET_CHROM]
                if len(chrom_df) > 0:
                    chrom_file = output_dir / f"{tf}_{TARGET_CHROM}_sites.csv"
                    chrom_df.to_csv(chrom_file, index=False)
                    print(f"   Saved {tf} {TARGET_CHROM} data to {chrom_file}")
    
    print("\n✅ TF availability check complete!")
    print(f"   Results saved to: {output_dir}/")
    print("="*80)


if __name__ == "__main__":
    main()