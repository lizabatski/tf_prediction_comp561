#!/usr/bin/env python3

print("========================================")
print(" Testing TF-MoDISco-related imports")
print("========================================")

# ----------------------------------------------------
# 1. Test hdf5plugin import BEFORE h5py
# ----------------------------------------------------
print("\n[1] Importing hdf5plugin...")
try:
    import hdf5plugin
    print("✓ hdf5plugin import OK")
except Exception as e:
    print("✗ hdf5plugin FAILED:", e)

# ----------------------------------------------------
# 2. Now import h5py (requires hdf5plugin to be already loaded)
# ----------------------------------------------------
print("\n[2] Importing h5py...")
try:
    import h5py
    print("✓ h5py import OK, version:", h5py.__version__)
except Exception as e:
    print("✗ h5py FAILED:", e)

# ----------------------------------------------------
# 3. Test modiscolite
# ----------------------------------------------------
print("\n[3] Importing modiscolite package...")
try:
    import modiscolite
    print("✓ modiscolite import OK")
except Exception as e:
    print("✗ modiscolite FAILED:", e)

# ----------------------------------------------------
# 4. Import TFMoDISco specifically
# ----------------------------------------------------
print("\n[4] Importing TFMoDISco class from modiscolite.tfmodisco...")
try:
    from modiscolite.tfmodisco import TFMoDISco
    print("✓ TFMoDISco import OK")
except Exception as e:
    print("✗ TFMoDISco FAILED:", e)

print("\n========================================")
print(" Finished import test")
print("========================================")
