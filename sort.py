import os
import shutil

BASE = "."   # current directory

# Output folders
folders = {
    "struct_only": "struct_only",
    "pwm_only": "pwm_only",
    "struct_pwm": "struct_pwm",
}

# Create directories if missing
for f in folders.values():
    os.makedirs(f, exist_ok=True)

# Scan files in directory
for fname in os.listdir(BASE):

    if os.path.isdir(fname):
        continue  # skip folders

    # Decide category
    if "struct_pwm" in fname:
        dest = folders["struct_pwm"]
    elif "struct_only" in fname:
        dest = folders["struct_only"]
    elif "pwm_only" in fname:
        dest = folders["pwm_only"]
    else:
        continue  # not a result file

    src_path = os.path.join(BASE, fname)
    dst_path = os.path.join(dest, fname)

    print(f"Moving {fname} → {dest}/")
    shutil.move(src_path, dst_path)

print("\nDone! Files sorted into:")
for key, val in folders.items():
    print(f"  - {val}/")
