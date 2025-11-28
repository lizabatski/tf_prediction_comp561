import os
import shutil

def main():
    files = [f for f in os.listdir('.') if f.endswith('.png')]

    for fname in files:
        if "struct_only" in fname:
            out_dir = "struct_only"
        elif "pwm_only" in fname:
            out_dir = "pwm_only"
        else:
            print(f"Skipping (no mode match): {fname}")
            continue

        os.makedirs(out_dir, exist_ok=True)

        src = os.path.join(".", fname)
        dst = os.path.join(out_dir, fname)

        print(f"Moving {fname} → {dst}")
        shutil.move(src, dst)

    print("\nDone! Files sorted into:")
    print("  struct_only/")
    print("  pwm_only/")


if __name__ == "__main__":
    main()
