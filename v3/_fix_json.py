"""One-shot script to fix NaN/Infinity in existing JSON files."""
import json, re, glob, os

raw_dir = os.path.join("experiments", "output", "raw")
archive_dir = os.path.join("experiments", "output", "archive")

patterns = [
    os.path.join(raw_dir, "*.json"),
    os.path.join(archive_dir, "**", "*.json"),
]

fixed_count = 0
for pat in patterns:
    for path in glob.glob(pat, recursive=True):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        n_nan = text.count("NaN")
        n_inf = text.count("Infinity")
        if n_nan == 0 and n_inf == 0:
            continue
        fixed = re.sub(r'(?<!["\w])NaN(?!["\w])', "null", text)
        fixed = re.sub(r'(?<!["\w])-?Infinity(?!["\w])', "null", fixed)
        # Validate
        try:
            json.loads(fixed)
        except json.JSONDecodeError as e:
            print(f"  SKIP (parse error): {path} — {e}")
            continue
        with open(path, "w", encoding="utf-8") as f:
            f.write(fixed)
        print(f"  FIXED: {path}  ({n_nan} NaN, {n_inf} Infinity → null)")
        fixed_count += 1

print(f"\nDone. Fixed {fixed_count} file(s).")
