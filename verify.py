"""Check that the committed outputs are what the code produces.

  python3 verify.py            regenerate all tables and figures from outputs/json and compare
                               every file in outputs/json, paper/tables and paper/figs with MANIFEST.sha256
  python3 verify.py --write    (maintainers) rebuild MANIFEST.sha256 from the current files

Run ./reproduce.sh first to also regenerate outputs/json from data/ (about 45 minutes).
"""
import hashlib, os, subprocess, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
TRACKED = ("outputs/json", "paper/tables", "paper/figs")
MANIFEST = os.path.join(ROOT, "MANIFEST.sha256")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def current():
    out = {}
    for d in TRACKED:
        for f in sorted(os.listdir(os.path.join(ROOT, d))):
            if f.endswith((".json", ".tex", ".pdf")):
                out[f"{d}/{f}"] = digest(os.path.join(ROOT, d, f))
    return out


def regenerate():
    for s in ("redraw_budget.py", "make_figures.py", "make_tables.py", "make_fs_tables.py", "make_recon_table.py", "make_appendix_extras.py"):
        subprocess.run([sys.executable, os.path.join(ROOT, "analysis", s)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    if "--write" in sys.argv:
        regenerate()
        with open(MANIFEST, "w") as f:
            for k, v in current().items():
                f.write(f"{v}  {k}\n")
        print(f"wrote {MANIFEST}")
        sys.exit(0)
    want = dict(line.split()[::-1] for line in open(MANIFEST) if line.strip())
    regenerate()
    have = current()
    bad = [k for k in want if have.get(k) != want[k]]
    missing = [k for k in have if k not in want]
    for k in bad:
        print("MISMATCH", k)
    for k in missing:
        print("UNTRACKED", k)
    print(f"{len(want) - len(bad)}/{len(want)} tracked files match the manifest")
    sys.exit(1 if bad else 0)
