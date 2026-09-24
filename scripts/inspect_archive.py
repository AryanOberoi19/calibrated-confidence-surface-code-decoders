"""Inventory the Willow 105-qubit archive without extracting it.

Usage:
    python scripts/inspect_archive.py data/raw/google_105Q_surface_code_d3_d5_d7.zip
    python scripts/inspect_archive.py <zip> --md5      # also verify the download checksum (reads 5.7 GB)

Answers the open questions in spec section 14 (items 3-4) from the archive itself:
  * which files each experiment directory contains (label file? noisy circuit? DEM? predictions?)
  * distances, chip patches, bases and round counts present
  * shots per directory (from file sizes and detector counts), and whether files agree
  * what metadata.json holds (e.g. timestamps -> is shot order time order?)

Writes:
    docs/archive_README.txt   the archive's own README, verbatim
    docs/data_inventory.json  machine-readable inventory, one record per directory
    docs/data_inventory.md    human summary to paste into docs/data.md
"""
import argparse
import collections
import hashlib
import json
import math
import pathlib
import re
import sys
import zipfile

ZENODO_MD5 = "21fa6ad35b395d838ebcdbc92e364a12"   # as listed on zenodo.org/records/13273331; re-check there if it mismatches
CONFIG_RE = re.compile(r"(?P<d>d\d+)_at_(?P<patch>q\d+_\d+)/(?P<basis>[XZ])/r(?P<rounds>\d+)$")
REPO = pathlib.Path(__file__).resolve().parents[1]


def md5_of(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def shots_from_size(size, bits_per_shot, fmt):
    """Shot count implied by a Stim shot-data file size, or None if it doesn't divide evenly."""
    if bits_per_shot is None:
        return None
    if fmt == "b8":
        per = math.ceil(bits_per_shot / 8)
    elif fmt == "01":
        per = bits_per_shot + 1          # one char per bit plus newline
    else:
        return None
    return size // per if per and size % per == 0 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zip", type=pathlib.Path)
    ap.add_argument("--md5", action="store_true", help="verify the archive checksum (slow: reads the whole file)")
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "docs")
    args = ap.parse_args()

    if not args.zip.exists():
        sys.exit(f"not found: {args.zip}")
    size_gb = args.zip.stat().st_size / 1e9
    print(f"archive: {args.zip}  ({size_gb:.2f} GB)")

    if args.md5:
        digest = md5_of(args.zip)
        verdict = "OK" if digest == ZENODO_MD5 else "MISMATCH (re-check the value on the Zenodo page, then re-download)"
        print(f"md5: {digest}  expected {ZENODO_MD5}  -> {verdict}")

    try:
        zf = zipfile.ZipFile(args.zip)
    except zipfile.BadZipFile:
        sys.exit("not a readable zip: the download is probably incomplete or corrupt")

    try:
        import stim
    except ImportError:
        stim = None
        print("note: stim not installed, so detector counts and shot counts will be skipped")

    infos = [i for i in zf.infolist() if not i.is_dir()]
    print(f"files in archive: {len(infos)}")

    # --- README -----------------------------------------------------------
    args.out.mkdir(parents=True, exist_ok=True)
    readmes = [i for i in infos if re.search(r"readme", i.filename, re.I)]
    for i in readmes:
        print(f"README found: {i.filename} ({i.file_size} bytes)")
    if readmes:
        text = "\n\n".join(f"===== {i.filename} =====\n" + zf.read(i).decode("utf-8", "replace") for i in readmes)
        (args.out / "archive_README.txt").write_text(text)
        print(f"  -> saved to {args.out / 'archive_README.txt'}")
    else:
        print("no README in the archive")

    # --- group files by directory -----------------------------------------
    dirs = collections.defaultdict(dict)
    for i in infos:
        d, _, base = i.filename.rpartition("/")
        dirs[d][base] = i.file_size
    configs = []
    for d, files in sorted(dirs.items()):
        m = CONFIG_RE.search(d)
        if m:
            configs.append({"dir": d, "distance": int(m["d"][1:]), "patch": m["patch"], "basis": m["basis"],
                            "rounds": int(m["rounds"]), "files": files})
    other_dirs = sorted(d for d in dirs if not CONFIG_RE.search(d))
    print(f"experiment directories matching <d>_at_<patch>/<basis>/r<rounds>: {len(configs)}")
    if other_dirs:
        print(f"other directories ({len(other_dirs)}): " + ", ".join(other_dirs[:10]) + (" ..." if len(other_dirs) > 10 else ""))
    if not configs:
        sys.exit("no experiment directories recognised: the layout differs from expectations; inspect the README")

    # --- per-directory detail -------------------------------------------------
    meta_keys = collections.Counter()
    meta_example = None
    for c in configs:
        files = c["files"]
        stim_files = sorted(f for f in files if f.endswith(".stim"))
        c["circuit_files"] = stim_files
        c["num_detectors"] = c["num_observables"] = None
        if stim and stim_files:
            circ = stim.Circuit(zf.read(f"{c['dir']}/{stim_files[0]}").decode())
            c["num_detectors"], c["num_observables"] = circ.num_detectors, circ.num_observables
        shots = {}
        for f, size in files.items():
            if f.startswith("detection_events") and f.endswith(".b8"):
                shots[f] = shots_from_size(size, c["num_detectors"], "b8")
            elif "obs" in f and (f.endswith(".01") or f.endswith(".b8")):
                shots[f] = shots_from_size(size, c["num_observables"], f.rsplit(".", 1)[1])
            elif f.startswith("measurements") and f.endswith(".b8"):
                shots[f] = None   # needs num_measurements; not needed for the inventory
        c["shots_by_file"] = shots
        known = {v for v in shots.values() if v}
        c["shots"] = known.pop() if len(known) == 1 else (None if not known else sorted(known))
        if "metadata.json" in files:
            try:
                meta = json.loads(zf.read(f"{c['dir']}/metadata.json"))
                if isinstance(meta, dict):
                    meta_keys.update(meta.keys())
                    meta_example = meta_example or {"dir": c["dir"], "metadata": meta}
            except json.JSONDecodeError:
                meta_keys.update(["<unparseable>"])

    # --- summaries ----------------------------------------------------------
    by_d = collections.defaultdict(list)
    for c in configs:
        by_d[c["distance"]].append(c)
    signatures = collections.Counter(tuple(sorted(c["files"])) for c in configs)
    all_files = collections.Counter(f for c in configs for f in c["files"])

    lines = ["# Data inventory (generated by scripts/inspect_archive.py)", "",
             f"Archive: `{args.zip.name}`, {size_gb:.2f} GB, {len(infos)} files, {len(configs)} experiment directories.", ""]
    lines += ["## Configurations", "", "| Distance | Directories | Patches | Bases | Round counts | Shots per directory |",
              "|---|---|---|---|---|---|"]
    for d in sorted(by_d):
        cs = by_d[d]
        shots = sorted({c["shots"] for c in cs if isinstance(c["shots"], int)})
        shot_txt = (f"{shots[0]:,}" if len(shots) == 1 else
                    f"{min(shots):,}-{max(shots):,} (varies)" if shots else "unknown")
        lines.append(f"| {d} | {len(cs)} | {len({c['patch'] for c in cs})} ({', '.join(sorted({c['patch'] for c in cs}))}) | "
                     f"{', '.join(sorted({c['basis'] for c in cs}))} | {', '.join(str(r) for r in sorted({c['rounds'] for c in cs}))} | {shot_txt} |")
    lines += ["", "## Files present in experiment directories", "", "| File | Directories containing it |", "|---|---|"]
    for f, n in all_files.most_common():
        lines.append(f"| `{f}` | {n} of {len(configs)} |")
    lines += ["", f"Distinct file sets: {len(signatures)}"]
    lines += ["", "## Checks", ""]
    lines.append(f"- Label file candidates (name contains 'obs'): "
                 + (", ".join(f"`{f}`" for f in all_files if "obs" in f) or "**none found** - read the README"))
    lines.append(f"- Noisy circuit or DEM present: "
                 + (", ".join(f"`{f}`" for f in all_files if f.endswith(".dem") or ("noisy" in f and f.endswith(".stim")))
                    or "no (build the prior DEM from circuit_ideal.stim + SI1000)"))
    lines.append(f"- Decoder predictions present: "
                 + (", ".join(f"`{f}`" for f in all_files if "predict" in f) or "no (validate against the paper's rates)"))
    mism = [c["dir"] for c in configs if isinstance(c["shots"], list)]
    lines.append(f"- Directories whose files disagree on shot count: {len(mism)}" + (f" (e.g. `{mism[0]}`)" if mism else ""))
    d3z = [c for c in configs if c["distance"] == 3 and c["basis"] == "Z" and c["num_detectors"] is not None]
    bad = [c for c in d3z if c["num_detectors"] != 8 * c["rounds"]]
    lines.append(f"- d=3 Z-basis detector count equals 8r: {len(d3z) - len(bad)} of {len(d3z)} directories")
    lines += ["", "## metadata.json", "", "Keys (count of directories): " +
              (", ".join(f"`{k}` ({v})" for k, v in meta_keys.most_common()) or "none"), ""]
    if meta_example:
        lines += ["Example:", "", "```json", json.dumps(meta_example, indent=2)[:3000], "```"]
    time_keys = [k for k in meta_keys if re.search(r"time|date|start|stamp|order", k, re.I)]
    lines += ["", f"Time-related keys: {', '.join(f'`{k}`' for k in time_keys) if time_keys else 'none found'}"]

    (args.out / "data_inventory.md").write_text("\n".join(lines) + "\n")
    (args.out / "data_inventory.json").write_text(json.dumps(configs, indent=1))
    print("\n".join(lines))
    print(f"\nwrote {args.out / 'data_inventory.md'} and {args.out / 'data_inventory.json'}")


if __name__ == "__main__":
    main()
