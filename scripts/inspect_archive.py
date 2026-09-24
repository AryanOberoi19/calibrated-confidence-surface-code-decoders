"""Inventory the Willow 105-qubit archive without extracting it.

Usage:
    python scripts/inspect_archive.py data/raw/google_105Q_surface_code_d3_d5_d7.zip
    python scripts/inspect_archive.py <zip> --md5      # also verify the download checksum (reads 5.7 GB)

Answers the open questions in spec section 14 (items 3-4) from the archive itself:
  * which files each experiment directory contains (label file? noisy circuit? DEM? predictions?),
    including decoder outputs under <exp>/decoding_results/<pathway>/
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

ZENODO_MD5 = "21fa6ad35b395d838ebcdbc92e364a12"   # as listed on zenodo.org/records/13273331; confirmed on a full download
CONFIG_RE = re.compile(r"(?P<d>d\d+)_at_(?P<patch>q\d+_\d+)/(?P<basis>[XZ])/r(?P<rounds>\d+)$")
# decoder outputs live one level below an experiment directory: <exp>/decoding_results/<pathway>/<file>
PATHWAY_RE = re.compile(r"^(?P<exp>.+)/decoding_results/(?P<pathway>[^/]+)/(?P<file>[^/]+)$")
TEXT_EXT = (".md", ".txt", ".rst")
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
    readme_all = [i for i in infos if re.search(r"readme", i.filename, re.I)]
    readmes = [i for i in readme_all if i.filename.lower().endswith(TEXT_EXT)]
    for i in readme_all:
        print(f"README file: {i.filename} ({i.file_size} bytes)" + ("" if i in readmes else "  [not text, not copied]"))
    if readmes:
        text = "\n\n".join(f"===== {i.filename} =====\n" + zf.read(i).decode("utf-8", "replace") for i in readmes)
        (args.out / "archive_README.txt").write_text(text)
        print(f"  -> saved to {args.out / 'archive_README.txt'}")
    else:
        print("no README in the archive")

    # --- group files by directory -----------------------------------------
    dirs = collections.defaultdict(dict)
    pathways = collections.defaultdict(lambda: collections.defaultdict(dict))
    for i in infos:
        m = PATHWAY_RE.match(i.filename)
        if m and CONFIG_RE.search(m["exp"]):
            pathways[m["exp"]][m["pathway"]][m["file"]] = i.file_size
            continue
        d, _, base = i.filename.rpartition("/")
        dirs[d][base] = i.file_size
    configs = []
    for d, files in sorted(dirs.items()):
        m = CONFIG_RE.search(d)
        if m:
            configs.append({"dir": d, "distance": int(m["d"][1:]), "patch": m["patch"], "basis": m["basis"],
                            "rounds": int(m["rounds"]), "files": files,
                            "pathways": {p: dict(f) for p, f in sorted(pathways.get(d, {}).items())}})
    other_dirs = sorted(d for d in dirs if not CONFIG_RE.search(d))
    print(f"experiment directories matching <d>_at_<patch>/<basis>/r<rounds>: {len(configs)}")
    print(f"decoding-pathway folders attached to experiment directories: {sum(len(v) for v in pathways.values())}")
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
        for p, pfiles in c["pathways"].items():
            for f, size in pfiles.items():
                if f.startswith("obs_flips_predicted") and f.endswith((".b8", ".01")):
                    shots[f"{p}/{f}"] = shots_from_size(size, c["num_observables"], f.rsplit(".", 1)[1])
        c["shots_by_file"] = shots
        known = {v for v in shots.values() if v}
        c["shots"] = known.pop() if len(known) == 1 else (None if not known else sorted(known))
        c["meta_shots"] = None
        if "metadata.json" in files:
            try:
                meta = json.loads(zf.read(f"{c['dir']}/metadata.json"))
                if isinstance(meta, dict):
                    meta_keys.update(meta.keys())
                    meta_example = meta_example or {"dir": c["dir"], "metadata": meta}
                    c["meta_shots"] = meta.get("shots")
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

    # --- decoding pathways (decoding_results/<pathway>/) ------------------------
    pw_count = collections.Counter(p for c in configs for p in c["pathways"])
    pw_files = collections.Counter(f for c in configs for pf in c["pathways"].values() for f in pf)
    lines += ["", "## Decoding pathways", ""]
    if pw_count:
        lines += ["| Pathway | Directories with it | Missing from (distance, rounds: count) |", "|---|---|---|"]
        for p, n in sorted(pw_count.items()):
            miss = collections.Counter((c["distance"], c["rounds"]) for c in configs if p not in c["pathways"])
            miss_txt = ", ".join(f"d{d} r{r}: {k}" for (d, r), k in sorted(miss.items())) or "-"
            lines.append(f"| `{p}` | {n} of {len(configs)} | {miss_txt} |")
        lines += ["", "Files inside pathway folders: " + ", ".join(f"`{f}` ({n})" for f, n in pw_files.most_common())]
    else:
        lines.append("none")

    lines += ["", "## Checks", ""]
    lines.append(f"- Label file candidates (name contains 'obs' and 'actual'): "
                 + (", ".join(f"`{f}`" for f in all_files if "obs" in f and "predict" not in f) or "**none found** - read the README"))
    dems = [f for f in all_files if f.endswith(".dem")] + (["`<pathway>/error_model.dem`"] if pw_files.get("error_model.dem") else [])
    lines.append(f"- Noisy circuit or DEM present: "
                 + (", ".join([f"`{f}`" for f in all_files if "noisy" in f and f.endswith(".stim")] + dems)
                    or "no (build the prior DEM from circuit_ideal.stim + SI1000)"))
    lines.append(f"- Decoder predictions present: "
                 + (f"yes, {len(pw_count)} pathways (see above)" if any("predict" in f for f in pw_files)
                    else ", ".join(f"`{f}`" for f in all_files if "predict" in f) or "no (validate against the paper's rates)"))
    mism = [c["dir"] for c in configs if isinstance(c["shots"], list)]
    lines.append(f"- Directories whose files (incl. predictions) disagree on shot count: {len(mism)}"
                 + (f" (e.g. `{mism[0]}`)" if mism else ""))
    meta_mism = [c["dir"] for c in configs if c["meta_shots"] is not None and c["meta_shots"] != c["shots"]]
    lines.append(f"- Directories where metadata.json `shots` differs from the file-derived count: {len(meta_mism)}"
                 + (f" (e.g. `{meta_mism[0]}`)" if meta_mism else ""))
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
