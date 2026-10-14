#!/usr/bin/env python
"""Recover per-run MedMNIST results from existing output folders. READ-ONLY.

Run on the training machine, from the folder that contains `experiments-main/`:

    pip install tensorboard numpy
    python extract_run_results.py

For every finished run it reads the per-epoch val/test AUC that the released
script logged to TensorBoard, and computes two scores:

  best_val : test AUC at the epoch with the highest val AUC. This is what the
             CURRENT released script reports (best_model = deepcopy(model)) and
             what we have reported so far.
  last     : test AUC after the final epoch. The paper-era script (commit
             d385743) had `best_model = model`, an alias rather than a copy, so
             its final evaluation was effectively the last-epoch model.

Nothing existing is modified. New files are written to the current folder:
  per_run_results.csv, summary_by_dataset.csv,
  spreadsheet_runs_best_val.csv, spreadsheet_runs_last_epoch.csv,
  raw_outputs_backup.zip  (CSVs, logs and TensorBoard events; no model weights)
"""
import argparse
import csv
import os
import re
import sys
import zipfile

import numpy as np

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    sys.exit("Missing dependency. Run:  pip install tensorboard")

PAPER_AUC = {
    "chestmnist": 0.773, "pathmnist": 0.989, "dermamnist": 0.912, "octmnist": 0.958,
    "pneumoniamnist": 0.962, "retinamnist": 0.716, "breastmnist": 0.866,
    "bloodmnist": 0.997, "tissuemnist": 0.932, "organamnist": 0.998,
    "organcmnist": 0.993, "organsmnist": 0.975, "organmnist3d": 0.994,
    "nodulemnist3d": 0.875, "fracturemnist3d": 0.725, "adrenalmnist3d": 0.828,
    "vesselmnist3d": 0.907, "synapsemnist3d": 0.851,
}
# Dataset order used in the output tables.
ORDER = [
    "chestmnist", "pathmnist", "dermamnist", "octmnist", "pneumoniamnist", "retinamnist",
    "breastmnist", "bloodmnist", "tissuemnist", "organamnist", "organcmnist", "organsmnist",
    "organmnist3d", "nodulemnist3d", "fracturemnist3d", "adrenalmnist3d", "vesselmnist3d",
    "synapsemnist3d",
]
# Epoch count each dataset is reported at in the report and spreadsheet.
REPORTED_EPOCHS = {d: 20 for d in ORDER}
for _d in ("breastmnist", "nodulemnist3d", "adrenalmnist3d", "vesselmnist3d", "synapsemnist3d"):
    REPORTED_EPOCHS[_d] = 100

RUN_RE = re.compile(r"^run([1-9]\d*)(_ep100)?(_bs64)?$")
CSV_RE = re.compile(
    r"^(?P<flag>[a-z0-9]+)_test_\[AUC\](?P<auc>[\d.]+)_\[ACC\](?P<acc>[\d.]+)@(?P<run>.+)\.csv$"
)
TAGS = ("val_auc", "test_auc", "val_acc", "test_acc")


def read_scalars(tb_dir):
    ea = EventAccumulator(tb_dir, size_guidance={"scalars": 0})
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))
    out = {}
    for tag in TAGS:
        if tag not in available:
            return None
        by_step = {}
        for ev in ea.Scalars(tag):
            by_step[ev.step] = ev.value  # last write wins if an event file was restarted
        out[tag] = np.array([by_step[s] for s in sorted(by_step)], dtype=float)
    if len({len(v) for v in out.values()}) != 1:
        return None
    return out


def find_run_dirs(root):
    found = []
    for sub in ("MedMNIST2D", "MedMNIST3D"):
        base = os.path.join(root, "experiments-main", sub, "output")
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, _ in os.walk(base):
            if os.path.basename(dirpath) == "Tensorboard_Results":
                found.append(os.path.dirname(dirpath))
                dirnames[:] = []
    return sorted(found)


def analyse(run_dir):
    match = None
    for name in os.listdir(run_dir):
        match = CSV_RE.match(name)
        if match:
            break
    if match is None:
        return None, "no test CSV (unfinished run?)"
    run_name = match.group("run")
    if not RUN_RE.match(run_name):
        return None, "ignored run name %r" % run_name
    sc = read_scalars(os.path.join(run_dir, "Tensorboard_Results"))
    if sc is None:
        return None, "TensorBoard scalars missing or incomplete"
    n = len(sc["test_auc"])
    best = int(np.argmax(sc["val_auc"]))  # first maximum, same as the script's strict '>'
    row = {
        "dataset": match.group("flag"),
        "run": run_name,
        "variant": "bs64" if RUN_RE.match(run_name).group(3) else "",
        "epochs": n,
        "best_epoch": best,
        "val_auc_at_best": float(sc["val_auc"][best]),
        "test_auc_best_val": float(sc["test_auc"][best]),
        "test_acc_best_val": float(sc["test_acc"][best]),
        "val_auc_last": float(sc["val_auc"][-1]),
        "test_auc_last": float(sc["test_auc"][-1]),
        "test_acc_last": float(sc["test_acc"][-1]),
        "file_auc": float(match.group("auc")),
        "run_dir": run_dir,
    }
    row["consistent_with_filename"] = abs(row["test_auc_best_val"] - row["file_auc"]) < 0.001
    return row, None


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="folder that contains experiments-main/")
    args = ap.parse_args()

    run_dirs = find_run_dirs(args.root)
    if not run_dirs:
        sys.exit("No Tensorboard_Results folders found under %s/experiments-main/*/output" % args.root)
    print("Found %d run folders with TensorBoard logs." % len(run_dirs))

    rows, skipped = {}, []
    for rd in run_dirs:
        row, why = analyse(rd)
        if row is None:
            skipped.append((rd, why))
            continue
        key = (row["dataset"], row["run"], row["epochs"])
        if key in rows:
            print("WARNING duplicate run %s, keeping the later folder." % (key,))
        rows[key] = row  # run_dirs are sorted, so the later timestamp wins
    rows = list(rows.values())
    print("Usable runs: %d.  Skipped: %d" % (len(rows), len(skipped)))
    for rd, why in skipped:
        print("  skipped %s -> %s" % (rd, why))
    bad = [r for r in rows if not r["consistent_with_filename"]]
    if bad:
        print("WARNING: %d runs whose best-val test AUC disagrees with the CSV filename:" % len(bad))
        for r in bad:
            print("  %s %s ep%d  logged=%.4f  filename=%.3f" % (
                r["dataset"], r["run"], r["epochs"], r["test_auc_best_val"], r["file_auc"]))

    fields = ["dataset", "run", "variant", "epochs", "best_epoch", "val_auc_at_best", "test_auc_best_val",
              "test_acc_best_val", "val_auc_last", "test_auc_last", "test_acc_last", "file_auc",
              "consistent_with_filename", "run_dir"]
    rows.sort(key=lambda r: (ORDER.index(r["dataset"]) if r["dataset"] in ORDER else 99,
                             r["epochs"], r["variant"], int(RUN_RE.match(r["run"]).group(1))))
    write_csv("per_run_results.csv", rows, fields)

    # ---- aggregate per (dataset, epochs, variant), signed gap = ours - paper, in AUC points ----
    groups = {}
    for r in rows:
        groups.setdefault((r["dataset"], r["epochs"], r["variant"]), []).append(r)
    summary = []
    for (ds, ep, var), rs in groups.items():
        paper = PAPER_AUC.get(ds)
        bv = np.array([r["file_auc"] for r in rs])  # 3-decimal values, as reported so far
        la = np.array([r["test_auc_last"] for r in rs])
        entry = {
            "dataset": ds, "epochs": ep, "variant": var, "n_runs": len(rs), "paper_auc": paper,
            "best_val_mean": bv.mean(), "best_val_std": bv.std(),
            "best_val_gap_pts": (bv.mean() - paper) * 100 if paper else None,
            "last_mean": la.mean(), "last_std": la.std(),
            "last_gap_pts": (la.mean() - paper) * 100 if paper else None,
            "mean_best_epoch": float(np.mean([r["best_epoch"] for r in rs])),
            "mean_val_auc_at_best": float(np.mean([r["val_auc_at_best"] for r in rs])),
        }
        summary.append(entry)
    summary.sort(key=lambda e: (ORDER.index(e["dataset"]) if e["dataset"] in ORDER else 99, e["epochs"], e["variant"]))
    sfields = ["dataset", "epochs", "variant", "n_runs", "paper_auc", "best_val_mean", "best_val_std",
               "best_val_gap_pts", "last_mean", "last_std", "last_gap_pts", "mean_best_epoch",
               "mean_val_auc_at_best"]
    write_csv("summary_by_dataset.csv", summary, sfields)

    print("\nSigned gap = ours minus paper, AUC points (negative = a performance LOSS). std is population std.")
    print("%-16s %3s %4s %6s | %-24s | %-24s | %s" % (
        "dataset", "ep", "runs", "paper", "best-val (reported so far)", "last epoch (paper-era)", "best ep"))
    for e in summary:
        print("%-16s %3d %4d  %.3f | %.4f +/- %.4f (%+6.2f) | %.4f +/- %.4f (%+6.2f) | %5.1f" % (
            e["dataset"] + (" bs64" if e["variant"] else ""), e["epochs"], e["n_runs"], e["paper_auc"] or 0,
            e["best_val_mean"], e["best_val_std"], e["best_val_gap_pts"] or 0,
            e["last_mean"], e["last_std"], e["last_gap_pts"] or 0, e["mean_best_epoch"]))

    # ---- spreadsheet rows 3-7: Run 1..5 x 18 datasets, at each dataset's reported epoch count ----
    for label, field, rnd in (("best_val", "file_auc", 3), ("last_epoch", "test_auc_last", 3)):
        table = {}
        for r in rows:
            idx = int(RUN_RE.match(r["run"]).group(1))
            if r["epochs"] == REPORTED_EPOCHS.get(r["dataset"]) and idx <= 5 and not r["variant"]:
                table[(r["dataset"], idx)] = round(r[field], rnd)
        out = []
        for i in range(1, 6):
            line = {"Run": "Run %d" % i}
            for ds in ORDER:
                line[ds] = table.get((ds, i), "")
            out.append(line)
        write_csv("spreadsheet_runs_%s.csv" % label, out, ["Run"] + ORDER)

    # ---- durable backup of everything small ----
    zpath = "raw_outputs_backup.zip"
    count = 0
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for sub in ("MedMNIST2D", "MedMNIST3D"):
            base = os.path.join(args.root, "experiments-main", sub, "output")
            for dirpath, _, files in os.walk(base):
                for fn in files:
                    if fn.lower().endswith(".pth"):
                        continue
                    full = os.path.join(dirpath, fn)
                    z.write(full, os.path.relpath(full, args.root))
                    count += 1
    print("\nBackup: %s  (%d files, %.1f MB, model weights excluded)" % (
        zpath, count, os.path.getsize(zpath) / 1e6))
    print("Wrote: per_run_results.csv, summary_by_dataset.csv, "
          "spreadsheet_runs_best_val.csv, spreadsheet_runs_last_epoch.csv")
    print("Please copy ALL of these (especially the zip) off the machine before anything else.")


if __name__ == "__main__":
    main()
