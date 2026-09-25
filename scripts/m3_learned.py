"""M3: a small learned decoder at d = 3 (spec section 8.3), trained in two stages and then run on every shot.

  pretrain   shots simulated from each experiment's RL-optimized DEM (Stim), fresh every step
  finetune   hardware shots from the Train split only: 15,000 of each experiment's 20,000 Train shots
             (13,500 to fit, 1,500 to pick the best epoch). The other 5,000 Train shots are left unseen so the
             calibrators and thresholds of M4 are not fitted on shots the network was trained on.
  infer      prediction and logit for all 50,000 shots of each experiment

Scope: d = 3, all 9 patches and both bases, r in {1, 10, 30, 50} (the same subset as belief-matching), RL prior.
Outputs: results/cache/learned/{pretrained,finetuned}.pt, per-shot outputs in results/cache/soft/nn/rl/<key>.npz
(pred, logit, fit_mask), the training log results/m3_training.csv and per-experiment counts results/m3_learned.csv.

    python scripts/m3_learned.py all                      # pretrain, finetune, infer (MPS if available)
    python scripts/m3_learned.py pretrain --steps 20000
    python scripts/m3_learned.py finetune --epochs 4
    python scripts/m3_learned.py infer --device cpu
"""
import argparse
import csv
import math
import pathlib
import time

import numpy as np
import torch
from torch import nn

from qeccal.data import list_experiments, split_indices
from qeccal.data.splits import SEED
from qeccal.learned.features import Layout
from qeccal.learned.model import GRUDecoder, device
from qeccal.soft.cache import CACHE, cache_path, load, save

REPO = pathlib.Path(__file__).resolve().parents[1]
RES = REPO / "results"
CKPT = CACHE.parent / "learned"
METHOD, PRIOR = "nn", "rl"
ROUNDS = (1, 10, 30, 50)
FIT, VAL = 0.675, 0.075                            # of each experiment's Train indices: 13,500 and 1,500 of 20,000
LOG_FIELDS = ["stage", "step", "epoch", "shots_seen", "loss", "val_loss", "val_error", "seconds"]


class Data:
    """The d = 3 experiments with their layouts, patch and basis ids."""

    def __init__(self):
        self.exps = list_experiments(distance=3, rounds=ROUNDS)
        self.patches = sorted({e.patch for e in self.exps})
        circuits = {e.key: e.circuit() for e in self.exps}
        self.slots = max(Layout(c).slots for c in circuits.values())       # 8 at d = 3; r = 1 steps have 4
        self.layout = {k: Layout(c, self.slots) for k, c in circuits.items()}

    def ids(self, e, n, dev):
        return (torch.full((n,), self.patches.index(e.patch), dtype=torch.long, device=dev),
                torch.full((n,), int(e.basis == "X"), dtype=torch.long, device=dev))


def log_row(rows, **kw):
    rows.append({k: kw.get(k, "") for k in LOG_FIELDS})
    print("  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in kw.items()), flush=True)


def write_log(rows):
    path = RES / "m3_training.csv"
    old = []
    if path.exists():
        with path.open() as f:
            old = [r for r in csv.DictReader(f) if r["stage"] != rows[0]["stage"]]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        w.writeheader()
        w.writerows(old + rows)


def cosine(step, total, warmup=500):
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


def hardware_sets(data):
    """Fine-tuning shots per experiment: (fit x, fit y, val x, val y); plus the fit mask over all shots."""
    sets, masks = {}, {}
    for e in data.exps:
        tr = split_indices(e)["train"]
        n_fit, n_val = round(FIT * tr.size), round(VAL * tr.size)
        fit, val = tr[:n_fit], tr[n_fit:n_fit + n_val]
        x = data.layout[e.key].sequence(e.detection_events(packed=True), packed=True)
        y = e.observable_flips()
        sets[e.key] = (x[fit], y[fit], x[val], y[val])
        m = np.zeros(e.shots, dtype=bool)
        m[tr[:n_fit + n_val]] = True
        masks[e.key] = m
    return sets, masks


@torch.no_grad()
def evaluate(model, data, sets, dev, batch=4096):
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    tot, err, n = 0.0, 0, 0
    for e in data.exps:
        _, _, xv, yv = sets[e.key]
        for i in range(0, len(yv), batch):
            x = torch.from_numpy(xv[i:i + batch]).float().to(dev)
            y = torch.from_numpy(yv[i:i + batch]).float().to(dev)
            logit = model(x, *data.ids(e, len(y), dev))
            tot += float(loss_fn(logit, y))
            err += int(((logit > 0) != (y > 0.5)).sum())
            n += len(y)
    model.train()
    return tot / n, err / n


def pretrain(data, args, dev):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = GRUDecoder(data.slots, data.patches, hidden=args.hidden).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine(s, args.steps))
    samplers = {e.key: e.dem(PRIOR).compile_sampler(seed=int(rng.integers(2 ** 31))) for e in data.exps}
    loss_fn = nn.BCEWithLogitsLoss()
    sets, _ = hardware_sets(data)                       # only the validation part is used here, as a monitor
    rows, t0, run = [], time.time(), []
    for step in range(args.steps):
        e = data.exps[rng.integers(len(data.exps))]
        det, obs, _ = samplers[e.key].sample(args.batch)
        x = torch.from_numpy(data.layout[e.key].sequence(det)).float().to(dev)
        y = torch.from_numpy(obs[:, 0]).float().to(dev)
        loss = loss_fn(model(x, *data.ids(e, args.batch, dev)), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run.append(loss.detach())
        if (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            vl, ve = evaluate(model, data, sets, dev)
            log_row(rows, stage="pretrain", step=step + 1, shots_seen=(step + 1) * args.batch, loss=torch.stack(run).mean().item(),
                    val_loss=vl, val_error=ve, seconds=round(time.time() - t0, 1))
            run = []
    CKPT.mkdir(parents=True, exist_ok=True)
    model.save(CKPT / "pretrained.pt", stage="pretrain", steps=args.steps)
    write_log(rows)


def finetune(data, args, dev):
    torch.manual_seed(SEED + 1)
    rng = np.random.default_rng(SEED + 1)
    model = GRUDecoder.load(CKPT / "pretrained.pt").to(dev)
    sets, _ = hardware_sets(data)
    batches = [(e, i) for e in data.exps for i in range(0, len(sets[e.key][1]), args.batch)]
    total = args.epochs * len(batches)
    opt = torch.optim.AdamW(model.parameters(), lr=args.ft_lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine(s, total, warmup=200))
    loss_fn = nn.BCEWithLogitsLoss()
    vl, ve = evaluate(model, data, sets, dev)
    rows, t0 = [], time.time()
    log_row(rows, stage="finetune", epoch=0, step=0, val_loss=vl, val_error=ve, seconds=0.0)
    best = (vl, 0)
    model.save(CKPT / "finetuned.pt", stage="finetune", epoch=0)
    step = 0
    for epoch in range(1, args.epochs + 1):
        perm = {e.key: rng.permutation(len(sets[e.key][1])) for e in data.exps}
        run = []
        for j in rng.permutation(len(batches)):
            e, i = batches[j]
            xf, yf, _, _ = sets[e.key]
            sel = perm[e.key][i:i + args.batch]
            x = torch.from_numpy(xf[sel]).float().to(dev)
            y = torch.from_numpy(yf[sel]).float().to(dev)
            loss = loss_fn(model(x, *data.ids(e, len(sel), dev)), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run.append(loss.detach())
            step += 1
        vl, ve = evaluate(model, data, sets, dev)
        log_row(rows, stage="finetune", epoch=epoch, step=step, shots_seen=step * args.batch, loss=torch.stack(run).mean().item(),
                val_loss=vl, val_error=ve, seconds=round(time.time() - t0, 1))
        if vl < best[0]:
            best = (vl, epoch)
            model.save(CKPT / "finetuned.pt", stage="finetune", epoch=epoch)
    print(f"best epoch {best[1]} (validation loss {best[0]:.5f})")
    write_log(rows)


@torch.no_grad()
def infer(data, args, dev):
    model = GRUDecoder.load(CKPT / "finetuned.pt").to(dev).eval()
    _, masks = hardware_sets(data)
    out = []
    for e in data.exps:
        x_all = data.layout[e.key].sequence(e.detection_events(packed=True), packed=True)
        logits = []
        for i in range(0, e.shots, 4096):
            x = torch.from_numpy(x_all[i:i + 4096]).float().to(dev)
            logits.append(model(x, *data.ids(e, len(x), dev)).float().cpu().numpy())
        logit = np.concatenate(logits)
        pred = logit > 0
        save(METHOD, PRIOR, e.key, pred, logit=logit.astype(np.float32),
             fit_mask=np.packbits(masks[e.key], bitorder="little"))
        obs = e.observable_flips()
        te = split_indices(e)["test"]
        row = {"key": e.key, "distance": e.distance, "patch": e.patch, "basis": e.basis, "rounds": e.rounds,
               "test_n": te.size, "nn_test_errors": int((pred[te] != obs[te]).sum())}
        if cache_path("mwpm_gap", PRIOR, e.key).exists():
            row["mwpm_test_errors"] = int((load("mwpm_gap", PRIOR, e.key)["pred"][te] != obs[te]).sum())
        out.append(row)
        print(f"{e.key}: Test errors nn {row['nn_test_errors']}, mwpm {row.get('mwpm_test_errors', '-')}", flush=True)
    with (RES / "m3_learned.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["key", "distance", "patch", "basis", "rounds", "test_n", "nn_test_errors",
                                          "mwpm_test_errors"])
        w.writeheader()
        w.writerows(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("pretrain", "finetune", "infer", "all"))
    ap.add_argument("--device")
    ap.add_argument("--steps", type=int, default=20_000, help="pre-training steps")
    ap.add_argument("--epochs", type=int, default=4, help="fine-tuning epochs")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ft-lr", type=float, default=2e-4)
    ap.add_argument("--log-every", type=int, default=1000)
    args = ap.parse_args()
    dev = device(args.device)
    print(f"device {dev}", flush=True)
    data = Data()
    stages = ("pretrain", "finetune", "infer") if args.stage == "all" else (args.stage,)
    for s in stages:
        t0 = time.time()
        {"pretrain": pretrain, "finetune": finetune, "infer": infer}[s](data, args, dev)
        print(f"{s} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
