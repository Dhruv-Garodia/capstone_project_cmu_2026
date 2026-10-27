"""Supervised experiment pipeline for the final dataset (runs unattended until the deadline).

    caffeinate -i python eponge/scripts/pipeline.py --deadline "2026-10-25 12:00"

* Jobs run one at a time in priority order; a job whose output already exists is skipped (restartable).
* Watchdog: a job is killed if its log has not changed for --stall minutes or it exceeds its time limit;
  it is retried once, then skipped, so one failure never blocks the rest.
* After the core runs, the proposed model is chosen by mean validation mIoU (never test) among the new
  architectures; its seeds, ablations, cross-stack test, twin pretraining and public benchmark follow.
* After --deadline no new training starts; the final report always runs.
* Status: ../runs/pipeline_status.json (and pipeline.log).
"""
import argparse
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT.parent / "runs"
FM = RUNS / "final_models"
PY = str(ROOT.parent / ".venv/bin/python")
NN = str(ROOT.parent / ".venv/bin")
STATUS = RUNS / "pipeline_status.json"
NN_ENV = {"nnUNet_raw": str(ROOT.parent / "nnunet/raw"), "nnUNet_preprocessed": str(ROOT.parent / "nnunet/preprocessed"),
          "nnUNet_results": str(ROOT.parent / "nnunet/results"), "nnUNet_n_proc_DA": "2", "PYTHONUNBUFFERED": "1"}
CANDIDATES = {   # new architectures; the best by validation becomes "the proposed model"
    "millnet_sa2": "--arch millnet --k 3 --learned-slide --slide-max 1.5 --slide-reg 0.05",
    "zafunet_r34": "--arch zafunet --encoder resnet34 --k 3",
    "millnet_sa": "--arch millnet --k 3 --learned-slide",
}
ABLATIONS = {
    "zafunet_r34": {"meanfusion": "--fusion mean", "centreonly": "--fusion centre", "nopretrain": "--no-pretrain",
                    "k1": None, "k2": None},
    "millnet_sa2": {"noslide": None, "nobank": "--no-bank", "nostats": "--no-stats", "noaxial": "--no-axial",
                    "noslidereg": None},
    "millnet_sa": {"noslide": None, "nobank": "--no-bank", "nostats": "--no-stats", "noaxial": "--no-axial"},
}
LONG = "--crop 512 --bs 6 --iters 7500"     # same training budget as nnU-Net 2D (30 x 250 iterations, batch 6, 512 px)
STATE = {"started": dt.datetime.now().isoformat(timespec="seconds"), "jobs": []}


def log(msg):
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(RUNS / "pipeline.log", "a") as f:
        f.write(line + "\n")


def save():
    STATUS.write_text(json.dumps(STATE, indent=1))


def train_job(name, args, timeout_h=3.0):
    out = FM / name
    return {"name": name, "cmd": [PY, "eponge/scripts/train_final.py", "--out", str(out)] + args.split(),
            "done": out / "summary.json", "log": FM / f"{name}.log", "timeout": timeout_h * 3600, "kind": "train"}


def group_rss_gb(pgid):
    """Resident memory of a job's whole process group (job + its workers), GB."""
    try:
        out = subprocess.run(["ps", "-g", str(pgid), "-o", "rss="], capture_output=True, text=True).stdout
        return sum(int(x) for x in out.split()) / 1048576
    except Exception:
        return 0.0


def run(job, stall_min, mem_gb=11.0):
    if job["done"].exists():
        STATE["jobs"].append({"name": job["name"], "state": "skipped (done)"}); save(); return True
    for attempt in (1, 2):
        rec = {"name": job["name"], "state": "running", "attempt": attempt, "start": dt.datetime.now().isoformat(timespec="seconds")}
        STATE["jobs"].append(rec); save()
        log(f"START {job['name']} (attempt {attempt})")
        job["log"].parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "PYTHONUNBUFFERED": "1", **job.get("env", {})}
        with open(job["log"], "a") as lf:
            p = subprocess.Popen(job["cmd"], cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            t0 = time.time(); reason = None
            while p.poll() is None:
                time.sleep(30)
                age = time.time() - max(job["log"].stat().st_mtime, t0)          # a stale log from an earlier run does not count
                rss = group_rss_gb(p.pid)
                if time.time() - t0 > job["timeout"]:
                    reason = "timeout"
                elif rss > job.get("mem_gb", mem_gb):
                    reason = f"memory {rss:.1f} GB"
                elif age > stall_min * 60:
                    reason = f"stalled {age / 60:.0f} min"
                if reason:
                    os.killpg(p.pid, signal.SIGKILL); p.wait(); break
        rc = p.returncode
        ok = job["done"].exists() and rc == 0 if job.get("strict", True) else job["done"].exists()
        rec.update(end=dt.datetime.now().isoformat(timespec="seconds"), rc=rc, minutes=round((time.time() - t0) / 60, 1),
                   state="ok" if ok else f"failed ({reason or f'rc {rc}'})"); save()
        log(f"END {job['name']}: {rec['state']} in {rec['minutes']} min")
        if ok:
            return True
    return False


def nnunet_jobs(cfg, epochs, iters, hours, chk=None):
    pred = ROOT.parent / f"nnunet/pred_{cfg}"
    env = {**NN_ENV, "NNUNET_EPOCHS": str(epochs), "NNUNET_ITERS": str(iters), "NNUNET_VAL_ITERS": "20"}
    final = ROOT.parent / f"nnunet/results/Dataset501_PFIB/nnUNetTrainer_Pfib__nnUNetPlans__{cfg}/fold_0/{chk or 'checkpoint_final.pth'}"
    return [
        {"name": f"nnunet_{cfg}_train", "cmd": [f"{NN}/nnUNetv2_train", "501", cfg, "0", "-tr", "nnUNetTrainer_Pfib", "-device", "mps"],
         "done": final, "log": RUNS / f"nnunet_{cfg}_train.log", "timeout": hours * 3600, "env": env, "strict": False},
        {"name": f"nnunet_{cfg}_predict", "cmd": [f"{NN}/nnUNetv2_predict", "-i", str(ROOT.parent / "nnunet/raw/Dataset501_PFIB/imagesTs"),
                                                   "-o", str(pred), "-d", "501", "-c", cfg, "-f", "0", "-tr", "nnUNetTrainer_Pfib",
                                                   "-device", "mps", "-npp", "1", "-nps", "1"] + (["-chk", chk] if chk else []),
         "done": pred / "pristine_test_101.nii.gz", "log": RUNS / f"nnunet_{cfg}_predict.log", "timeout": 3 * 3600, "env": env, "strict": False},
        {"name": f"nnunet_{cfg}_eval", "cmd": [PY, "eponge/scripts/nnunet_baseline.py", "evaluate", "--config", cfg, "--pred", str(pred)],
         "done": FM / f"nnunet_{cfg}/summary.json", "log": RUNS / f"nnunet_{cfg}_eval.log", "timeout": 1800},
    ]


def best_val(name):
    s = FM / name / "summary.json"
    return json.loads(s.read_text())["best_val"] if s.exists() else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deadline", default="2026-10-25 12:00")
    ap.add_argument("--stall", type=float, default=45)
    a = ap.parse_args()
    deadline = dt.datetime.fromisoformat(a.deadline)
    before = lambda: dt.datetime.now() < deadline
    log(f"pipeline start; deadline for new training {deadline}")

    # Phase A: core comparison (seed 0) and the nnU-Net 2D baseline
    core = [train_job("zafunet_r34_k3_s0", CANDIDATES["zafunet_r34"] + " --seed 0", 3),
            train_job("millnet_sa2_k3_s0", CANDIDATES["millnet_sa2"] + " --seed 0"),
            train_job("unet_r34_k3_s0", "--arch unet_r34 --k 3 --seed 0")]
    for j in core:
        before() and run(j, a.stall)
    for j in nnunet_jobs("2d", 30, 250, 6):
        before() and run(j, a.stall)

    # choose the proposed model on validation only
    vals = {c: best_val(f"{c}_k3_s0") for c in CANDIDATES}
    proposed = max(vals, key=vals.get)
    STATE["proposed"] = {"model": proposed, "val_mIoU_seed0": vals}; save()
    (FM / "proposed.json").write_text(json.dumps(STATE["proposed"], indent=1))
    log(f"PROPOSED = {proposed} (validation mIoU {vals})")
    P = CANDIDATES[proposed]
    heavy = 3 if proposed == "zafunet_r34" else 1.5
    # proposed model at nnU-Net's training budget (seed 0 now, more seeds at the end if time allows)
    before() and run(train_job(f"{proposed}_long_k3_s0", f"{P} {LONG} --seed 0", 4.5), a.stall)
    # MillTrace (the stack read as a video, slide-warped recurrent memory): seed 0 at the standard budget,
    # two more seeds only if it beats the proposed model on validation
    MT = "--arch milltrace --k 7 --slide-max 1.5 --slide-reg 0.05"
    before() and run(train_job("milltrace_k7_s0", f"{MT} --seed 0", 3), a.stall)
    if best_val("milltrace_k7_s0") > best_val(f"{proposed}_k3_s0"):
        for s in (1, 2):
            before() and run(train_job(f"milltrace_k7_s{s}", f"{MT} --seed {s}", 3), a.stall)
    # MillTrace v2: slide-aligned full-resolution path (v1's slide field stayed at ~0); same rule for more seeds
    before() and run(train_job("milltrace2_k7_s0", f"{MT} --trace-v2 --seed 0", 3), a.stall)
    if best_val("milltrace2_k7_s0") > best_val(f"{proposed}_k3_s0"):
        for s in (1, 2):
            before() and run(train_job(f"milltrace2_k7_s{s}", f"{MT} --trace-v2 --seed {s}", 3), a.stall)

    # Phase B: three seeds for the proposed model and the main baselines
    jobs = [train_job(f"{proposed}_k3_s{s}", f"{P} --seed {s}", heavy) for s in (1, 2)]
    jobs += [train_job(f"{b}_k3_s{s}", f"{arg} --seed {s}") for s in (1, 2)
             for b, arg in (("unet_r34", "--arch unet_r34 --k 3"), ("millnet", "--arch millnet --k 3"), ("resunet", "--arch resunet --k 3"))]
    # Phase C: ablations of the proposed model
    for ab, extra in ABLATIONS[proposed].items():
        if ab in ("k1", "k2"):
            args = P.replace("--k 3", f"--k {ab[1]}")
        elif ab == "noslide":
            continue                                    # = millnet_k3_s0, already trained
        elif ab == "noslidereg":
            continue                                    # = millnet_sa_k3_s0, already trained
        else:
            args = f"{P} {extra}"
        jobs.append(train_job(f"abl_{proposed}_{ab}", args + " --seed 0", heavy))
    # Phase D: cross-stack generalisation (train on two stacks, test on the third)
    for held in ("comp", "uncomp", "pristine"):
        tr = ",".join(s for s in ("comp", "uncomp", "pristine") if s != held)
        for nm, arg, hh in ((proposed, P, heavy), ("unet_r34", "--arch unet_r34 --k 3", 1.5), ("millnet", "--arch millnet --k 3", 1.5)):
            jobs.append(train_job(f"loso_{held}_{nm}", f"{arg} --train-stacks {tr} --test-stacks {held}", hh))
    for j in jobs:
        before() and run(j, a.stall)

    # Phase E: SliceGAN digital twin -> pretrain -> fine-tune (proposed model)
    tw = ROOT / f"runs/twin_gan3/{proposed}_k3"
    twargs = {"zafunet_r34": "--arch zafunet --encoder resnet34", "millnet_sa2": "--arch millnet --learned-slide",
              "millnet_sa": "--arch millnet --learned-slide"}[proposed]
    if before():
        run({"name": "twin_pretrain", "cmd": [PY, "eponge/scripts/twin_pretrain.py"] + twargs.split() + ["--k", "3", "--out", str(tw)],
             "done": tw / "best.pt", "log": RUNS / "twin_pretrain.log", "timeout": 3 * 3600}, a.stall)
        if (tw / "best.pt").exists():
            run(train_job(f"{proposed}_k3_twin_s0", f"{P} --init {tw / 'best.pt'} --seed 0", heavy), a.stall)

    # Phase F: other baselines, nnU-Net 3D, public benchmark
    for j in [train_job("unetpp_r34_k3_s0", "--arch unetpp_r34 --k 3 --lr 4e-4 --seed 0"),
              train_job("segformer_k3_s0", "--arch segformer --k 3 --seed 0")]:
        before() and run(j, a.stall)
    for j in nnunet_jobs("3d_fullres", 30, 100, 7, chk="checkpoint_best.pth"):   # training stopped at epoch 26/30 (power loss)
        before() and run(j, a.stall)
    # Phase G first (headline error bars and the slide-field ablation), then the public benchmark
    for j in [train_job(f"{proposed}_long_k3_s1", f"{P} {LONG} --seed 1", 4.5),
              train_job("millnet_long_k3_s0", f"--arch millnet --k 3 {LONG} --seed 0", 4.5),
              train_job(f"{proposed}_long_k3_s2", f"{P} {LONG} --seed 2", 4.5)]:
        before() and run(j, a.stall)

    # Phase C2: what makes MillNet work (base MillNet, standard budget; MillNet 3 seeds = 0.915 +- 0.004)
    M = "--arch millnet --k 3"
    for nm, extra, seeds in (("raw", "--no-bank --no-milling", (0, 1, 2)), ("plainhead", "--plain-head", (0,)),
                             ("nostats", "--no-stats", (0,)),
                             ("minimal", "--no-bank --no-milling --no-stats --no-axial --plain-head", (0,))):
        for sd in seeds:
            before() and run(train_job(f"abl_millnet_{nm}_s{sd}", f"{M} {extra} --seed {sd}", 1.5), a.stall)

    # Phase H: few-shot adaptation to an unseen acquisition. Cross-stack models (trained without pristine)
    # are fine-tuned on N evenly spaced labelled pristine train slices; no validation-based selection.
    for n in (2, 5, 10):
        for nm, arg, init, lr in ((proposed, P, f"loso_pristine_{proposed}", 5e-4),
                                  ("unet_r34", "--arch unet_r34 --k 3", "loso_pristine_unet_r34", 1e-4),
                                  (f"{proposed}_scratch", P, None, None)):
            extra = f" --init {FM / init / 'best.pt'} --lr {lr}" if init else ""
            before() and run(train_job(f"fs_pristine_{n}_{nm}", f"{arg} --train-stacks pristine --test-stacks pristine "
                                                               f"--train-frames {n} --iters 600{extra} --seed 0", 1), a.stall)

    # Phase C3: does slice context matter? base MillNet with 1 slice (k=0) and 3 slices (k=1), standard budget
    for kk in (0, 1):
        before() and run(train_job(f"abl_millnet_k{kk}_s0", f"--arch millnet --k {kk} --seed 0", 1.5), a.stall)

    # Phase I: digital twin rebuilt per acquisition. SliceGAN on each acquisition's hand masks (train slices),
    # GAN-twin appearance (runs/twin_refiner_v2), then twin pretraining and fine-tuning of the proposed model.
    SG = ROOT / "eponge/scripts/slicegan_twin.py"
    for st in ("comp", "uncomp", "pristine"):
        sg = ROOT / f"runs/slicegan_{st}"
        before() and run({"name": f"slicegan_{st}", "cmd": ["/bin/zsh", "-c",
                          f"{PY} {SG} train --stack {st} --minutes 150 && {PY} {SG} generate --stack {st} --n 14 && "
                          f"{PY} {SG} validate --stack {st}"],
                          "done": sg / "validation.json", "log": RUNS / f"slicegan_{st}.log", "timeout": 3.5 * 3600}, a.stall)
    GANRUN = ROOT / "runs/twin_refiner_v2"
    # pristine is the hardest look for the GAN twin: fine-tune on pristine alone, keep it only if closer on validation
    GV3, TRF = ROOT / "runs/twin_refiner_v3_pristine", ROOT / "eponge/scripts/twin_refiner.py"
    before() and run({"name": "gan_v3_pristine", "cmd": ["/bin/zsh", "-c",
                      f"{PY} {TRF} train_v2 --iters 3000 --stacks pristine --init-from {GANRUN / 'G.pt'} --out {GV3} && "
                      f"{PY} {TRF} downstream --split val --stacks pristine --noise 1.0 --out {GV3} && "
                      f"mv {GV3}/downstream_millnet_sa2_long_k3_s0_val_noise1.0.json {GV3}/downstream_millnet_sa2_long_k3_s0_val.json && "
                      f"{PY} {ROOT / 'eponge/scripts/gan_select.py'} --base runs/twin_refiner_v2 --cand runs/twin_refiner_v3_pristine --stack pristine"],
                      "done": GANRUN / "per_stack.json", "log": RUNS / "gan_v3_pristine.log", "timeout": 4 * 3600}, a.stall)
    per_stack = all((ROOT / f"runs/slicegan_{st}/volumes.npz").exists() for st in ("comp", "uncomp", "pristine"))
    if per_stack and before():
        run({"name": "twin_export_gan", "cmd": [PY, "eponge/scripts/export_twin_dataset.py", "--gan", str(GANRUN),
                                                "--geometry", "per_stack"],
             "done": ROOT.parent / "twin_dataset/index.json", "log": RUNS / "twin_export_gan.log", "timeout": 3 * 3600}, a.stall)
        tg = ROOT / f"runs/twin_gan_ps/{proposed}_k3"
        run({"name": "twin_pretrain_gan", "cmd": [PY, "eponge/scripts/twin_pretrain.py"] + twargs.split() +
                     ["--k", "3", "--geometry", "per_stack", "--gan", str(GANRUN), "--out", str(tg)],
             "done": tg / "best.pt", "log": RUNS / "twin_pretrain_gan.log", "timeout": 3 * 3600}, a.stall)
        if (tg / "best.pt").exists():
            run(train_job(f"{proposed}_k3_gantwin_s0", f"{P} --init {tg / 'best.pt'} --seed 0", heavy), a.stall)

    # matched-budget pretrained U-Net (fairness), after the experiments above
    before() and run(train_job("unet_r34_long_k3_s0", f"--arch unet_r34 --k 3 {LONG} --seed 0", 4.5), a.stall)
    bench = {"zafunet_r34": "zafunet_7slice", "millnet_sa2": "millnet_sa_7slice", "millnet_sa": "millnet_sa_7slice"}[proposed]
    trial = ROOT / "runs/external_trial"
    ok = before() and run({"name": f"public_trial_{bench}", "cmd": [PY, "eponge/scripts/external_benchmark.py", "--only", bench,
                                                               "--iters", "60", "--out", str(trial)],
                           "done": trial / f"{bench}.pt", "log": RUNS / f"public_trial_{bench}.log", "timeout": 1800, "mem_gb": 8}, 20)
    for s in ((0, 1, 2) if ok else ()):
        tag = bench + (f"_s{s}" if s else "")
        before() and run({"name": f"public_{tag}", "cmd": [PY, "eponge/scripts/external_benchmark.py", "--only", bench, "--seed", str(s),
                                                           "--out", str(ROOT / "runs/external_new")],
                          "done": ROOT / f"runs/external_new/{tag}.pt", "log": RUNS / f"public_{tag}.log", "timeout": 2 * 3600}, a.stall)

    # Phase G: matched-budget pretrained U-Net, then more seeds of the matched-budget proposed model
    for j in [train_job("millnet_long_k3_s0", f"--arch millnet --k 3 {LONG} --seed 0", 4.5),       # slide-field ablation at matched budget
              train_job("unet_r34_long_k3_s0", f"--arch unet_r34 --k 3 {LONG} --seed 0", 4.5)] + \
             [train_job(f"{proposed}_long_k3_s{s}", f"{P} {LONG} --seed {s}", 4.5) for s in (1, 2)]:
        before() and run(j, a.stall)

    # Always: twin dataset export and the final report
    run({"name": "twin_export", "cmd": [PY, "eponge/scripts/export_twin_dataset.py"], "done": ROOT.parent / "twin_dataset/index.json",
         "log": RUNS / "twin_export.log", "timeout": 3600}, a.stall)
    rep_extra = (["--twin-gan", str(ROOT / "runs/twin_refiner_v2"), "--twin-geometry", "per_stack"]
                 if all((ROOT / f"runs/slicegan_{st}/volumes.npz").exists() for st in ("comp", "uncomp", "pristine")) else [])
    run({"name": "report", "cmd": [PY, "eponge/scripts/report_final.py", "--ref", "nnunet_2d"] + rep_extra,
         "done": FM / "report/results.md", "log": RUNS / "report_final.log", "timeout": 8 * 3600}, 90)
    STATE["finished"] = dt.datetime.now().isoformat(timespec="seconds"); save()
    log("PIPELINE DONE")


if __name__ == "__main__":
    main()
