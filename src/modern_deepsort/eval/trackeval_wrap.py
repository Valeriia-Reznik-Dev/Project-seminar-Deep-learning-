"""HOTA scoring via TrackEval (the official MOTChallenge protocol).

We score the six test videos as their native benchmarks so that TrackEval
applies the correct preprocessing rules (MOT16 removes distractor classes /
honors ignore regions; MOT15 has none):

    MOT15 group: TUD-Campus, TUD-Stadtmitte, KITTI-17, PETS09-S2L1
    MOT16 group: MOT16-09, MOT16-11

For each group we build the layout TrackEval expects, call its Python API,
and read HOTA from the returned results dict. The final number reported by
the project is HOTA **averaged across all six videos**.

Requires TrackEval to be importable (see scripts/setup_trackeval.sh).

Layout built under <work_dir>:
    gt/<BENCH>-train/<seq>/gt/gt.txt
    gt/<BENCH>-train/<seq>/seqinfo.ini
    gt/seqmaps/<BENCH>-train.txt
    trackers/<BENCH>-train/<tracker>/data/<seq>.txt
"""
from __future__ import annotations

import os
import shutil

GROUPS = {
    "MOT15": ["TUD-Campus", "TUD-Stadtmitte", "KITTI-17", "PETS09-S2L1"],
    "MOT16": ["MOT16-09", "MOT16-11"],
}


def _stage_group(bench, seqs, mot_root, results_dir, work_dir, tracker):
    split = f"{bench}-train"
    gt_base = os.path.join(work_dir, "gt", split)
    tr_base = os.path.join(work_dir, "trackers", split, tracker, "data")
    seqmap_dir = os.path.join(work_dir, "gt", "seqmaps")
    os.makedirs(seqmap_dir, exist_ok=True)
    os.makedirs(tr_base, exist_ok=True)

    present = []
    for seq in seqs:
        src = os.path.join(mot_root, seq)
        res = os.path.join(results_dir, f"{seq}.txt")
        if not (os.path.isdir(src) and os.path.isfile(res)):
            continue
        present.append(seq)
        dst = os.path.join(gt_base, seq)
        os.makedirs(os.path.join(dst, "gt"), exist_ok=True)
        shutil.copyfile(os.path.join(src, "gt", "gt.txt"),
                        os.path.join(dst, "gt", "gt.txt"))
        shutil.copyfile(os.path.join(src, "seqinfo.ini"),
                        os.path.join(dst, "seqinfo.ini"))
        shutil.copyfile(res, os.path.join(tr_base, f"{seq}.txt"))

    # seqmap
    with open(os.path.join(seqmap_dir, f"{split}.txt"), "w") as fh:
        fh.write("name\n")
        for seq in present:
            fh.write(seq + "\n")
    return split, present


def score_hota(mot_root, results_dir, work_dir, tracker="modern_deepsort",
               benchmarks=("MOT15", "MOT16")):
    """Run TrackEval per benchmark group; return per-seq + averaged HOTA."""
    import trackeval  # imported here so the module loads without TrackEval

    if os.path.isdir(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    per_seq = {}
    for bench in benchmarks:
        seqs = GROUPS[bench]
        split, present = _stage_group(bench, seqs, mot_root, results_dir,
                                      work_dir, tracker)
        if not present:
            continue

        eval_config = trackeval.Evaluator.get_default_eval_config()
        eval_config.update({"USE_PARALLEL": False, "PRINT_CONFIG": False,
                            "PRINT_RESULTS": False, "OUTPUT_SUMMARY": False,
                            "OUTPUT_DETAILED": False, "PLOT_CURVES": False})
        ds_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        ds_config.update({
            "GT_FOLDER": os.path.join(work_dir, "gt"),
            "TRACKERS_FOLDER": os.path.join(work_dir, "trackers"),
            "BENCHMARK": bench,
            "SPLIT_TO_EVAL": "train",
            "TRACKERS_TO_EVAL": [tracker],
            "DO_PREPROC": bench != "MOT15",  # MOT15 gt has no distractor classes
            "SEQMAP_FILE": os.path.join(work_dir, "gt", "seqmaps", f"{split}.txt"),
        })
        evaluator = trackeval.Evaluator(eval_config)
        dataset = trackeval.datasets.MotChallenge2DBox(ds_config)
        res, _ = evaluator.evaluate([dataset], [trackeval.metrics.HOTA()])
        seq_res = res["MotChallenge2DBox"][tracker]
        for seq in present:
            hota = seq_res[seq]["pedestrian"]["HOTA"]["HOTA"]
            per_seq[seq] = float(hota.mean())  # HOTA averaged over alpha thresholds

    avg = sum(per_seq.values()) / len(per_seq) if per_seq else float("nan")
    return {"per_sequence": per_seq, "average_HOTA": avg}
