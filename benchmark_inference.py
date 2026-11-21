"""Validate inference views on FER-2013 PublicTest and score once on PrivateTest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import precision_recall_fscore_support
import torch
from torch.nn import functional as F

from model import EMOTIONS, IMAGE_SIZE, load_checkpoint


VIEW_MODES = {
    "original": ("original",),
    "mirrored": ("original", "flip"),
    "mirror_and_zoom": ("original", "flip", "zoom_pair"),
}


@torch.inference_mode()
def collect_logits(model, partition, batch_size, views, device):
    pixels = np.stack([
        np.fromstring(row, sep=" ", dtype=np.uint8) for row in partition["pixels"]
    ]).reshape(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
    logits = {view: [] for view in views}
    for number, batch in enumerate(torch.from_numpy(pixels).split(batch_size), 1):
        images = batch.to(device, dtype=torch.float32).div_(127.5).sub_(1)
        for view in views:
            if view == "zoom_pair":
                zoom = F.interpolate(images[:, :, 1:-1, 1:-1], size=images.shape[-2:],
                                     mode="bilinear", align_corners=False)
                zoomed = torch.cat((zoom, zoom.flip((-1,))))
                output = model(zoomed).reshape(2, len(images), -1).mean(dim=0)
            else:
                output = model(images.flip((-1,)) if view == "flip" else images)
            logits[view].append(output.cpu())
        if number % 8 == 0:
            print(f"Scored {min(number * batch_size, len(pixels)):,} of {len(pixels):,} images", flush=True)
    return partition["emotion"].to_numpy(copy=True), {
        view: torch.cat(batches) for view, batches in logits.items()
    }


def combined(logits, mode):
    if mode == "mirror_and_zoom":
        return (logits["original"] + logits["flip"] + 2 * logits["zoom_pair"]) / 4
    return torch.stack([logits[view] for view in VIEW_MODES[mode]]).mean(0)


def measure(labels, logits, temperature=1.0):
    log_probabilities = (logits / temperature).log_softmax(dim=1)
    confidence, top = log_probabilities.exp().max(dim=1)
    confidence, top = confidence.numpy(), top.numpy()
    correct = top == labels
    bins = np.minimum((15 * confidence).astype(int), 14)
    calibration_error = 0.0
    for bucket in range(15):
        in_bucket = bins == bucket
        if in_bucket.any():
            calibration_error += in_bucket.mean() * abs(
                correct[in_bucket].mean() - confidence[in_bucket].mean())
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, top, labels=range(len(EMOTIONS)), zero_division=0)
    return {
        "accuracy": float(correct.mean()),
        "macro_f1": float(f1.mean()),
        "nll": float(-log_probabilities[range(len(labels)), labels].mean()),
        "ece15": float(calibration_error),
        "per_class": {
            name: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(n)}
            for name, p, r, f, n in zip(EMOTIONS, precision, recall, f1, support)
        },
    }


def improves(candidate, baseline):
    return candidate["accuracy"] > baseline["accuracy"] and candidate["macro_f1"] > baseline["macro_f1"]


def select_mode(results):
    baseline = results["original"]
    improved = [mode for mode in ("mirrored", "mirror_and_zoom") if improves(results[mode], baseline)]
    if not improved:
        return "original"
    winner = max(improved, key=lambda mode: results[mode]["macro_f1"])
    if winner == "mirror_and_zoom" and "mirrored" in improved:
        slower = results[winner]
        faster = results["mirrored"]
        if slower["macro_f1"] - faster["macro_f1"] < 0.003 and slower["accuracy"] - faster["accuracy"] < 0.003:
            return "mirrored"
    return winner


def fit_temperature(labels, logits):
    targets = torch.from_numpy(labels)
    fitted = minimize_scalar(
        lambda temperature: F.cross_entropy(logits / temperature, targets).item(),
        bounds=(0.25, 10.0), method="bounded")
    return float(fitted.x)


@torch.inference_mode()
def measure_cpu_latency(model):
    model.to("cpu")
    sample = torch.rand(1, 1, IMAGE_SIZE, IMAGE_SIZE)
    two_views = torch.cat((sample, sample.flip((-1,))))
    results = {}
    for threads in (1, 4):
        torch.set_num_threads(threads)
        for label, batch in (("original", sample), ("mirrored", two_views)):
            for _ in range(5):
                model(batch)
            timings = []
            for _ in range(24):
                started = perf_counter()
                model(batch)
                timings.append(perf_counter() - started)
            results[f"{threads}_{label}"] = float(1000 * np.median(timings))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/fer2013.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/best_model.pt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=96)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if args.output.resolve() in (args.csv.resolve(), args.checkpoint.resolve()):
        parser.error("output must be a separate file from the CSV and checkpoint")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable with the current PyTorch runtime")

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    model, metadata = load_checkpoint(args.checkpoint, device)
    columns = ["emotion", "pixels", "Usage"]
    frames = []
    for chunk in pd.read_csv(args.csv, usecols=columns, chunksize=4000):
        frames.append(chunk.loc[chunk["Usage"].isin(("PublicTest", "PrivateTest"))])
    frame = pd.concat(frames, ignore_index=True)
    validation = frame[frame["Usage"].eq("PublicTest")]
    test = frame[frame["Usage"].eq("PrivateTest")]
    if validation.empty or test.empty:
        parser.error("the CSV must include nonempty PublicTest and PrivateTest partitions")

    print("Evaluating the three view modes on PublicTest.", flush=True)
    y, validation_logits = collect_logits(model, validation, args.batch_size,
                                          VIEW_MODES["mirror_and_zoom"], device)
    validation_results = {mode: measure(y, combined(validation_logits, mode)) for mode in VIEW_MODES}
    chosen = select_mode(validation_results)
    temperature = fit_temperature(y, combined(validation_logits, chosen))
    fast_temperature = fit_temperature(y, validation_logits["original"])
    print(f"Frozen selection: {chosen}, with temperature {temperature:.4f}.", flush=True)

    print("Evaluating the baseline and frozen selection on PrivateTest.", flush=True)
    y, test_logits = collect_logits(model, test, args.batch_size, VIEW_MODES[chosen], device)
    test_results = {
        "baseline": measure(y, test_logits["original"]),
        "calibrated_single_view": measure(y, test_logits["original"], fast_temperature),
        "selected": measure(y, combined(test_logits, chosen), temperature),
    }
    report = {
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "checkpoint_epoch": metadata.get("epoch"),
        "validation_samples": len(validation),
        "test_samples": len(test),
        "selected": chosen,
        "temperature": temperature,
        "fast_temperature": fast_temperature,
        "validation": validation_results,
        "test": test_results,
        "median_cpu_inference_ms": measure_cpu_latency(model),
    }
    if args.output.name == "inference_profile.json":
        baseline = test_results["baseline"]
        selected = test_results["selected"]
        if chosen != "mirrored" or not improves(selected, baseline):
            parser.error("the selected mode did not meet the runtime profile promotion criteria")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for label, result in test_results.items():
        print(f"{label}: accuracy {result['accuracy']:.2%}, macro-F1 {result['macro_f1']:.4f}, "
              f"ECE {result['ece15']:.2%}")


if __name__ == "__main__":
    main()
