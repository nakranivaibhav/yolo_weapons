import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path

import albumentations as A

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

DEFAULT_DATASET_ROOT = Path("/workspace/yolo_dataset_4_dec")
DEFAULT_TRAIN_IMAGES = DEFAULT_DATASET_ROOT / "images" / "train"
DEFAULT_VAL_IMAGES = DEFAULT_DATASET_ROOT / "images" / "valid"
DEFAULT_TEST_IMAGES = DEFAULT_DATASET_ROOT / "images" / "test"
DEFAULT_BASE_YAML = DEFAULT_DATASET_ROOT / "data.yaml"

EVAL_EPOCHS = (50, 100)
METRIC_KEYS = ("precision", "recall", "f1", "map50", "map50_95")


def normalize_path(path_str: str) -> str:
    return str(Path(path_str).expanduser().resolve(strict=False))


def list_images(image_dir: Path) -> list[str]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    images = [
        normalize_path(str(p))
        for p in image_dir.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS
    ]
    return sorted(images)


def load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required: pip install pyyaml") from exc
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML format in {path}")
    return data


def save_yaml(data: dict, path: Path) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required: pip install pyyaml") from exc
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def compute_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def read_train_list(path: Path) -> list[str]:
    return [
        normalize_path(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_augmentations():
    return [
        A.OneOf(
            [
                A.MotionBlur(blur_limit=(7, 25), p=1.0),
                A.Defocus(radius=(3, 7), p=1.0),
            ],
            p=0.4,
        ),
        A.OneOf(
            [
                A.GaussNoise(std_range=(0.03, 0.2), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            ],
            p=0.3,
        ),
        A.ImageCompression(quality_range=(40, 90), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.4),
        A.Downscale(scale_range=(0.4, 0.85), p=0.2),
        A.RandomShadow(num_shadows_limit=(1, 2), shadow_roi=(0, 0.5, 1, 1), p=0.2),
    ]


def build_data_yaml(
    base_yaml_path: Path | None,
    dataset_root: Path,
    train_list_path: Path,
    val_images_dir: Path,
    test_images_dir: Path,
) -> dict:
    data_cfg: dict = {}
    if base_yaml_path and base_yaml_path.exists():
        base_cfg = load_yaml(base_yaml_path)
        for key in ("names", "nc"):
            if key in base_cfg:
                data_cfg[key] = base_cfg[key]
    data_cfg["path"] = str(dataset_root)
    data_cfg["train"] = str(train_list_path)
    data_cfg["val"] = str(val_images_dir)
    data_cfg["test"] = str(test_images_dir)
    return data_cfg


def write_metrics(metrics_path: Path, metrics: dict) -> None:
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Monte Carlo influence for YOLOv11 with fixed F1 evaluation."
    )
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--train-images-dir", default=str(DEFAULT_TRAIN_IMAGES))
    parser.add_argument("--val-images-dir", default=str(DEFAULT_VAL_IMAGES))
    parser.add_argument("--test-images-dir", default=str(DEFAULT_TEST_IMAGES))
    parser.add_argument("--base-data-yaml", default=str(DEFAULT_BASE_YAML))
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--subset-size", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out-dir", default="monte_carlo_influence/runs")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--skip-existing", action="store_true")

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    train_images_dir = Path(args.train_images_dir).expanduser().resolve()
    val_images_dir = Path(args.val_images_dir).expanduser().resolve()
    test_images_dir = Path(args.test_images_dir).expanduser().resolve()
    base_yaml_path = Path(args.base_data_yaml).expanduser().resolve()

    if not val_images_dir.exists():
        val_images_dir = test_images_dir

    if args.subset_size <= 0:
        print("--subset-size must be positive.", file=sys.stderr)
        return 2
    if args.runs <= 0:
        print("--runs must be positive.", file=sys.stderr)
        return 2
    if args.epochs <= 0:
        print("--epochs must be positive.", file=sys.stderr)
        return 2

    all_images = list_images(train_images_dir)
    if len(all_images) < args.subset_size:
        print(
            f"Not enough training images ({len(all_images)}) for subset size {args.subset_size}.",
            file=sys.stderr,
        )
        return 2

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics is required: pip install ultralytics") from exc

    augments = build_augmentations()

    n_total = len(all_images)
    path_to_index = {p: i for i, p in enumerate(all_images)}

    inc_sum = {
        epoch: {metric: [0.0] * n_total for metric in METRIC_KEYS}
        for epoch in EVAL_EPOCHS
    }
    inc_count = {epoch: [0] * n_total for epoch in EVAL_EPOCHS}
    total_sum = {epoch: {metric: 0.0 for metric in METRIC_KEYS} for epoch in EVAL_EPOCHS}
    total_runs = {epoch: 0 for epoch in EVAL_EPOCHS}
    run_rows = {epoch: [] for epoch in EVAL_EPOCHS}

    for run_idx in range(args.runs):
        run_name = f"mc_{run_idx:03d}"
        run_dir = out_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        train_list_path = run_dir / "train.txt"
        run_data_yaml = run_dir / "data.yaml"

        if args.skip_existing:
            has_metrics = all(
                (run_dir / f"epoch_{epoch:03d}" / "metrics.json").exists()
                for epoch in EVAL_EPOCHS
            )
            has_train_list = train_list_path.exists()
            use_existing = has_metrics and has_train_list
        else:
            use_existing = False

        run_seed = args.seed + run_idx
        if use_existing:
            subset_paths = read_train_list(train_list_path)
        else:
            rng = random.Random(run_seed)
            subset_paths = rng.sample(all_images, args.subset_size)
            train_list_path.write_text("\n".join(subset_paths) + "\n", encoding="utf-8")

        run_data_cfg = build_data_yaml(
            base_yaml_path if base_yaml_path.exists() else None,
            dataset_root,
            train_list_path,
            val_images_dir,
            test_images_dir,
        )
        save_yaml(run_data_cfg, run_data_yaml)

        def save_epoch_checkpoint(trainer):
            epoch_num = trainer.epoch + 1
            if epoch_num in EVAL_EPOCHS:
                weights_dir = Path(trainer.save_dir) / "weights"
                src = weights_dir / "last.pt"
                if src.exists():
                    dst = weights_dir / f"epoch_{epoch_num:03d}.pt"
                    shutil.copy2(src, dst)

        model = None
        if not use_existing:
            model = YOLO(args.model)
            model.add_callback("on_fit_epoch_end", save_epoch_checkpoint)

        if not use_existing:
            results = model.train(
                data=str(run_data_yaml),
                epochs=args.epochs,
                patience=100,
                batch=args.batch,
                imgsz=args.imgsz,
                augmentations=augments,
                hsv_h=0.015,
                hsv_s=0.4,
                hsv_v=0.4,
                degrees=15.0,
                translate=0.1,
                scale=0.4,
                shear=3.0,
                perspective=0.0002,
                fliplr=0.5,
                flipud=0.0,
                mosaic=0.3,
                mixup=0.1,
                copy_paste=0.0,
                close_mosaic=15,
                label_smoothing=0.1,
                optimizer="SGD",
                momentum=0.937,
                lr0=0.01,
                lrf=0.01,
                weight_decay=0.0005,
                warmup_epochs=3.0,
                warmup_momentum=0.8,
                warmup_bias_lr=0.1,
                box=7.5,
                cls=0.5,
                dfl=1.5,
                iou=0.7,
                max_det=300,
                val=False,
                project=str(out_dir),
                name=run_name,
                device=args.device,
                workers=args.workers,
                plots=True,
                save=True,
            )
            _ = results

        weights_dir = run_dir / "weights"
        for epoch in EVAL_EPOCHS:
            eval_dir = run_dir / f"epoch_{epoch:03d}"
            metrics_path = eval_dir / "metrics.json"
            if args.skip_existing and metrics_path.exists():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            else:
                ckpt_path = weights_dir / f"epoch_{epoch:03d}.pt"
                if not ckpt_path.exists():
                    print(f"Missing checkpoint for epoch {epoch}: {ckpt_path}")
                    continue

                eval_model = YOLO(str(ckpt_path))
                eval_results = eval_model.val(
                    data=str(run_data_yaml),
                    split="test",
                    conf=args.conf,
                    iou=args.iou,
                    batch=args.batch,
                    imgsz=args.imgsz,
                    device=args.device,
                    plots=False,
                    save_json=False,
                    project=str(run_dir),
                    name=f"epoch_{epoch:03d}",
                    exist_ok=True,
                )

                precision = float(
                    eval_results.results_dict.get("metrics/precision(B)", 0.0)
                )
                recall = float(
                    eval_results.results_dict.get("metrics/recall(B)", 0.0)
                )
                map50 = float(eval_results.results_dict.get("metrics/mAP50(B)", 0.0))
                map50_95 = float(
                    eval_results.results_dict.get("metrics/mAP50-95(B)", 0.0)
                )
                f1 = compute_f1(precision, recall)

                metrics = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "map50": map50,
                    "map50_95": map50_95,
                    "conf": args.conf,
                    "iou": args.iou,
                    "subset_size": args.subset_size,
                    "run": run_name,
                    "seed": run_seed,
                    "epoch": epoch,
                }
                eval_dir.mkdir(parents=True, exist_ok=True)
                write_metrics(metrics_path, metrics)

            total_runs[epoch] += 1
            for metric in METRIC_KEYS:
                total_sum[epoch][metric] += float(metrics.get(metric, 0.0))
            for img_path in subset_paths:
                idx = path_to_index.get(img_path)
                if idx is None:
                    continue
                inc_count[epoch][idx] += 1
                for metric in METRIC_KEYS:
                    inc_sum[epoch][metric][idx] += float(metrics.get(metric, 0.0))

            run_rows[epoch].append(
                {
                    "run": run_name,
                    "epoch": epoch,
                    "precision": float(metrics.get("precision", 0.0)),
                    "recall": float(metrics.get("recall", 0.0)),
                    "f1": float(metrics.get("f1", 0.0)),
                    "map50": float(metrics.get("map50", 0.0)),
                    "map50_95": float(metrics.get("map50_95", 0.0)),
                    "subset_size": args.subset_size,
                }
            )

    for epoch in EVAL_EPOCHS:
        summary_path = out_dir / f"run_summary_epoch_{epoch:03d}.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "run",
                    "epoch",
                    "precision",
                    "recall",
                    "f1",
                    "map50",
                    "map50_95",
                    "subset_size",
                ],
            )
            writer.writeheader()
            writer.writerows(run_rows[epoch])

        influence_path = out_dir / f"influence_epoch_{epoch:03d}.csv"
        with influence_path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "image_path",
                "included_count",
                "excluded_count",
                "mean_included_precision",
                "mean_excluded_precision",
                "influence_precision",
                "mean_included_recall",
                "mean_excluded_recall",
                "influence_recall",
                "mean_included_map50",
                "mean_excluded_map50",
                "influence_map50",
                "mean_included_map50_95",
                "mean_excluded_map50_95",
                "influence_map50_95",
                "mean_included_f1",
                "mean_excluded_f1",
                "influence_f1",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for img_path in all_images:
                idx = path_to_index[img_path]
                inc_c = inc_count[epoch][idx]
                exc_c = total_runs[epoch] - inc_c

                row = {
                    "image_path": img_path,
                    "included_count": inc_c,
                    "excluded_count": exc_c,
                }

                for metric in METRIC_KEYS:
                    inc_s = inc_sum[epoch][metric][idx]
                    if inc_c > 0:
                        mean_inc = inc_s / inc_c
                    else:
                        mean_inc = ""
                    if exc_c > 0:
                        mean_exc = (total_sum[epoch][metric] - inc_s) / exc_c
                    else:
                        mean_exc = ""
                    if inc_c > 0 and exc_c > 0:
                        influence = mean_inc - mean_exc
                    else:
                        influence = ""

                    row[f"mean_included_{metric}"] = mean_inc
                    row[f"mean_excluded_{metric}"] = mean_exc
                    row[f"influence_{metric}"] = influence

                writer.writerow(row)

        print(f"Epoch {epoch} summary: {summary_path}")
        print(f"Epoch {epoch} influence: {influence_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
