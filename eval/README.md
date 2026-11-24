# Shared DocLayNet evaluator

Model-neutral tooling for canonical YOLO26s and RT-DETRv4-S prediction files.

## Prediction contract

A JSON list of records with exactly `image_id`, `category_id`, `bbox` (absolute original-pixel COCO `xywh`), and `score`. Canonical categories are 1–11. Export floor is 0.001 and exporters may emit at most 300 predictions per image. Standard headline evaluation remains `maxDets=[1,10,100]`.

## Install

```bash
python -m pip install -r eval/requirements.txt
```

## Recommended validation sequence

```bash
python eval/selftest_gt.py --gt data/doclaynet/COCO/val.json --out results/eval-selftest.json
python eval/sanitize_predictions.py --gt data/doclaynet/COCO/val.json --pred results/yolo/val-raw.json --out results/yolo/val-sanitized.json --report-out results/yolo/val-sanitation.json
python eval/validate_predictions.py --gt data/doclaynet/COCO/val.json --pred results/yolo/val-sanitized.json --summary-out results/yolo/val-validation.json
python eval/coco_eval.py --gt data/doclaynet/COCO/val.json --pred results/yolo/val-sanitized.json --max-dets 1 10 100 --out results/yolo/val-metrics.json
```

Add `--image-ids eval/val_50_image_ids.json` to every applicable command when testing a subset. The evaluator explicitly assigns `params.imgIds`, preventing unprocessed pages from lowering recall.
