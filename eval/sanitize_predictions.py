#!/usr/bin/env python3
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
from common import image_dimensions, load_json, sha256, write_json


def main() -> None:
    p = argparse.ArgumentParser(description="Clip canonical COCO xywh predictions to original image bounds.")
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--pred", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--report-out", type=Path)
    args = p.parse_args()

    gt, predictions = load_json(args.gt), load_json(args.pred)
    dims = image_dimensions(gt)
    out, dropped = [], Counter()
    clipped = 0
    for i, p0 in enumerate(predictions):
        try:
            image_id = int(p0["image_id"]); category_id = int(p0["category_id"])
            x, y, w, h = map(float, p0["bbox"]); score = float(p0["score"])
            iw, ih = dims[image_id]
        except Exception as e:
            raise SystemExit(f"Malformed record {i}: {e}")
        x1, y1, x2, y2 = x, y, x + w, y + h
        cx1, cy1 = min(max(x1, 0.0), iw), min(max(y1, 0.0), ih)
        cx2, cy2 = min(max(x2, 0.0), iw), min(max(y2, 0.0), ih)
        if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
            clipped += 1
        nw, nh = cx2 - cx1, cy2 - cy1
        if nw <= 0 or nh <= 0:
            dropped[category_id] += 1
            continue
        out.append({"image_id": image_id, "category_id": category_id, "bbox": [cx1, cy1, nw, nh], "score": score})
    write_json(args.out, out)
    report = {
        "input": str(args.pred), "input_sha256": sha256(args.pred),
        "output": str(args.out), "output_sha256": sha256(args.out),
        "records_before": len(predictions), "records_after": len(out),
        "records_clipped": clipped, "records_dropped": sum(dropped.values()),
        "dropped_by_category": dict(sorted(dropped.items())),
    }
    if args.report_out:
        write_json(args.report_out, report)
    for k, v in report.items(): print(f"{k}: {v}")

if __name__ == "__main__": main()
