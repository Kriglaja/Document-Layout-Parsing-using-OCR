#!/usr/bin/env python3
from __future__ import annotations
import argparse, contextlib, io, json
from pathlib import Path
from common import load_json, selected_ids, write_json


def backend():
    try:
        from faster_coco_eval import COCO, COCOeval_faster as COCOeval
        return COCO, COCOeval, "faster-coco-eval"
    except ImportError:
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            return COCO, COCOeval, "pycocotools"
        except ImportError as e:
            raise SystemExit("Install evaluator: python -m pip install 'faster-coco-eval==1.8.0'") from e


def per_class(coco_eval, coco_gt, category_ids):
    precision = coco_eval.eval["precision"]  # T,R,K,A,M
    rows=[]
    for k, cid in enumerate(category_ids):
        name = coco_gt.loadCats([cid])[0]["name"]
        all_iou = precision[:, :, k, 0, -1]
        at50 = precision[0, :, k, 0, -1]
        valid = all_iou[all_iou > -1]; valid50 = at50[at50 > -1]
        rows.append({"category_id": cid, "category_name": name,
                     "ap50_95": float(valid.mean()) if valid.size else None,
                     "ap50": float(valid50.mean()) if valid50.size else None})
    return rows


def evaluate(gt: Path, pred: Path, image_ids_path: Path|None, max_dets: list[int]):
    COCO, COCOeval, backend_name = backend()
    coco_gt = COCO(str(gt))
    predictions = load_json(pred)
    if not predictions:
        raise SystemExit("Prediction list is empty; COCO loadRes cannot evaluate it")
    coco_dt = coco_gt.loadRes(str(pred))
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.params.maxDets = max_dets
    ids = selected_ids(image_ids_path)
    if ids is not None: ev.params.imgIds = ids
    ev.evaluate(); ev.accumulate()
    capture=io.StringIO()
    with contextlib.redirect_stdout(capture): ev.summarize()
    summary_text=capture.getvalue(); print(summary_text, end="")
    s=ev.stats
    metrics={"ap50_95":float(s[0]),"ap50":float(s[1]),"ap75":float(s[2]),"ap_small":float(s[3]),"ap_medium":float(s[4]),"ap_large":float(s[5]),
             f"ar_at_{max_dets[0]}":float(s[6]),f"ar_at_{max_dets[1]}":float(s[7]),f"ar_at_{max_dets[2]}":float(s[8]),
             "ar_small":float(s[9]),"ar_medium":float(s[10]),"ar_large":float(s[11])}
    cats=sorted(coco_gt.getCatIds())
    return {"backend":backend_name,"max_dets":max_dets,"image_count":len(ev.params.imgIds),"metrics":metrics,"per_class":per_class(ev,coco_gt,cats),"summary_text":summary_text}


def main():
    p=argparse.ArgumentParser(description="Shared COCO bbox evaluator for YOLO26s and RT-DETRv4-S.")
    p.add_argument("--gt",type=Path,required=True); p.add_argument("--pred",type=Path,required=True)
    p.add_argument("--image-ids",type=Path); p.add_argument("--max-dets",type=int,nargs=3,default=[1,10,100]); p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(); result=evaluate(a.gt,a.pred,a.image_ids,a.max_dets); write_json(a.out,result)
    print("Per-class AP:")
    for r in result["per_class"]: print(f'{r["category_id"]:2d} {r["category_name"]:18s} AP={r["ap50_95"]} AP50={r["ap50"]}')
    print(f"Metrics written to {a.out}")
if __name__=="__main__": main()
