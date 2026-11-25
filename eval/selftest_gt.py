#!/usr/bin/env python3
from __future__ import annotations
import argparse, tempfile
from pathlib import Path
from common import load_json, selected_ids, write_json
from coco_eval import evaluate


def main():
    p=argparse.ArgumentParser(description="Evaluate ground truth as perfect predictions.")
    p.add_argument("--gt",type=Path,required=True); p.add_argument("--image-ids",type=Path); p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(); gt=load_json(a.gt); ids=selected_ids(a.image_ids); selected=set(ids) if ids is not None else {int(i["id"]) for i in gt["images"]}
    preds=[]
    for ann in gt["annotations"]:
        if int(ann["image_id"]) in selected and float(ann.get("iscrowd",0)) == 0:
            preds.append({"image_id":int(ann["image_id"]),"category_id":int(ann["category_id"]),"bbox":[float(x) for x in ann["bbox"]],"score":1.0})
    with tempfile.TemporaryDirectory() as td:
        pred=Path(td)/"perfect.json"; write_json(pred,preds)
        result=evaluate(a.gt,pred,a.image_ids,[1,10,100])
    write_json(a.out,result)
    score=result["metrics"]["ap50_95"]
    if score < 0.999:
        raise SystemExit(f"Self-test failed: AP50-95={score}, expected >=0.999")
    print(f"Self-test passed: AP50-95={score}")
if __name__=="__main__": main()
