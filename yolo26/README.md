# YOLO26-S DocLayNet

This directory contains the reproducible setup and inference plumbing for the
external YOLO26-S checkpoint used in the YOLO26-S versus RT-DETRv4-S comparison.

## Checkpoint

- Source repository: `hantian/yolo-doclaynet`
- File: `yolo26s-doclaynet.pt`
- Revision: `6728bcbd5cba8be72810d366cb3a1c9e4f893824`
- Dataset: DocLayNet
- Framework: Ultralytics YOLO

The checkpoint is not committed to Git. Download it with:

```bash
bash yolo26/download_checkpoint.sh