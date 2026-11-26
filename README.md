# Document Layout Parsing using OCR

Comparison of YOLO26s and RT-DETRv4-S for document-layout detection on DocLayNet using a common evaluation protocol, followed by an OCR demonstration on detected document regions.

## Structure

TBD

## Dataset setup

To be added after the DocLayNet pipeline has been verified.

## RT-DETRv4-S

Setup, training, inference, and evaluation commands will be added here.

## YOLO26s

Setup, training, inference, and evaluation commands will be added here.

## Evaluation

Instructions for running the common evaluator will be added here.

## OCR demonstration

`python process_document.py INPUT... --output OUTPUT --formats png pdf json txt`

Inputs may be images, PDFs, or directories. JSON is always preserved. The default OCR runtime is the `doclaynet-ocr` Distrobox; RT-DETR inference runs on the host.
