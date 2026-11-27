# Document Layout Parsing using OCR

Comparison of **YOLO26s** and **RT-DETRv4-S** for document-layout detection on **DocLayNet** under a common evaluation protocol, followed by a detector-neutral OCR pipeline for arbitrary images and PDFs.

The repository serves two related purposes:

1. **Research comparison:** evaluate YOLO26s and RT-DETRv4-S using the same annotations, prediction schema, sanitation rules, and COCO evaluator.
2. **Document processing application:** detect semantic page regions, run Tesseract OCR on text-bearing regions, and export visual, searchable, editable, and machine-readable outputs.

## Features

- RT-DETRv4-S and YOLO26s detector backends.
- PNG, JPEG, TIFF, WebP, PDF, and directory inputs.
- Automatic CUDA use when available, with CPU fallback.
- Tesseract OCR over detected text-bearing regions.
- Original-image detector and OCR coordinates preserved throughout the pipeline.
- Spatial OCR reconstruction with token-level positioning.
- Searchable PDF, semantic DOCX/ODT, structured JSON, plain text, and PNG outputs.
- Shared COCO evaluation tooling for model-neutral comparison.

## Repository structure

```text
.
├── eval/                  # Shared prediction export, validation, and evaluation
├── ocr/                   # Detector backends, OCR pipeline, data models, exporters
│   ├── detectors/         # RT-DETRv4 and YOLO26 implementations
│   └── exporters/         # PNG, PDF, DOCX, ODT, JSON, and TXT exporters
├── rtdetrv4/              # RT-DETRv4 source, configs, checkpoints, and tools
├── scripts/               # Dataset audit, experiment, and utility scripts
├── yolo26/                # YOLO checkpoint setup and smoke-test utilities
├── process_document.py    # Main user-facing application
├── requirements.txt       # Application and evaluator dependencies
└── README.md
```

## Installation

### 1. Create a Python environment

Python 3.11 or 3.12 is the safest general choice. Install **PyTorch** and **Torchvision** first using the command appropriate for the operating system and CUDA version from the official PyTorch installation selector. CPU-only PyTorch works but detector inference will be slower.

Install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

The requirements include the application and evaluation dependencies, including Pillow, pytesseract, PyMuPDF, python-docx, odfpy, PyYAML, Ultralytics, and `faster-coco-eval==1.8.0`.

### 2. Install Tesseract

Tesseract is an external executable and must be available in `PATH`.

Ubuntu or Debian:

```bash
sudo apt install tesseract-ocr tesseract-ocr-eng
```

Fedora:

```bash
sudo dnf install tesseract tesseract-langpack-eng
```

macOS or Linux with Homebrew:

```bash
brew install tesseract
```

Windows 11:

1. Install Tesseract OCR.
2. Add its installation directory to `PATH`, or pass the executable explicitly:

```powershell
python .\process_document.py .\input.pdf `
  --output .\results\input `
  --tesseract-command "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

Verify the installation:

```bash
tesseract --version
tesseract --list-langs
```

The language list must contain `eng` for the default configuration.

## Model checkpoints

Model weights are local artifacts and are intentionally excluded from Git.

### YOLO26s

Download the pinned external DocLayNet checkpoint:

```bash
bash yolo26/download_checkpoint.sh
```

Expected location:

```text
yolo26/checkpoints/yolo26s-doclaynet.pt
```

The checkpoint must expose these classes at indices 0–10:

```text
Caption, Footnote, Formula, List-item, Page-footer, Page-header,
Picture, Section-header, Table, Text, Title
```

### RT-DETRv4-S

Place the selected fine-tuned checkpoint and runtime configuration at:

```text
rtdetrv4/checkpoints/best_stg1.pth
rtdetrv4/checkpoints/runtime-config.yml
```

`best_stg1.pth` is the validation-selected DocLayNet model used for inference. It is separate from `rtv4_hgnetv2_s_coco.pth`, the original COCO-pretrained training initializer.

The application disables pretrained-weight downloads during inference and explicitly restores the selected checkpoint, preferring EMA weights when present.

## Process a document

The root-level application is the supported user interface:

```bash
python process_document.py INPUT... \
  --output OUTPUT_DIRECTORY \
  --detector rtdetrv4 \
  --formats png pdf json txt
```

JSON is always generated as the normalized machine-readable document model, even when `json` is omitted from `--formats`.

### RT-DETRv4-S

```bash
python process_document.py report.pdf \
  --output results/report-rtdetr \
  --detector rtdetrv4 \
  --formats png pdf docx json txt \
  --device auto
```

### YOLO26s

```bash
python process_document.py report.pdf \
  --output results/report-yolo \
  --detector yolo26 \
  --formats png pdf docx json txt \
  --device auto
```

### Other input modes

Single image:

```bash
python process_document.py scan.png \
  --output results/scan \
  --formats png pdf json
```

Directory:

```bash
python process_document.py scans/ \
  --recursive \
  --output results/scans \
  --formats pdf json txt
```

Selected PDF pages:

```bash
python process_document.py report.pdf \
  --pages 1,3-5 \
  --pdf-dpi 200 \
  --output results/report-selection \
  --formats png pdf json
```

Custom output name for one logical document:

```bash
python process_document.py report.pdf \
  --output results/report \
  --name annual-report-ocr \
  --formats pdf docx json
```

## Application options

| Option | Default | Description |
|---|---:|---|
| `--detector` | `rtdetrv4` | `rtdetrv4` or `yolo26` |
| `--checkpoint` | backend-specific | Override the selected detector checkpoint |
| `--config` | RT-DETR runtime config | RT-DETR only |
| `--device` | `auto` | `auto`, `cuda`, or `cpu` |
| `--input-size` | `1024` | Detector inference size |
| `--detector-confidence` | `0.50` | Minimum detector confidence for application output |
| `--max-detections` | `300` | Maximum detections per page |
| `--lang` | `eng` | Tesseract language |
| `--psm` | `6` | Tesseract page segmentation mode per detected region |
| `--token-confidence` | `30` | Minimum Tesseract token confidence |
| `--pdf-dpi` | `200` | Rasterization resolution for PDF inputs |
| `--preserve-intermediates` | off | Retain temporary rasterized pages and resolved config |

The application threshold is distinct from evaluation export. COCO evaluation preserves detections down to `0.001`; the application defaults to `0.50` to avoid applying OCR to weak candidate regions.

## Output formats

For an input named `report.pdf`, the application can produce:

```text
OUTPUT_DIRECTORY/
├── report.json
├── report.txt
├── report.pdf
├── report.docx
├── report.odt
├── report-pages/
│   ├── report-page-0001/
│   │   ├── 01-layout.png
│   │   ├── 02-ocr-spatial.png
│   │   ├── 03-ocr-token-boxes.png
│   │   ├── 04-ocr-hybrid.png
│   │   └── 05-comparison.png
│   └── ...
└── run-config.json
```

- **PNG:** layout detection, spatial OCR, token boxes, hybrid reconstruction, and comparison views.
- **PDF:** original page image with an invisible searchable/selectable OCR text layer.
- **DOCX:** semantic, editable reconstruction using headings, paragraphs, and list items.
- **ODT:** semantic OpenDocument reconstruction.
- **JSON:** pages, semantic regions, scores, OCR tokens, confidences, and coordinates.
- **TXT:** simplified reading-order text.

Tables, pictures, and formulas are preserved as visual regions in the hybrid PNG output. Tesseract OCR is applied to captions, footnotes, list items, page headers and footers, section headers, text, and titles.

## DocLayNet dataset setup

The shared evaluation tools expect DocLayNet COCO annotations and images in this local layout:

```text
data/doclaynet/
├── COCO/
│   ├── train.json
│   ├── val.json
│   └── test.json
└── PNG/
    └── *.png
```

Dataset contents are excluded from Git. Utilities in `scripts/` support dataset audits, controlled subsets, and annotation visualization.

## Shared evaluation protocol

Both detectors export predictions in canonical COCO result format:

```json
[
  {
    "image_id": 123,
    "category_id": 10,
    "bbox": [100.0, 200.0, 300.0, 80.0],
    "score": 0.93
  }
]
```

The common contract is:

- original DocLayNet image IDs;
- canonical category IDs 1–11;
- absolute COCO `xywh` boxes in original-page pixels;
- scores in `[0,1]`;
- confidence floor `0.001`;
- no more than 300 exported predictions per image;
- the same sanitation and schema validation;
- `faster-coco-eval==1.8.0`;
- standard COCO `maxDets=[1,10,100]`;
- model-native frozen postprocessing: RT-DETR top-score selection and YOLO class-aware NMS with `iou=0.7` and `agnostic_nms=False`.

### Verify the evaluator

The ground-truth-as-predictions self-test should return AP50–95 of approximately 1.0:

```bash
python eval/selftest_gt.py \
  --gt data/doclaynet/COCO/val.json \
  --out results/evaluator/val-ground-truth-selftest.json
```

### Export and evaluate YOLO26s

A complete validation run performs canonical export, sanitation, schema validation, and shared COCO evaluation:

```bash
python eval/run_yolo_doclaynet_eval.py \
  --split val \
  --batch-size 8 \
  --device 0 \
  --iou 0.7 \
  --no-agnostic-nms
```

The YOLO exporter explicitly freezes `conf=0.001`, `iou=0.7`, `max_det=300`, and `agnostic_nms=False`. YOLO class indices 0–10 map to canonical DocLayNet category IDs 1–11.

If the selected batch size exceeds available VRAM, reduce `--batch-size`; batch size does not change the evaluation protocol.

Outputs are written to `results/yolo/`:

```text
val-raw.json
val-export-metadata.json
val-sanitized.json
val-sanitation.json
val-validation.json
val-metrics.json
```

### Evaluate existing canonical predictions

Sanitize:

```bash
python eval/sanitize_predictions.py \
  --gt data/doclaynet/COCO/val.json \
  --pred results/MODEL/val-raw.json \
  --out results/MODEL/val-sanitized.json \
  --report-out results/MODEL/val-sanitation.json
```

Validate:

```bash
python eval/validate_predictions.py \
  --gt data/doclaynet/COCO/val.json \
  --pred results/MODEL/val-sanitized.json \
  --summary-out results/MODEL/val-validation.json
```

Evaluate:

```bash
python eval/coco_eval.py \
  --gt data/doclaynet/COCO/val.json \
  --pred results/MODEL/val-sanitized.json \
  --max-dets 1 10 100 \
  --out results/MODEL/val-metrics.json
```

The evaluator reports AP50–95, AP50, AP75, scale-specific AP, AR@1, AR@10, AR@100, and per-class AP.

## Validation comparison

Both models were evaluated on all 6,489 DocLayNet validation pages using the shared protocol above.

| Metric | RT-DETRv4-S | YOLO26s | RT-DETR advantage |
|---|---:|---:|---:|
| AP50–95 | **0.8081** | 0.7527 | +0.0554 |
| AP50 | **0.9319** | 0.9027 | +0.0292 |
| AP75 | **0.8705** | 0.7868 | +0.0838 |
| AP small | **0.6354** | 0.6261 | +0.0093 |
| AP medium | **0.7126** | 0.6535 | +0.0591 |
| AP large | **0.8296** | 0.8078 | +0.0217 |
| AR@1 | **0.4045** | 0.3778 | +0.0266 |
| AR@10 | **0.8299** | 0.7645 | +0.0654 |
| AR@100 | **0.8896** | 0.8070 | +0.0827 |

RT-DETRv4-S leads on every aggregate validation metric. The largest AP difference is at IoU 0.75, where RT-DETRv4-S leads by 0.0838, indicating materially tighter localization. The small-object difference is comparatively narrow at 0.0093.

These are validation results used for development and model comparison. They are not final held-out test results.

## Test evaluation

Run the test split only after the protocol and reporting choices are frozen:

```bash
python eval/run_yolo_doclaynet_eval.py \
  --split test \
  --batch-size 8 \
  --device 0 \
  --iou 0.7 \
  --no-agnostic-nms
```

Apply the same fixed external protocol and RT-DETR-native postprocessing to RT-DETRv4-S test predictions. Do not tune thresholds, sanitation, category mappings, or evaluator settings after inspecting test results.

## Reproducibility and generated files

The following remain local and excluded from Git:

- DocLayNet data;
- Python virtual environments;
- downloaded `.pt` and `.pth` checkpoints;
- generated prediction arrays and evaluation outputs;
- application outputs under `results/`;
- Python bytecode caches.

Commit compact provenance files, final metric summaries, and selected figures only when deliberately placed in a versioned documentation or artifact directory.

## Known limitations

- OCR quality depends on Tesseract and source resolution.
- Tables and formulas are preserved visually rather than structurally parsed.
- DOCX and ODT are semantic reflows, not pixel-faithful reproductions.
- The searchable PDF preserves the original page image and adds an invisible OCR text layer.
- Reading order is geometry-based and may be imperfect on complex multi-column pages.
- Detector accuracy outside the DocLayNet domain depends on document similarity and image quality.

## License

See `LICENSE` for repository licensing information. External model checkpoints and datasets remain subject to their own licenses and terms.
