# Document Layout Parsing using OCR

Comparison of **YOLO26s** and **RT-DETRv4-S** for document-layout detection on **DocLayNet** under a common evaluation protocol, followed by a detector-neutral OCR pipeline for arbitrary images and PDFs.

The repository serves two related purposes:

1. **Research comparison:** evaluate YOLO26s and RT-DETRv4-S using the same DocLayNet annotations, prediction schema, sanitation rules, and COCO evaluator.
2. **Document processing application:** detect semantic page regions, run Tesseract OCR on text-bearing regions, and export visual, searchable, editable, and machine-readable outputs.

## Features

- RT-DETRv4-S and YOLO26s detector backends.
- Arbitrary PNG, JPEG, TIFF, WebP, and PDF inputs.
- Automatic CUDA use when available, with CPU fallback.
- Tesseract OCR applied to detected text-bearing regions.
- Original-image detector coordinates preserved throughout the pipeline.
- Spatial OCR reconstruction with token-level positioning.
- Searchable PDF, semantic DOCX/ODT, structured JSON, plain text, and PNG visualizations.
- Shared COCO evaluation tooling for model-neutral comparison.
- Reproducible scripts for RT-DETR training and prediction export.

## Repository structure

```text
.
├── data/                  # Local DocLayNet data; not committed
├── eval/                  # Shared prediction validation and COCO evaluation
├── ocr/                   # Detector backends, OCR pipeline, data models, exporters
│   ├── detectors/         # RT-DETRv4 and YOLO26 implementations
│   └── exporters/         # PNG, PDF, DOCX, ODT, JSON, and TXT outputs
├── rtdetrv4/              # RT-DETRv4 source, configs, checkpoints, and tools
├── scripts/               # Dataset audit, experiment and export scripts
├── yolo26/                # YOLO checkpoint setup and smoke-test utilities
├── process_document.py    # Main user-facing application
├── requirements.txt       # Application dependencies except PyTorch/Torchvision
└── README.md
```

## Installation

### 1. Create or activate a Python environment

The application uses one environment for detector inference, OCR integration, and document export. Python 3.11 or 3.12 is the safest general choice. The project has also been tested locally with Python 3.14 and CUDA-enabled PyTorch.

Install **PyTorch** and **Torchvision** first using the command appropriate for the operating system and CUDA version from the official PyTorch installation selector. CPU-only PyTorch also works but detector inference will be slower.

Install the remaining application dependencies:

```bash
python -m pip install -r requirements.txt
```

The requirements include:

- Pillow
- pytesseract
- PyMuPDF
- python-docx
- odfpy
- PyYAML
- ultralytics

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

macOS or Aurora through Homebrew:

```bash
brew install tesseract
```

Windows 11:

1. Install Tesseract OCR.
2. Add the installation directory to `PATH`, or provide the executable explicitly:

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

The checkpoint should expose these 11 classes in indices 0–10:

```text
Caption, Footnote, Formula, List-item, Page-footer, Page-header,
Picture, Section-header, Table, Text, Title
```

### RT-DETRv4-S

Place the selected DocLayNet checkpoint and its runtime configuration at:

```text
rtdetrv4/checkpoints/best_stg1.pth
rtdetrv4/checkpoints/runtime-config.yml
```

TODO: add link to rtdetrv4s checkpoint later
`best_stg1.pth` is the validation-selected fine-tuned model used for inference. It is separate from `rtv4_hgnetv2_s_coco.pth`, which is the original COCO-pretrained training initializer.

The application disables model-pretraining downloads during inference and explicitly restores the selected checkpoint, preferring EMA weights when present.

## Process a document

The root-level application is the supported user interface:

```bash
python process_document.py INPUT... \
  --output OUTPUT_DIRECTORY \
  --detector rtdetrv4 \
  --formats png pdf json txt
```

JSON is always generated as the normalized machine-readable document model, even when `json` is omitted from `--formats`.

### RT-DETRv4-S example

```bash
python process_document.py report.pdf \
  --output results/report-rtdetr \
  --detector rtdetrv4 \
  --formats png pdf docx json txt \
  --device auto
```

### YOLO26s example

```bash
python process_document.py report.pdf \
  --output results/report-yolo \
  --detector yolo26 \
  --formats png pdf docx json txt \
  --device auto
```

### Image input

```bash
python process_document.py scan.png \
  --output results/scan \
  --formats png pdf json
```

### Directory input

```bash
python process_document.py scans/ \
  --recursive \
  --output results/scans \
  --formats pdf json txt
```

### Selected PDF pages

```bash
python process_document.py report.pdf \
  --pages 1,3-5 \
  --pdf-dpi 200 \
  --output results/report-selection \
  --formats png pdf json
```

### Custom output name

For one logical document:

```bash
python process_document.py report.pdf \
  --output results/report \
  --name annual-report-ocr \
  --formats pdf docx json
```

## Application options

Important options:

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

The application confidence threshold is distinct from evaluation export. COCO evaluation should preserve low-confidence detections, typically down to `0.001`, while the application defaults to `0.50` to avoid OCR processing weak candidate regions.

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

Format behavior:

- **PNG:** project visualizations showing layout detection, spatial OCR, token boxes, and hybrid reconstruction.
- **PDF:** original page image with an invisible searchable/selectable OCR text layer.
- **DOCX:** semantic, editable reconstruction using headings, paragraphs, and list items.
- **ODT:** semantic OpenDocument reconstruction.
- **JSON:** pages, semantic regions, scores, OCR tokens, confidences, and coordinates.
- **TXT:** simplified reading-order text.

Tables, pictures, and formulas are treated as visual regions in the hybrid PNG output. Ordinary Tesseract OCR is applied to captions, footnotes, list items, page headers and footers, section headers, text, and titles.

## DocLayNet dataset setup

The shared evaluation tools expect the canonical DocLayNet COCO annotations and images. A typical local layout is:

```text
data/doclaynet/
├── COCO/
│   ├── train.json
│   ├── val.json
│   └── test.json
└── PNG/
    └── *.png
```

Dataset contents are excluded from Git. Use the scripts in `scripts/` to audit the dataset, inspect class distributions, create controlled subsets, and visualize annotations before training or evaluation.

Useful utilities include:

```text
scripts/audit_doclaynet.py
scripts/create_doclaynet_overfit_subset.py
scripts/visualize_doclaynet.py
```

## YOLO26s setup

YOLO uses the immutable external checkpoint pinned by revision in:

```text
yolo26/download_checkpoint.sh
```

Basic checkpoint validation:

```bash
python - <<'PY'
from ultralytics import YOLO

model = YOLO("yolo26/checkpoints/yolo26s-doclaynet.pt")
print(model.names)
PY
```

## Shared evaluation

Both detectors must export predictions in canonical COCO result format:

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

The contract is:

- original DocLayNet image IDs;
- canonical category IDs 1–11;
- absolute COCO `xywh` boxes in original-page pixels;
- scores in `[0,1]`;
- no more than 300 exported predictions per image;
- development and protocol validation on the validation split;
- one frozen evaluation on the test split.

### Install the evaluator

```bash
python -m pip install -r eval/requirements.txt
```

### Verify the evaluator

The ground-truth-as-predictions self-test should return AP50–95 of approximately 1.0:

```bash
python eval/selftest_gt.py \
  --gt data/doclaynet/COCO/val.json \
  --out results/evaluator/val-ground-truth-selftest.json
```

### Sanitize predictions

```bash
python eval/sanitize_predictions.py \
  --gt data/doclaynet/COCO/val.json \
  --pred results/MODEL/val-raw.json \
  --out results/MODEL/val-sanitized.json \
  --report-out results/MODEL/val-sanitation.json
```

Sanitation clips boxes to original image bounds and drops boxes with no positive area after clipping.

### Validate the prediction schema

```bash
python eval/validate_predictions.py \
  --gt data/doclaynet/COCO/val.json \
  --pred results/MODEL/val-sanitized.json \
  --summary-out results/MODEL/val-validation.json
```

### Run common COCO evaluation

```bash
python eval/coco_eval.py \
  --gt data/doclaynet/COCO/val.json \
  --pred results/MODEL/val-sanitized.json \
  --max-dets 1 10 100 \
  --out results/MODEL/val-metrics.json
```

The evaluator reports:

- AP50–95, AP50, and AP75;
- AP for small, medium, and large objects;
- AR@1, AR@10, and AR@100;
- per-class AP50–95 and AP50.

Use the same ground truth, sanitation, evaluator version, score floor, image IDs, category mapping, and `maxDets` policy for both detectors.

## Known limitations

- OCR quality is determined by the off-the-shelf Tesseract engine and source resolution.
- Tables and formulas are preserved as visual regions rather than structurally parsed.
- DOCX and ODT are semantic reflows, not pixel-faithful reproductions.
- The searchable PDF preserves the original page image and adds an invisible OCR text layer; it is not the same representation as the semantic DOCX/ODT output.
- Reading order is geometry-based and may be imperfect on highly complex multi-column pages.
- Detector accuracy outside the DocLayNet document domain depends on document similarity and image quality.

## License

See `LICENSE` for repository licensing information. External model checkpoints and datasets remain subject to their own licenses and terms.
